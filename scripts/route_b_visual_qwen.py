"""Route B: end-to-end visual features -> projection -> frozen Qwen -> Chinese.

Leverages the professor's "hand hidden states to a large model" idea: camera
RGB/motion/landmark cached features are projected into Qwen's embedding space
as a prefix; the frozen Qwen then autoregressively produces the sentence.
Only the lightweight visual projector is trained (teacher forcing).

This is the true end-to-end route (no gold-gloss oracle): the model must
encode the visual signal itself into language.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch import nn

REPO = "/home/su127/FYP/domain-bounded-cslr"
if REPO not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))
from cslr.translation.metrics import bleu, chrf, rouge_l  # noqa: E402

MODALITIES = ("rgb", "motion", "landmark")

# Fixed instruction appended after the visual prefix so the frozen LLM always
# sees the translation task (LLaVA-style setup); identical in train & generate.
INSTRUCTION = "请根据手语动作，翻译成通顺的中文句子："


class VisualProjector(nn.Module):
    def __init__(
        self,
        feature_dim: dict[str, int],
        hidden: int,
        qwen_hidden: int,
        token_per_mod: int = 16,
    ) -> None:
        super().__init__()
        self.token_per_mod = token_per_mod
        self.per_mod = nn.ModuleDict()
        for mod in MODALITIES:
            d = feature_dim[mod]
            self.per_mod[mod] = nn.Sequential(
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
        self.pool = nn.AdaptiveAvgPool1d(token_per_mod)
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, qwen_hidden)
        # keep the injected embeddings at a healthy scale
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> torch.Tensor:
        toks: list[torch.Tensor] = []
        for mod in MODALITIES:
            x = self.per_mod[mod](feats[mod])  # [B,T,h]
            x = self.pool(x.transpose(1, 2)).transpose(1, 2)  # [B,K,h]
            toks.append(x)
        x = torch.cat(toks, dim=1)  # [B, K*3, h]
        x = self.norm(x)
        return self.out(x)  # [B, K*3, qwen_hidden]


def build_pairs(label_csv: str, feature_root: str) -> list[dict]:
    import csv
    pairs = []
    with open(label_csv, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            sid = (r.get("Number") or "").strip()
            chinese = (r.get("Chinese Sentences") or "").strip()
            fc = os.path.join(feature_root, sid)
            if not sid or not chinese:
                continue
            if all(os.path.exists(f"{fc}.{m}.npy") for m in MODALITIES):
                pairs.append({"sample_id": sid, "reference": chinese})
    return pairs


def load_feats(feature_root: str, sid: str) -> dict[str, np.ndarray]:
    return {m: np.load(f"{feature_root}/{sid}.{m}.npy").astype(np.float32) for m in MODALITIES}


def train(
    pairs, feature_root, model_path, *, qwen_hidden, device, epochs, lr,
    batch_size, max_len, token_per_mod,
) -> nn.Module:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.float16
    ).to(device).eval()
    for p in qwen.parameters():
        p.requires_grad_(False)

    first = load_feats(feature_root, pairs[0]["sample_id"])
    feature_dim = {m: first[m].shape[1] for m in MODALITIES}
    proj = VisualProjector(
        feature_dim, hidden=qwen_hidden, qwen_hidden=qwen_hidden, token_per_mod=token_per_mod
    ).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr)

    emb = qwen.get_input_embeddings()
    losses: list[float] = []
    step = 0
    for epoch in range(epochs):
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            feats = [load_feats(feature_root, p["sample_id"]) for p in chunk]
            ft: dict[str, torch.Tensor] = {}
            for mod in MODALITIES:
                arrs = [torch.from_numpy(f[mod]) for f in feats]
                max_t = max(a.shape[0] for a in arrs)
                pad = torch.zeros(max_t, arrs[0].shape[1], dtype=arrs[0].dtype)
                stacked = torch.stack([torch.cat([a, pad[: max_t - a.shape[0]]]) for a in arrs])
                ft[mod] = stacked.to(device)  # fp32 for projector
            masks = {m: torch.ones(ft[m].shape[:2], dtype=torch.bool).to(device) for m in MODALITIES}

            refs = [p["reference"] for p in chunk]
            inst_enc = tok(INSTRUCTION, return_tensors="pt", add_special_tokens=False)
            inst_ids = inst_enc.input_ids.to(device)  # [1, IL]
            inst_len = inst_ids.shape[1]
            enc = tok(refs, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(device)
            in_ids = enc.input_ids[:, :-1]  # [B, TL-1]
            lab = enc.input_ids[:, 1:]       # [B, TL-1]
            text_len = in_ids.shape[1]

            vis_prefix = proj(ft, masks).half()  # [B, K*3, qwen_hidden]
            inst_emb = emb(inst_ids.expand(in_ids.shape[0], -1))  # [B, IL, qh]
            text_emb = emb(in_ids)  # [B, TL-1, qh]
            inputs_embeds = torch.cat([vis_prefix, inst_emb, text_emb], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long).to(device)

            opt.zero_grad()
            logits = qwen(inputs_embeds=inputs_embeds, attention_mask=attn).logits  # [B, tot, V]
            # local offset of the text block: vis + inst
            text_start = vis_prefix.shape[1] + inst_len
            text_logits = logits[:, text_start: text_start + text_len, :]
            loss_ce = nn.functional.cross_entropy(
                text_logits.reshape(-1, qwen.config.vocab_size),
                lab.reshape(-1),
            )
            loss_ce.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), max_norm=5.0)
            opt.step()
            losses.append(float(loss_ce.item()))
            step += 1
            if step % 20 == 0:
                print(f"[B-train] ep{epoch} step{step} loss={loss_ce.item():.4f}", flush=True)
    print(f"[B-train] done epochs={epochs} steps={step} loss_start={losses[0]:.4f} loss_end={losses[-1]:.4f}", flush=True)
    return proj, qwen, tok


def manual_generate(qwen, tok, prefix_emb: torch.Tensor, max_new: int = 48) -> str:
    """Autoregressive decode with a fixed embedding prefix (visual+instruction).

    transformers' generate() mishandles inputs_embeds-only prefixes (it treated
    the prefix length as 0 in our probe), so we decode step by step: the first
    step conditions on the visual-prefix embeddings, then we continue with real
    token ids using KV cache.
    """
    past = None
    gen_ids: list[int] = []
    # first step: visual + instruction prefix as embeddings
    out = qwen(inputs_embeds=prefix_emb, use_cache=True)
    logits = out.logits[:, -1, :]  # [1, V]
    past = out.past_key_values
    nxt = int(torch.argmax(logits, dim=-1))
    gen_ids.append(nxt)
    for _ in range(max_new - 1):
        inp = torch.tensor([[nxt]], device=prefix_emb.device)
        out = qwen(input_ids=inp, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        nxt = int(torch.argmax(logits, dim=-1))
        gen_ids.append(nxt)
        if nxt == tok.eos_token_id:
            break
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def generate(proj, qwen, tok, feats, device, token_per_mod, max_new=48) -> str:
    proj.eval()
    ft = {m: torch.from_numpy(feats[m]).unsqueeze(0).to(device) for m in MODALITIES}
    vis_prefix = proj(ft, masks=ft).half()  # [1, Vtok, qh]
    inst_enc = tok(INSTRUCTION, return_tensors="pt", add_special_tokens=False)
    inst_emb = qwen.get_input_embeddings()(inst_enc.input_ids.to(device))  # [1, IL, qh]
    prefix = torch.cat([vis_prefix, inst_emb], dim=1)
    return manual_generate(qwen, tok, prefix, max_new=max_new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--feature-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--token-per-mod", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pairs = build_pairs(args.label, args.feature_root)
    if args.limit and args.limit < len(pairs):
        pairs = pairs[:args.limit]
    train_pairs = pairs[:int(0.8 * len(pairs))]
    eval_pairs = pairs[int(0.8 * len(pairs)):]

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    qwen_hidden = m.config.hidden_size
    del m
    import torch
    torch.cuda.empty_cache()

    proj, qwen, tok = train(
        train_pairs, args.feature_root, args.model,
        qwen_hidden=qwen_hidden, device=args.device,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        max_len=args.max_len, token_per_mod=args.token_per_mod,
    )

    records = []
    for p in eval_pairs[:args.eval_n]:
        feats = load_feats(args.feature_root, p["sample_id"])
        pred = generate(proj, qwen, tok, feats, args.device, args.token_per_mod)
        ref = p["reference"]
        records.append({
            "sample_id": p["sample_id"], "reference": ref, "prediction": pred,
            "bleu_1": round(bleu(ref, pred, 1), 4), "bleu_2": round(bleu(ref, pred, 2), 4),
            "rouge_l": round(rouge_l(ref, pred), 4), "chrf": round(chrf(ref, pred), 4),
            "exact_match": float(ref == pred),
        })
        if (len(records)) % 20 == 0:
            print(f"[B-eval] {len(records)}/{min(args.eval_n, len(eval_pairs))}", flush=True)

    preds = [r["prediction"] for r in records]
    cnt = Counter(preds)
    summ = {
        "samples": len(records), "distinct": len(cnt),
        "collapsed": len(cnt) <= 2,
        "top_prediction_share": round(cnt.most_common(1)[0][1] / max(1, len(records)), 3),
        "bleu_1": round(sum(r["bleu_1"] for r in records) / max(1, len(records)), 4),
        "bleu_2": round(sum(r["bleu_2"] for r in records) / max(1, len(records)), 4),
        "rouge_l": round(sum(r["rouge_l"] for r in records) / max(1, len(records)), 4),
        "chrf": round(sum(r["chrf"] for r in records) / max(1, len(records)), 4),
        "exact_match": round(sum(r["exact_match"] for r in records) / max(1, len(records)), 4),
        "uses_external_weights": True, "test_split_read": False,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summ, "per_sample": records}, fh, ensure_ascii=False, indent=2)
    print("=== SUMMARY ===")
    print(json.dumps(summ, ensure_ascii=False, indent=2))
    for r in records[:12]:
        print(f"  {r['sample_id']} | ref={r['reference']!r} -> {r['prediction']!r}")


if __name__ == "__main__":
    main()