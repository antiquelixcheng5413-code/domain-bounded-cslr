"""TFNet 训练脚本（官方超参+ 我们的特征）

设计要点（全部来自今天踩过的坑）：
  1. **特征路径可切换**：新特征（Tasks API，presence 已修）/ 旧特征（离线 holistic）
  2. **词表可扩**：min_frequency=1, max_tokens=None -> 3516
  3. **口径走官方**：用official_wer.evaluate()，主指标 WER_official
  4. **跑满 epoch**：不用 early stop（P58 已证「最佳 ep< patience」= 伪造）
  5. **每10 epoch 存 ckpt**（用户明确要求）
  6. **lr 按官方 35/45 降 80%**（若 epochs != 55 则按比例缩放）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from torch import nn                                           # noqa: E402

from cslr.recognition.gloss_sequence import (                 # noqa: E402
    build_ordered_vocabulary, split_gloss_sequence, GlossSequenceConfig)
from official_tfnet import TFNet, OFFICIAL_CONFIG            # noqa: E402
from official_wer import evaluate, position_vs_official       # noqa: E402
from cslr.recognition.training import decode_batch             # noqa: E402


def read_csv(p):
    import csv
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}


class Batcher:
    """按 gloss 数切目标（+1 因为 CTC 类 0 是 blank，P23 的 off-by-one 修复）。"""

    def __init__(self, root: Path, split: str):
        self.root = root / split

    def items(self, labels, voc, max_len=24):
        out = []
        for sid, raw in labels.items():
            p = self.root / (sid + ".landmark.npy")
            if not p.exists():
                continue
            ids = voc.encode(raw)
            if not ids or len(ids) > max_len:
                continue
            out.append({"sid": sid, "raw": raw,
                        "ids": [i + 1 for i in ids],
                        "feat": np.load(p).astype(np.float32)})
        return out


def collate(batch, device):
    n = len(batch)
    T = max(x["feat"].shape[0] for x in batch)
    F = batch[0]["feat"].shape[1]
    L = max(len(x["ids"]) for x in batch)
    lm = torch.zeros(n, T, F)
    tg = torch.zeros(n, L, dtype=torch.long)
    il = torch.zeros(n, dtype=torch.long)
    tl = torch.zeros(n, dtype=torch.long)
    for i, x in enumerate(batch):
        t, f = x["feat"].shape
        lm[i, :t] = torch.from_numpy(x["feat"])
        l = len(x["ids"])
        tg[i, :l] = torch.tensor(x["ids"], dtype=torch.long)
        il[i] = t
        tl[i] = l
    return (lm.to(device), tg.to(device), il.to(device), tl.to(device))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-root", default=str(
        REPO / "artifacts/part3_features_tasksapi"))
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = 不截断（3516）")
    ap.add_argument("--min-frequency", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=55,
                    help="官方 55；调试可用更少")
    ap.add_argument("--batch", type=int, default=16,
                    help="官方 2；landmark 特征显存小，可放大")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--w-vae", type=float, default=0.1,
                    help="论文未给权重，默认 0.1")
    ap.add_argument("--no-vae", action="store_true")
    ap.add_argument("--temporal-jitter", type=float, default=0.2,
                    help="官方时序增强 ±20%%；0 = 关闭")
    ap.add_argument("--tag", default="p72")
    ap.add_argument("--max-train", type=int, default=0)
    a = ap.parse_args()

    cfg = GlossSequenceConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")

    mt = None if a.max_tokens in (0, -1) else a.max_tokens
    voc, _ = build_ordered_vocabulary(lab_tr.values(),
                                      min_frequency=a.min_frequency,
                                      max_tokens=mt)
    print("=" * 68)
    print("TFNet 训练（官方超参 + 我们的 landmark 特征）")
    print("=" * 68)
    print("device          %s" % dev)
    print("特征根          %s" % a.feature_root)
    print("词表            %d（CTC 类数 = %d，含 blank）"
          % (voc.size - 1, voc.size))
    print("官方配置        lr=%g wd=%g epochs=%d batch=%d"
          % (OFFICIAL_CONFIG["lr"], OFFICIAL_CONFIG["weight_decay"],
             OFFICIAL_CONFIG["epochs"], OFFICIAL_CONFIG["batch_size"]))
    print("我们的配置      lr=%g wd=%g epochs=%d batch=%d"
          % (a.lr, a.weight_decay, a.epochs, a.batch))

    tr = Batcher(Path(a.feature_root), "train").items(lab_tr, voc)
    dv = Batcher(Path(a.feature_root), "validation").items(lab_dv, voc)
    if a.max_train:
        tr = tr[:a.max_train]
    print("\ntrain %d 条  dev %d 条" % (len(tr), len(dv)))
    if len(tr) < 10 or len(dv) < 10:
        print("❌ 样本不足，先跑 p53 提取特征")
        return

    net = TFNet(feat_dim=368, hidden=a.hidden, vocab=int(voc.size),
                dropout=0.3, use_vae=not a.no_vae).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print("参数 %.2f M" % (n_par / 1e6))

    opt = torch.optim.Adam(net.parameters(), lr=a.lr,
                           weight_decay=a.weight_decay)
    # 官方：35/45 降 80%。epochs != 55 时按比例缩放
    ratio = a.epochs / OFFICIAL_CONFIG["epochs"]
    ms = [max(1, int(round(OFFICIAL_CONFIG["lr_decay_epochs"][0] * ratio))),
          max(2, int(round(OFFICIAL_CONFIG["lr_decay_epochs"][1] * ratio)))]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms,
                                                 gamma=0.2)
    print("lr衰减里程碑     %s（官方 35/45 按比例缩放）" % ms)

    ck = REPO / "artifacts/checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    hist = []
    best = 1e9
    t0 = time.time()

    for ep in range(1, a.epochs + 1):
        net.train()
        rng = np.random.RandomState(ep)
        idx = list(range(len(tr)))
        rng.shuffle(idx)
        tot = n = 0
        for k in range(0, len(idx), a.batch):
            ch = [tr[i] for i in idx[k:k + a.batch]]
            if a.temporal_jitter > 0:
                # 官方 ±20% 时序增强：对已抽好的特征做长度抖动
                for x in ch:
                    scale = 1.0 + rng.uniform(-a.temporal_jitter,
                                              a.temporal_jitter)
                    f = x["feat"]
                    T2 = max(8, int(round(f.shape[0] * scale)))
                    src = np.linspace(0, f.shape[0] - 1, T2)
                    x["feat"] = np.stack(
                        [f[int(round(s0))] for s0 in src]).astype(np.float32)
            lm, tg, il, tl = collate(ch, dev)
            logits, aux = net(lm)
            # targets 需 flatten +去掉 blank 占位 0（官方未说明，按标准做法）
            tgm = torch.cat([tg[i, :tl[i]] for i in range(len(ch))])
            loss, ctc, vae = TFNet.loss(
                logits, aux, tgm, il, tl, w_ctc=1.0, w_vae=a.w_vae)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            n += 1
        sched.step()

        # ---- 评估（官方口径）----
        net.eval()
        refs_raw, hyps = [], []
        with torch.no_grad():
            for k in range(0, len(dv), a.batch):
                ch = dv[k:k + a.batch]
                lm, tg, il, tl = collate(ch, dev)
                lg, _ = net(lm)
                lp = lg.log_softmax(-1).cpu().numpy()
                ol = np.full((len(ch),), lg.size(1), dtype=np.int64)
                dec, _, _ = decode_batch(lp, ol, 1)
                for r, s in zip(ch, dec):
                    refs_raw.append(r["raw"])
                    hyps.append(voc.decode(list(s)))
        res = evaluate(refs_raw, hyps)
        w = res["WER_official"]
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1),
                     **{k: v for k, v in res.items()
                        if not k.startswith("_")}})
        print("  ep%03d loss %.4f  **WER_official %.2f%%**  %s"
              % (ep, tot / max(n, 1), w,
                 ("比SOTA 好 %.2fpp" % -res["gap_to_sota_TFNet"])
                 if res["gap_to_sota_TFNet"] > 0 else
                 ("差SOTA %.2fpp" % -res["gap_to_sota_TFNet"])),
              flush=True)

        if ep % 10 == 0 or ep == a.epochs:
            torch.save({"epoch": ep, "model_state": net.state_dict(),
                        "vocab_size": int(voc.size),
                        # 🔴 写进 ckpt，让服务端能自动对齐词表（P75 E1）
                        "vocab_params": {"min_frequency": a.min_frequency,
                                         "max_tokens": mt},
                        "feat_dim": 368, "hidden": a.hidden,
                        "use_vae": not a.no_vae,
                        "official_config": OFFICIAL_CONFIG,
                        "wer_official": w,
                        "dev_wer": w},
                       ck / ("%s-tfnet-ep%03d.pt" % (a.tag, ep)))
        if w < best:
            best = w
            torch.save({"epoch": ep, "model_state": net.state_dict(),
                        "vocab_size": int(voc.size),
                        "vocab_params": {"min_frequency": a.min_frequency,
                                         "max_tokens": mt},
                        "feat_dim": 368, "hidden": a.hidden,
                        "use_vae": not a.no_vae,
                        "official_config": OFFICIAL_CONFIG,
                        "wer_official": w, "dev_wer": w,
                        "notes": "TFNet 复刻；判优用官方 WER 口径"},
                       ck / ("%s-tfnet-best.pt" % a.tag))

    out = REPO / "artifacts/metrics/blank-gov/%s-tfnet.json" % a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "TFNet (复刻 arXiv:2409.11960v2 时频双域)",
        "feature_root": a.feature_root,
        "vocab": int(voc.size - 1), "ctc_classes": int(voc.size),
        "params_M": round(n_par / 1e6, 2),
        "config": {"lr": a.lr, "wd": a.weight_decay, "epochs": a.epochs,
                   "batch": a.batch, "w_vae": a.w_vae,
                   "temporal_jitter": a.temporal_jitter,
                   "lr_milestones": ms},
        "official_config": OFFICIAL_CONFIG,
        "official_benchmark_dev": {"TFNet": 42.1, "MAM-FSD": 44.9,
                                    "VAC": 45.1, "SEN": 46.5,
                                    "CorrNet": 47.2, "MSTNet": 54.4},
        "best_wer_official": round(best, 2),
        "baseline_p42_wer_official": 52.11,
        "improvement_vs_p42_pp": round(52.11 - best, 2),
        "minutes": round((time.time() - t0) / 60, 1),
        "history": hist,
        "replication_gaps": [
            "F_f 用 landmark 投影替代 MAM-FSD RGB backbone",
            "DFT 取实部（论文未说明具体形式）",
            "VAE 结构为标准实现（论文未给细节）",
            "L_CTC : L_VAE 权重 = 1.0 : 0.1（论文未给）",
            "裁剪/翻转对已抽好的 368 维特征不适用，只有时序抖动生效",
        ],
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("\n最佳 WER_official %.2f%%（P42 是 52.11%%，改善 %.2f pp）"
          % (best, 52.11 - best))
    print("收据 -> %s" % out)


if __name__ == "__main__":
    main()