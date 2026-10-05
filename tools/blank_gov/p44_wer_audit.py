# -*- coding: utf-8 -*-
"""P44 · 审计 dev WER=0.5211 的真实含义

## 为什么要做

用户质疑：「WER 真的有 0.5211？我测出来怎么感觉百分之八十都是 unk」。

这是对**指标口径**的质疑，不是对模型好坏的质疑。
本脚本把 0.5211 拆开，回答：
  「这 0.5211 里，多少是模型真认错了，多少是被口径送分的？」

## 三个口径陷阱（假设，待验证）

1. **折叠送分**：参考用 `folded = voc.decode(voc.encode(label))`，
   所有 OOV 折叠成同一个 `<unk>`。dev 有约 30% token 是 OOV（P9），
   全变成同一个符号 —— 模型吐一个 `<unk>` 就算命中，WER 被系统性低估。
2. **`<unk>` 是可预测的通配符**：若模型学会「不会就吐 unk」，
   而参考侧 unk 又极多，则 unk 命中率高但输出无信息量。
3. **样本级与 token 级给人的印象相反**：
   精确匹配 18/514 = 3.5%，但 WER 0.52 听起来像「对了一半」。

## 输出判据

- ref / hyp 两侧的 `<unk>` 占比（验证用户的直觉）
- 折叠 WER vs 严格 WER（差值 = 折叠送了多少分）
- 剔除 OOV 的「纯词表内 WER」（真正考识别能力）
- `<unk>` 作为通配符的命中率 vs 其他词
- 样本级 exact / partial / wrong

只读 dev，不训练，不碰 test。
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.dataset import FeatureNormalizer                # noqa: E402
from cslr.recognition.training import decode_batch                    # noqa: E402
# ⚠️ 必须用 p40 的**本地** levenshtein（第 279 行，返回标量）。
# p8_error_attribution 里那个同名函数返回 (dist, sub, ins, del) 四元组，
# 累加会 TypeError。P40/P42 的 WER 用的是本地版，所以口径要跟它一致。
from p40_rgb_main import levenshtein                                # noqa: E402
from p40_rgb_main import DualInputCTC, read_csv                       # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM_FEAT = REPO / "artifacts/part3_features"
UNK = "<unk>"


def read_csv(p: Path) -> dict:
    with open(p, newline="", encoding="utf-8") as f:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(f)}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        REPO / "artifacts/metrics/blank-gov/p44-wer-audit.json"))
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device, flush=True)

    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)
    # 注意：GlossSequenceVocabulary 没有 token_to_id 属性，
    # 用 `token in voc`（走 __contains__）判断是否属于词表。
    # ⚠️ `<unk>` 本身在词表里（tokens[0]），所以「属于词表」要用
    #    `t != UNK` 再排除一次，否则 OOV 会被误算成「已学词」。
    t2i = set(voc.tokens)
    print("vocab size = %d | <unk> in vocab = %s | id = %s"
          % (voc.size, UNK in voc.tokens,
             voc.tokens.index(UNK) if UNK in voc.tokens else None), flush=True)

    # ---- 构造 dev 样本（与 p40 的 make() 同口径，但不需要 RGB）----
    items = []
    for sid in sorted(lab_dv):
        raw = lab_dv.get(sid, "")
        toks = [t.strip() for t in raw.split("/") if t.strip()]
        if not toks:
            continue
        lf = LM_FEAT / "validation" / (sid + ".landmark.npy")
        if not lf.exists():
            continue
        L = np.load(lf).astype(np.float32)
        ids = voc.encode(raw)
        if not ids or len(ids) > 24:            # (48+1)//2
            continue
        items.append({
            "sid": sid, "lm": L,
            "target": [i + 1 for i in ids],
            "toks": toks,                        # 原始 gloss（含真实 OOV）
            "folded": voc.decode(list(ids)),     # 折叠后（训练/服务用的参考）
        })
    print("dev 样本 = %d" % len(items), flush=True)
    if not items:
        print("❌ 无 dev 样本，无法审计")
        return

    # 🔴 口径铁律：P40 的 to_batch() 是 `lm[i] = torch.from_numpy(s["lm"])`，
    # **不做任何归一化**（训练与 evaluate 都如此）。
    # 我第一版审计错误地套了 FeatureNormalizer，导致输入分布与训练不同，
    # 算出 WER 1.2851 而非 0.5211 —— 差 2.5 倍。
    # 审计必须复现**训练时的真实前向路径**，否则数字没有可比性。
    # （服务端的 ctc_landmark_service 反而做了归一化，那是另一个待查问题。）

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    print("checkpoint epoch=%s recorded_dev_wer=%s"
          % (blob.get("epoch"), blob.get("dev_wer")), flush=True)

    # ---- 推理 ----
    recs = []
    with torch.no_grad():
        for k in range(0, len(items), 32):
            ch = items[k:k + 32]
            lm = torch.stack([torch.from_numpy(s["lm"]) for s in ch]).to(device)
            il = torch.tensor([s["lm"].shape[0] for s in ch], device=device)
            logits = model(lm, il, None, None)
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            ol = np.full((logits.size(0),), logits.size(1), dtype=np.int64)
            dec, _, _ = decode_batch(lp, ol, 1)
            for r, seq in enumerate(dec):
                recs.append({"sid": ch[r]["sid"], "ref_folded": ch[r]["folded"],
                             "ref_raw": ch[r]["toks"], "hyp": voc.decode(list(seq))})
    print("decoded %d\n" % len(recs), flush=True)

    R_f = [t for r in recs for t in r["ref_folded"]]
    R_r = [t for r in recs for t in r["ref_raw"]]
    H = [t for r in recs for t in r["hyp"]]
    n_f, n_r, n_h = len(R_f), len(R_r), len(H)

    print("=== token 总量 ===")
    print("ref(folded)=%d  ref(raw)=%d  hyp=%d" % (n_f, n_r, n_h))
    print("ref 平均长度=%.2f  hyp 平均长度=%.2f" % (n_r / len(recs), n_h / len(recs)))

    u_rr = sum(1 for t in R_r if t == UNK)
    u_rf = sum(1 for t in R_f if t == UNK)
    u_h = sum(1 for t in H if t == UNK)
    print("\n=== <unk> 占比（用户直觉核对）===")
    print("ref(raw)   <unk>: %5.1f%%   <- 真实 OOV 比例" % (100 * u_rr / max(n_r, 1)))
    print("ref(folded)<unk>: %5.1f%%   <- 折叠后（几乎全是 OOV 堆出来的）" % (100 * u_rf / max(n_f, 1)))
    print("hyp        <unk>: %5.1f%%   <- 模型实际吐出多少" % (100 * u_h / max(n_h, 1)))

    E_f = sum(levenshtein(list(r["ref_folded"]), r["hyp"]) for r in recs)
    E_s = sum(levenshtein(list(r["ref_raw"]), r["hyp"]) for r in recs)
    w_f, w_s = E_f / max(n_f, 1), E_s / max(n_r, 1)
    print("\n=== WER 两口径 ===")
    print("宽松(folded, 训练/服务报的) = %.4f" % w_f)
    print("严格(raw, 保留真实 OOV)      = %.4f" % w_s)
    print("=> 折叠口径少算 %.4f" % (w_s - w_f))

    # ---- 纯词表内 WER：参考里属于词表的 token，模型认得准不准 ----
    E_iv = N_iv = 0
    for r in recs:
        ref_iv = [t for t in r["ref_raw"] if t in t2i and t != UNK]
        if not ref_iv:
            continue
        hyp_iv = [t for t in r["hyp"] if t != UNK]
        E_iv += levenshtein(ref_iv, hyp_iv)
        N_iv += len(ref_iv)
    w_iv = E_iv / max(N_iv, 1)
    print("\n=== 剔除 OOV 的纯词表内 WER ===")
    print("参考里属于词表的 token = %d（占 ref(raw) %.1f%%）" % (N_iv, 100 * N_iv / max(n_r, 1)))
    print("纯词表内 WER = %.4f  <-- 对已学词的真实水平" % w_iv)

    # ---- <unk> 通配符命中率 ----
    ref_unk = u_rf
    hit_unk = sum(min(collections.Counter(r["ref_folded"]).get(UNK, 0),
                      collections.Counter(r["hyp"]).get(UNK, 0)) for r in recs)
    print("\n=== <unk> 作为通配符 ===")
    print("ref(folded) 里 <unk> = %d，hyp 能对上 = %d（%.1f%%）"
          % (ref_unk, hit_unk, 100 * hit_unk / max(ref_unk, 1)))

    # ---- 样本级 ----
    eds = [levenshtein(list(r["ref_folded"]), r["hyp"]) for r in recs]
    ex = sum(1 for e in eds if e == 0)
    pa = sum(1 for r, e in zip(recs, eds) if 0 < e < len(r["ref_folded"]))
    print("\n=== 样本级（与 WER 给人的印象相反）===")
    print("完全正确 %d/%d (%.1f%%)" % (ex, len(recs), 100 * ex / len(recs)))
    print("部分正确 %d/%d (%.1f%%)" % (pa, len(recs), 100 * pa / len(recs)))
    print("全错     %d/%d (%.1f%%)" % (len(recs) - ex - pa, len(recs),
                                       100 * (len(recs) - ex - pa) / len(recs)))

    kinds = collections.Counter(H)
    print("\n=== 输出多样性 ===")
    print("hyp 不同 token 数 = %d" % len(kinds))
    print("最高频 10 =", kinds.most_common(10))

    print("\n=== 逐样本对照（前 10 条）===")
    for r in recs[:10]:
        print("  %-12s" % r["sid"])
        print("     ref(raw) : %s" % "/".join(r["ref_raw"]))
        print("     ref(fold): %s" % "/".join(r["ref_folded"]))
        print("     hyp      : %s" % "/".join(r["hyp"]), flush=True)

    receipt = {
        "experiment": "P44", "date": "2026-10-05",
        "question": "用户质疑 dev WER=0.5211，直觉上 80% 输出是 <unk>",
        "checkpoint": CKPT.name, "epoch": blob.get("epoch"),
        "recorded_dev_wer": blob.get("dev_wer"),
        "vocab_size": int(voc.size), "n_dev": len(recs),
        "token_counts": {"ref_folded": n_f, "ref_raw": n_r, "hyp": n_h},
        "unk_ratio": {"ref_raw": round(u_rr / max(n_r, 1), 4),
                      "ref_folded": round(u_rf / max(n_f, 1), 4),
                      "hyp": round(u_h / max(n_h, 1), 4)},
        "wer": {"folded_reported": round(w_f, 4), "strict": round(w_s, 4),
                "in_vocab_only": round(w_iv, 4), "n_iv_ref_tokens": N_iv,
                "folding_gift": round(w_s - w_f, 4)},
        "unk_wildcard": {"ref_unk_tokens": ref_unk, "matched": hit_unk,
                         "hit_rate": round(hit_unk / max(ref_unk, 1), 4)},
        "sample_level": {"exact": ex, "partial": pa,
                         "wrong": len(recs) - ex - pa,
                         "exact_ratio": round(ex / len(recs), 4)},
        "hyp_diversity": {"distinct": len(kinds), "top10": kinds.most_common(10)},
        "mean_len": {"ref": round(n_r / len(recs), 3), "hyp": round(n_h / len(recs), 3)},
        "examples": [{"sid": r["sid"], "ref_raw": r["ref_raw"],
                      "ref_folded": r["ref_folded"], "hyp": r["hyp"]}
                     for r in recs[:10]],
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
    print("\n收据 -> %s" % out)


if __name__ == "__main__":
    main()
