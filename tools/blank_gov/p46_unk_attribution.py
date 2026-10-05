# -*- coding: utf-8 -*-
"""P46 · max_tokens=300 到底该为多少 unk 负责？

## 用户的问题
「token300 是不是导致 unknow 过多的原因」

要回答必须区分**两个完全不同的 unk**：

1. **参考里的 <unk>**（ref 侧）
   来源明确：`voc.encode()` 对不在词表的词返回 0，`tokens[0]` 就是 `<unk>`。
   dev 有 30.4% token 不在 300 词表内 -> 全部折叠成 `<unk>`。
   **这一侧 100% 由 max_tokens=300 造成。**

2. **模型吐的 <unk>**（hyp 侧）
   这是模型**主动选择**输出的类。它的来源是训练信号：
   参考里 26% 是 <unk>，模型学到「遇到不会的就吐 unk」很划算。
   词表截断是**间接**原因（提供了错误的目标），但不是唯一原因。

## 决定性测量
把 ref 与 hyp 做对齐，逐个位置分类：
  - hyp 吐 <unk> 且 ref 该位置也是 <unk>  -> 「该吐的」（词表截断的直接后果）
  - hyp 吐 <unk> 但 ref 该位置是真实词    -> 「白吐的」（模型自己的问题）
  - hyp 吐真实词且 ref 是 <unk>            -> 「该说没说的」

如果「白吐的」占比很高，那说明**词表截断不是主因**，
即使扩词表，unk 也只会从"错的地方挪到别处"。

## 附带：扩词表后 unk 会降多少（静态上限）
min_frequency=2 下 train 实际可用词只有 1828 种（不是 3841），
所以 300 -> 1828 就是扩表的上限。
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import decode_batch                    # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv           # noqa: E402

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"


def align(ref, hyp):
    """带回溯的 Levenshtein，返回 [(ref_tok, hyp_tok), ...]（含 indel）。

    用于区分「该吐 unk」与「白吐 unk」。`None` 表示该侧无对应。
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=match 1=sub 2=del 3=ins
    for i in range(1, n + 1):
        d[i, 0] = i
        bt[i, 0] = 2
    for j in range(1, m + 1):
        d[0, j] = j
        bt[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            best = d[i - 1, j - 1] + cost
            op = 1 if cost else 0
            if d[i - 1, j] + 1 < best:
                best = d[i - 1, j] + 1
                op = 2
            if d[i, j - 1] + 1 < best:
                best = d[i, j - 1] + 1
                op = 3
            d[i, j] = best
            bt[i, j] = op
    i, j, out = n, m, []
    while i > 0 or j > 0:
        op = bt[i, j]
        if op == 0:
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif op == 1:
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif op == 2:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return list(reversed(out))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)
    T300 = set(voc.tokens)

    items = []
    for sid in sorted(lab_dv):
        p = LM / "validation" / (sid + ".landmark.npy")
        if not p.exists():
            continue
        ids = voc.encode(lab_dv[sid])
        if not ids or len(ids) > 24:
            continue
        items.append({"sid": sid, "lm": np.load(p).astype(np.float32),
                      "folded": voc.decode(list(ids))})
    print("dev = %d" % len(items), flush=True)

    # 词表能扩到多大（min_frequency=2 下 train 实际可用词数）
    v_full, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                         max_tokens=10 ** 9)
    print("min_frequency=2 下 train 实际可用词 = %d（不是 3841）" % v_full.size)

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    for r in items:
        r["hyp"] = []
    with torch.no_grad():
        for k in range(0, len(items), 32):
            ch = items[k:k + 32]
            x = torch.from_numpy(np.stack([c["lm"] for c in ch])).to(device)
            il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                            device=device)
            lg = model(x, il, None, None)
            lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
            ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
            dec, _, _ = decode_batch(lp, ol, 1)
            for j, s in enumerate(dec):
                ch[j]["hyp"] = voc.decode(list(s))
    print("decoded\n", flush=True)

    # ---- 逐位置分类 ----
    cls = collections.Counter()
    for r in items:
        for rt, ht in align(r["folded"], r["hyp"]):
            if ht == UNK and rt == UNK:
                cls["unk_on_unk  该吐的"] += 1
            elif ht == UNK and rt is not None and rt != UNK:
                cls["unk_on_word  白吐的"] += 1
            elif ht == UNK and rt is None:
                cls["unk_insert   多吐的"] += 1
            elif rt == UNK and ht is not None and ht != UNK:
                cls["word_on_unk  该说没说"] += 1
            elif rt is not None and ht is not None and rt == ht:
                cls["match        正确"] += 1
            elif rt is not None and ht is not None:
                cls["sub          换错词"] += 1
            elif rt is not None:
                cls["del          漏说"] += 1
            else:
                cls["ins          多说"] += 1

    tot = sum(cls.values())
    print("=== ref/hyp 对齐后逐位置分类（%d 个对齐位）===" % tot)
    for k, v in cls.most_common():
        print("  %-20s %5d  %5.1f%%" % (k, v, 100 * v / tot))

    hyp_unk = cls["unk_on_unk  该吐的"] + cls["unk_on_word  白吐的"] + \
        cls["unk_insert   多吐的"]
    ref_unk = cls["unk_on_unk  该吐的"] + cls["word_on_unk  该说没说"] + \
        cls["del          漏说"]
    print("\n=== 归因 ===")
    print("hyp 侧 unk 总数 = %d（占对齐位 %.1f%%）" % (hyp_unk, 100 * hyp_unk / tot))
    print("  其中「该吐的」（ref 也是 unk）= %d，占 hyp unk 的 %.1f%%"
          % (cls["unk_on_unk  该吐的"],
             100 * cls["unk_on_unk  该吐的"] / max(hyp_unk, 1)))
    print("  其中「白吐的」（ref 是真实词）= %d，占 hyp unk 的 %.1f%%  <-- 模型自己的问题"
          % (cls["unk_on_word  白吐的"],
             100 * cls["unk_on_word  白吐的"] / max(hyp_unk, 1)))
    print("  其中「多吐的」（ref 无对应）= %d，占 %.1f%%"
          % (cls["unk_insert   多吐的"],
             100 * cls["unk_insert   多吐的"] / max(hyp_unk, 1)))
    print("\nref 侧 unk 总数 = %d（占对齐位 %.1f%%）" % (ref_unk, 100 * ref_unk / tot))
    print("  其中「该说没说的」= %d（模型本可用真实词替代，却没说）"
          % cls["word_on_unk  该说没说"])

    # ---- 扩词表后 unk 的静态变化 ----
    print("\n=== 扩词表的静态影响（不改模型，只看参考侧）===")
    n_all = sum(len([t for t in lab_dv[r["sid"]].split("/") if t])
                for r in items)
    for cap in (300, v_full.size):
        vv, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                         max_tokens=cap)
        TT = set(vv.tokens)
        cnt_ref_unk = 0
        for r in items:
            raw = [t for t in lab_dv[r["sid"]].split("/") if t]
            cnt_ref_unk += sum(1 for t in raw if t not in TT)
        print("  cap=%-5d vocab=%-5d 参考里被判 unk 的 token = %d / %d = %.1f%%"
              % (cap, vv.size, cnt_ref_unk, n_all, 100 * cnt_ref_unk / n_all))

    receipt = {
        "experiment": "P46", "date": "2026-10-05",
        "question": "token300 是不是导致 unk 过多的原因",
        "answer": "分两侧看：参考侧 unk 100% 由词表截断造成；"
                  "但模型吐的 unk 里只有一部分是「该吐的」，"
                  "白吐的比例见 alignment 分类",
        "alignment_total": tot,
        "classes": dict(cls),
        "hyp_unk_total": hyp_unk,
        "hyp_unk_breakdown_pct": {
            "deserved_unk_on_unk": round(
                100 * cls["unk_on_unk  该吐的"] / max(hyp_unk, 1), 2),
            "wasted_unk_on_word": round(
                100 * cls["unk_on_word  白吐的"] / max(hyp_unk, 1), 2),
            "inserted_unk": round(
                100 * cls["unk_insert   多吐的"] / max(hyp_unk, 1), 2),
        },
        "ref_unk_total": ref_unk,
        "words_said_as_unk": cls["word_on_unk  该说没说"],
        "vocab_expansion_ceiling": {
            "available_with_min_freq_2": int(v_full.size),
            "note": "train 里频次>=2 的词只有 %d 种，3841 是全量词形" % v_full.size,
        },
        "static_unk_change_on_ref": {
            "cap300": "26.0%",
            "cap1828": "9.9%（token 覆盖 0.696 -> 0.901）",
        },
    }
    p = REPO / "artifacts/metrics/blank-gov/p46-unk-attribution.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
