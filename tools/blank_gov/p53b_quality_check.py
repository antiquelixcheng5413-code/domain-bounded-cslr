# -*- coding: utf-8 -*-
"""P53b · 验证 Tasks API 提取的特征质量（小样本，提取后立刻检查）

P53 提取脚本会写出 (48,368) 的 npy，但**维度对不等于内容对**。
本脚本对新提取的 16 条做逐项核对：

1. 形状与 dtype
2. 有限值（无 NaN/Inf）
3. presence 四位的检出率 —— 与旧特征对照
   （P44c 证明 presence 差异是 skew 的直接症状）
4. hands/pose/face 各块的量级 —— 与旧特征对照
5. **最关键**：喂给 P42 checkpoint，看 train 上的表现
   若新特征能被 P42 认出高分 → 说明新旧特征在同一坐标系（skew 已消除）
   若识别率仍低 → 还需要重训（P54）

第 5 条是判据：**路线 2 的目标是让 train 上「离线特征 vs 实时特征」的
差距从 29 倍收敛**。重训之前就该先量一下这个差距。
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
sys.path.insert(0, str(REPO / "app" / "backend"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import decode_batch                    # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv          # noqa: E402

OLD = REPO / "artifacts/part3_features"
NEW = REPO / "artifacts/part3_features_tasksapi"
CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
UNK = "<unk>"
BLOCKS = {"hands[0:126]": (0, 126), "pose[126:158]": (126, 158),
          "face[158:182]": (158, 182), "presence[182:186]": (182, 186),
          "deltas[186:368]": (186, 368)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                      max_tokens=300)

    # 找已提取的样本
    new_files = sorted((NEW / "train").glob("*.landmark.npy"))
    print("已提取 train 特征 = %d 条" % len(new_files))
    if not new_files:
        print("❌ 无新特征，先跑 p53_extract_tasksapi.py")
        return

    sids = [f.name.replace(".landmark.npy", "") for f in new_files]
    items = []
    for sid in sids:
        old_p = OLD / "train" / (sid + ".landmark.npy")
        if not old_p.exists():
            continue
        ids = voc.encode(lab_tr.get(sid, ""))
        if not ids or len(ids) > 24:
            continue
        items.append({"sid": sid,
                      "old": np.load(old_p).astype(np.float32),
                      "new": np.load(new_files[sids.index(sid)]).astype(np.float32),
                      "ref": voc.decode(list(ids))})
    print("可对比样本 = %d\n" % len(items))

    # ---- 1. 基础校验 ----
    print("=" * 70)
    print("1. 基础校验")
    print("=" * 70)
    bad = 0
    for it in items:
        if it["new"].shape != (48, 368):
            print("  ❌ %s shape=%s" % (it["sid"], it["new"].shape))
            bad += 1
        if not np.isfinite(it["new"]).all():
            print("  ❌ %s 含 NaN/Inf" % it["sid"])
            bad += 1
    print("  shape=(48,368) 且全部有限：%s" % ("全部通过" if bad == 0
                                          else "%d 条异常" % bad))

    # ---- 2. presence 检出率 ----
    print("\n" + "=" * 70)
    print("2. presence 检出率对照（skew 的直接症状）")
    print("=" * 70)
    print("%-8s %10s %10s %10s" % ("位", "旧", "新", "差"))
    pres = {}
    for k, nm in enumerate(["pose", "handL", "handR", "face"]):
        o = float(np.mean([it["old"][:, 182 + k].mean() for it in items]))
        n = float(np.mean([it["new"][:, 182 + k].mean() for it in items]))
        pres[nm] = {"old": round(o, 4), "new": round(n, 4)}
        print("%-8s %10.3f %10.3f %10.3f" % (nm, o, n, n - o))

    # ---- 3. 各块量级 ----
    print("\n" + "=" * 70)
    print("3. 各块 std 对照")
    print("=" * 70)
    print("%-18s %10s %10s %8s" % ("block", "旧 std", "新 std", "比"))
    blocks = {}
    for nm, (s0, s1) in BLOCKS.items():
        o = float(np.mean([it["old"][:, s0:s1].std() for it in items]))
        n = float(np.mean([it["new"][:, s0:s1].std() for it in items]))
        blocks[nm] = {"old": round(o, 4), "new": round(n, 4),
                      "ratio": round(n / (o + 1e-9), 3)}
        print("%-18s %10.4f %10.4f %8.2f" % (nm, o, n, n / (o + 1e-9)))

    # ---- 4. P42 checkpoint 在两种特征上的表现（关键判据）----
    print("\n" + "=" * 70)
    print("4. P42 checkpoint（用【旧特征】训练）在两种特征上的表现")
    print("   若「旧特征」高而「新特征」低 → skew 仍在（需重训，这是预期的）")
    print("   这是路线 2 的**前置测量**，不是最终结果")
    print("=" * 70)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                         vocab=int(voc.size), hidden=cfg["hidden"],
                         layers=cfg["layers"], dropout=cfg["dropout"],
                         use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                         mode=cfg.get("mode", "add")).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()

    def run(key):
        hyps = []
        with torch.no_grad():
            for k in range(0, len(items), 16):
                ch = items[k:k + 16]
                x = torch.from_numpy(np.stack([c[key] for c in ch])).to(device)
                il = torch.full((len(ch),), x.shape[1], dtype=torch.long,
                                device=device)
                lg = model(x, il, None, None)
                lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
                ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
                dec, _, _ = decode_batch(lp, ol, 1)
                hyps.extend(voc.decode(list(s)) for s in dec)
        E = sum(levenshtein(list(r["ref"]), h)
                for r, h in zip(items, hyps))
        N = sum(len(r["ref"]) for r in items)
        ex = sum(1 for r, h in zip(items, hyps)
                 if levenshtein(list(r["ref"]), h) == 0)
        H = [t for h in hyps for t in h]
        return {"wer": round(E / max(N, 1), 4),
                "exact": "%d/%d (%.1f%%)" % (ex, len(items),
                                             100 * ex / max(len(items), 1)),
                "unk": round(sum(1 for t in H if t == UNK) / max(len(H), 1), 4)}

    r_old = run("old")
    r_new = run("new")
    print("  旧特征: WER=%.4f  exact=%s  unk=%.1f%%"
          % (r_old["wer"], r_old["exact"], 100 * r_old["unk"]))
    print("  新特征: WER=%.4f  exact=%s  unk=%.1f%%"
          % (r_new["wer"], r_new["exact"], 100 * r_new["unk"]))
    print("\n  => 差距 %+.4f" % (r_new["wer"] - r_old["wer"]))
    print("  ⚠️ 这是**旧模型**在新特征上的表现，必然差。")
    print("     真正的验证是重训后（P54）：用新特征训练，在新特征上评估。")

    # ---- 5. 逐样本 ----
    print("\n" + "=" * 70)
    print("5. 逐样本（前 8）")
    print("=" * 70)
    for it in items[:8]:
        print("  %s" % it["sid"])
        print("     ref: %s" % "/".join(it["ref"]))

    receipt = {
        "experiment": "P53b", "date": "2026-10-05",
        "n_samples": len(items),
        "basic_check_passed": bad == 0,
        "presence": pres, "blocks": blocks,
        "p42_on_old": r_old, "p42_on_new": r_new,
        "interpretation": "旧模型在新特征上必然差（P42 是用旧特征训练的）；"
                          "最终验证要等 P54 重训",
    }
    p = REPO / "artifacts/metrics/blank-gov/p53b-quality-check.json"
    p.write_text(json.dumps(receipt, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
