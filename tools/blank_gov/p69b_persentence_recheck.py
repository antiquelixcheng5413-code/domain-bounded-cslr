"""P69b：逐句等权 WER 到底是 2.8774 还是 0.5128？

P69 用官方式(11)算 token 级= 0.5211 ✓，但逐句等权得**0.5128**。
而 P50 报的是 **2.8774**，差5.6 倍。

哪个对？必须查清 —— 这决定 P50「四件套」铁律是否成立。

可能原因：
  A. P50 分母用错（可能没排除 ref 为空的句子，或用错字段）
  B. P50 用了不同的样本集
  C. 2.8774 其实是别的量
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from p40_rgb_main import levenshtein, DualInputCTC, read_csv       # noqa: E402
from cslr.recognition.training import decode_batch                  # noqa: E402

lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
FEAT = REPO / "artifacts/part3_features/validation"
voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                  max_tokens=300)

items = []
for sid in sorted(lab_dv):
    p = FEAT / (sid + ".landmark.npy")
    if not p.exists():
        continue
    ids = voc.encode(lab_dv[sid])
    if not ids or len(ids) > 24:
        continue
    items.append({"sid": sid, "lm": np.load(p).astype(np.float32),
                  "ref": voc.decode(list(ids))})

blob = torch.load(REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt",
                  map_location="cpu", weights_only=False)
c = blob["config"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = DualInputCTC(lm_dim=c["lm_dim"], rgb_dim=c["rgb_dim"], vocab=int(voc.size),
                 hidden=c["hidden"], layers=c["layers"], dropout=c["dropout"],
                 use_rgb=False, use_lm=True, mode="add").to(dev)
m.load_state_dict(blob["model_state"])
m.eval()

recs = []
with torch.no_grad():
    for k in range(0, len(items), 32):
        ch = items[k:k + 32]
        lm = torch.from_numpy(np.stack([s["lm"] for s in ch])).to(dev)
        il = torch.full((len(ch),), lm.shape[1], dtype=torch.long, device=dev)
        lg = m(lm, il, None, None)
        lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
        ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
        dec, _, _ = decode_batch(lp, ol, 1)
        for r, seq in enumerate(dec):
            recs.append({"ref": ch[r]["ref"],
                         "hyp": voc.decode(list(seq))})

print("=" * 74)
print("逐句等权 WER 的多种算法（哪个是 2.8774？）")
print("=" * 74)

# 算法 1：per-sentence WER = edit/len(ref)，再对句子取均值
a1 = np.array([levenshtein(r["ref"], r["hyp"]) / max(len(r["ref"]), 1)
               for r in recs])
print("  A1  mean(edit_i / len(ref_i))        = %.4f  <- 常规理解" % a1.mean())

# 算法 2：mean(edit_i) / mean(len(ref_i))  —— 比值的均值
a2 = np.mean([levenshtein(r["ref"], r["hyp"]) for r in recs]) / \
     np.mean([len(r["ref"]) for r in recs])
print("  A2  mean(edit_i) / mean(len(ref_i))   = %.4f" % a2)

# 算法 3：sum(edit_i / len(ref_i)) 不取均值（累加）
print("  A3  sum(edit_i/len(ref_i))           = %.4f" % a1.sum())

# 算法 4：mean(edit_i) / len(ref_i)逐句算术平均（分母逐句，可能就是我P50的 bug）
a4 = np.mean([levenshtein(r["ref"], r["hyp"]) / len(r["ref"])
              if r["ref"] else 0.0 for r in recs])
print("  A4  同A1（等价）                      = %.4f" % a4)

print("\n  官方式(11)= sum(edit_i)/sum(len(ref_i)) = %.4f"
      % (sum(levenshtein(r["ref"], r["hyp"]) for r in recs)
         / sum(len(r["ref"]) for r in recs)))

print("\n" + "=" * 74)
print("检查 P50 收据里到底记了什么")
print("=" * 74)
p = REPO / "artifacts/metrics/blank-gov/p50-wer-audit.json"
if p.exists():
    d = json.load(open(p, encoding="utf-8"))
    def find(o, key, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if "per_sentence" in str(k) or "2.87" in str(v):
                    print("    %s%s = %s" % (path, k, v))
                find(v, key, path + k + ".")
    find(d, "per_sentence")
else:
    print("    p50 收据不存在")

print("\n" + "=" * 74)
print("逐句 WER 的分布（看是否有极端值拉高）")
print("=" * 74)
print("  per-sentence WER: min %.3f  median %.3f  mean %.4f  max %.3f"
      % (a1.min(), np.median(a1), a1.mean(), a1.max()))
print("  >1.0 的句子: %d / %d (%.1f%%)"
      % ((a1 > 1.0).sum(), len(a1), 100 * (a1 > 1.0).mean()))
print("  >2.0 的句子: %d / %d" % ((a1 > 2.0).sum(), len(a1)))
print("\n  最高的 8 句：")
top = np.argsort(-a1)[:8]
for i in top:
    print("    %.3f  ref(%d)=%s" % (a1[i], len(recs[i]["ref"]),
                                     "/".join(recs[i]["ref"])))
    print("           hyp(%d)=%s" % (len(recs[i]["hyp"]),
                                     "/".join(recs[i]["hyp"]) or "(空)"))