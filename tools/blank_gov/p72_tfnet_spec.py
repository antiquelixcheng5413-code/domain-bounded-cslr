"""P72：从官方论文提取 TFNet 的完整规格（供复刻用）

已知（从论文正文抽到）：
  式(1) f_frame  = F_f(V) ∈ R^{T×C'}            帧级特征
  式(2) f_temporal = F_s^t(f_frame) ∈ R^{T'×C''}  时域分支：1D CNN + BiLSTM
  式(3) f_freq     = F_s^f(DFT(f_frame)) ∈ R^{T'×C''} 频域分支：DFT + 1D CNN + BiLSTM
  式(4) f_classification = F_linear(f_temporal + f_freq) ∈ R^{T'×l}  相加后全连接
  损失 = L_VAE_t + L_VAE_f + L_CTC

缺：1D CNN 具体层数/通道、DFT 长度、训练超参、CTC 解码方式。
本脚本把这些段落抽全。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pypdf

PDF = Path("/mnt/c/Users/su127/Desktop/中文手语识别/_ce-csl-paper.pdf")
r = pypdf.PdfReader(str(PDF))
txt = "\n".join((p.extract_text() or "") for p in r.pages)
flat = re.sub(r"[ \t]+", " ", txt)

QUERIES = [
    ("loss", r"Loss Function"),
    ("vae", r"(?i)variational autoencoder|L_VAE|VAE"),
    ("ctc_head", r"(?i)CTC (?:loss|head|objective)"),
    ("impl_rules", r"Implementation rules"),
    ("hyper", r"(?i)learning rate|batch size|epoch|Adam|optimizer"),
    ("augment", r"(?i)random crop|flip|augmentation|± ?20|20%"),
    ("beam", r"(?i)beam search|beam width|greedy"),
    ("cnn_channels", r"(?i)1D CNN|convolution|kernel|channel"),
    ("frame_feat", r"(?i)MAM-FSD|ResNet|backbone|frame-level feature extrac"),
    ("official_code", r"(?i)github"),
]

out = {}
for key, pat in QUERIES:
    print("=" * 74)
    print("【%s】 %r" % (key, pat))
    print("=" * 74)
    seen = set()
    hits = []
    for m in re.finditer(pat, flat):
        s = max(0, m.start() - 320)
        e = min(len(flat), m.end() + 620)
        seg = flat[s:e].replace("\n", " ")
        k = seg[:60]
        if k in seen:
            continue
        seen.add(k)
        hits.append(seg)
        print("\n--- @%d ---" % m.start())
        print("  " + seg)
        if len(hits) >= 3:
            break
    out[key] = hits
    print()

p = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov"
         "/p72-tfnet-spec-raw.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n原始抽取 -> %s" % p)