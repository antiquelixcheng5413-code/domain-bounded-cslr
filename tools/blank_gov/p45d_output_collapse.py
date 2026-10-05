import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary
from cslr.recognition.training import decode_batch
from p40_rgb_main import DualInputCTC, read_csv

CKPT = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
LM = REPO / "artifacts/part3_features"
UNK = "<unk>"

device = "cuda" if torch.cuda.is_available() else "cpu"
lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                  max_tokens=300)
items = []
for sid in sorted(lab_dv):
    p = LM / "validation" / (sid + ".landmark.npy")
    if not p.exists():
        continue
    ids = voc.encode(lab_dv[sid])
    if not ids or len(ids) > 24:
        continue
    items.append({"lm": np.load(p).astype(np.float32),
                  "folded": voc.decode(list(ids))})

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
        il = torch.full((len(ch),), x.shape[1], dtype=torch.long, device=device)
        lg = model(x, il, None, None)
        lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
        ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
        dec, _, _ = decode_batch(lp, ol, 1)
        for j, s in enumerate(dec):
            ch[j]["hyp"] = voc.decode(list(s))

H = [t for r in items for t in r["hyp"]]
R = [t for r in items for t in r["folded"]]
cnt = collections.Counter(H)
tot = len(H)
print("输出 token 总数 = %d，参考 = %d" % (tot, len(R)))
print("\n=== 输出集中度（hyp 前 12）===")
cum = 0
for i, (w, c) in enumerate(cnt.most_common(12), 1):
    cum += c
    print("  %2d. %-8s %4d  %5.1f%%  累计 %.1f%%" % (i, w, c, 100 * c / tot,
                                               100 * cum / tot))
print("\n不同输出词数 = %d" % len(cnt))

# 真实词（非 unk）的集中度
real = collections.Counter(t for t in H if t != UNK)
rt = sum(real.values())
print("\n=== 剔除 unk 后的输出（真实词）===")
print("真实词 token 数 = %d，种类 = %d" % (rt, len(real)))
cum = 0
for i, (w, c) in enumerate(real.most_common(10), 1):
    cum += c
    print("  %2d. %-8s %4d  %5.1f%%  累计 %.1f%%" % (i, w, c, 100 * c / rt,
                                               100 * cum / rt))

# 命中率：参考里每个词，模型有没有输出过
ref_set = set(R)
hit = sum(1 for t in R if t != UNK and t in real)
n_iv = sum(1 for t in R if t != UNK)
print("\n=== 参考中「已学词」被输出的比例 ===")
print("  已学词 token = %d，其中模型至少输出过一次 = %d（%.1f%%）"
      % (n_iv, hit, 100 * hit / max(n_iv, 1)))
print("  另：已学词里模型**从未**输出过的种类 = %d"
      % len({t for t in R if t != UNK} - set(real)))

# 按训练频次看：哪些词模型从来不说
trc = collections.Counter()
for g in lab_tr.values():
    trc.update([t for t in g.split("/") if t])
never = {t for t in R if t != UNK} - set(real)
print("\n=== 模型从不输出的已学词：训练频次分布 ===")
buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 30), (31, 100), (101, 10 ** 9)]
for lo, hi in buckets:
    ws = [t for t in never if lo <= trc.get(t, 0) <= hi]
    tot_all = len([t for t in R if t != UNK and lo <= trc.get(t, 0) <= hi])
    if ws:
        print("  频次 %-8s 从不输出 %3d / 该桶已学词 %3d = %.0f%%"
              % ("%d-%d" % (lo, hi) if hi < 10 ** 9 else "%d+" % lo,
                 len(ws), tot_all, 100 * len(ws) / max(tot_all, 1)))
print("\n已学词总数(种类) = %d" % len({t for t in R if t != UNK}))
