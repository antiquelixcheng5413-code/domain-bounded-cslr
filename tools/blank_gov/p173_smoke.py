#!/usr/bin/env python
import sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO/"tools"/"blank_gov"))
sys.path.insert(0, str(REPO/"external"/"TFNet"))
sys.path.insert(0, str(REPO/"src"))

import p173_flow_probe as P173
import p78_train_official_tfnet as P

dev = "cuda"
word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
    str(P.CSV_DIR/"train.csv"), str(P.CSV_DIR/"dev.csv"), str(P.CSV_DIR/"test.csv"), "CE-CSL")
labels_dv = P.read_csv_split("dev.csv")

# 1) conv1 补丁是否生效
model = P.Net.moduleNet(1024, wordSetNum+1, "VAC", torch.device(dev), "CE-CSL", True).to(dev)
print("[smoke] conv1.in_channels =", model.conv2d.conv1.in_channels, "(期望 5)")
print("[smoke] conv1.weight.shape =", tuple(model.conv2d.conv1.weight.shape))

# 2) 数据集 5 通道样本
P.OFFICIAL_PAD = True; P.RGB_FIX = True
tr_tf, dv_tf = P.build_transforms(hflip=False, img_size=160)
ds = P173.FlowRGBSeqDataset("train", P.read_csv_split("train.csv"), word2idx, tr_tf, True)
s = ds[0]
vid = s["video"]
print("[smoke] sample video shape =", tuple(vid.shape), "dtype =", vid.dtype)
print("[smoke] sample rgb range = [%.3f, %.3f]" % (vid[:, :3].min(), vid[:, :3].max()))
print("[smoke] sample diff range = [%.3f, %.3f]" % (vid[:, 3:].min(), vid[:, 3:].max()))

# 3) 一次前向（用真实样本构造一个 batch）
from torch.utils.data import DataLoader
dl0 = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=P173.flow_collate, num_workers=0)
batch = next(iter(dl0))
vid_b, tgt, tgt_len, dl, tl, sids, ids = batch
print("[smoke] batch video shape =", tuple(vid_b.shape), "(期望 [B,5,T,H,W])")
vid_b = vid_b.to(dev)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model(vid_b, dl, True)
print("[smoke] forward out[0] shape =", tuple(out[0].shape), "out[5] =", tuple(out[5].shape))
# 一个训练步
ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
ls = torch.nn.LogSoftmax(dim=-1)
loss = ctc(ls(out[0].float()), tgt, out[5], tgt_len.cpu()).mean()
loss.backward()
print("[smoke] one train step loss = %.3f  OK" % float(loss))
print("SMOKE_DONE")
