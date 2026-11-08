#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU 验证：真实 model(vid, dl, True) 前向下 FeatureCapture 是否捕获主分支特征。

与 GPU 训练并行，纯 CPU，不占显存。目的是排除「hook 静默失效 ⇒ CSGC 空操作」。
"""
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

R = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (R / "tools/blank_gov", R / "external/TFNet", R / "src"):
    sys.path.insert(0, str(p))
import p78_train_official_tfnet as P78  # noqa
from csgc_module import FeatureCapture, GlossPrototypeBank  # noqa

P78.OFFICIAL_PAD = True
P78.RGB_FIX = True
P78.BLANK_BIAS = 0.0

lab = P78.read_csv_split("dev.csv")
w2i, n, i2w = P78.DPM.Word2Id(
    str(P78.CSV_DIR / "train.csv"), str(P78.CSV_DIR / "dev.csv"),
    str(P78.CSV_DIR / "test.csv"), "CE-CSL")
_, tf = P78.build_transforms(hflip=False, img_size=160, y_shift=0,
                             temporal_rescale=0.0)
ds = P78.RGBSeqDataset("dev", lab, w2i, tf, False)
ds.items = ds.items[:2]
dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=P78.collate,
                num_workers=0)

ck = torch.load(R / "artifacts/checkpoints/p184-tfnet-freeze-best.pt",
                map_location="cpu", weights_only=False)
m = P78.Net.moduleNet(1024, n + 1, "TFNet", "cpu", "CE-CSL", True)
m.load_state_dict(ck["model_state"], strict=True)
m.eval()

cap = FeatureCapture().attach(m)
bank = GlossPrototypeBank(m.classifier11.weight, momentum=0.99,
                          tau=0.07, n_neg=16)
print("proto", tuple(bank.proto.shape), flush=True)

cap.reset()
for vid, tgt, tl, vl, tll, sids, ids in dl:
    with torch.no_grad():
        out = m(vid, vl, True)          # 与训练一致的 isTrain=True
    lp = torch.log_softmax(out[0].float(), dim=-1)
    print("hook captured:", cap.feat is not None,
          "shape", tuple(cap.feat.shape) if cap.feat is not None else None,
          flush=True)
    print("out0", tuple(out[0].shape), "lgt", out[5].view(-1).tolist(),
          flush=True)
    cl = bank(lp, cap.feat.float(), tgt, tl, out[5])
    print("csgc loss =", float(cl), flush=True)
    assert cap.feat is not None, "hook FAILED: CSGC would be a silent no-op"
    assert cap.feat.shape[:2] == lp.shape[:2], "feature/logit time-batch mismatch"
    assert float(cl) > 0, "csgc loss is zero"
    break
cap.detach()
print("CSGC hook self-check PASS")
