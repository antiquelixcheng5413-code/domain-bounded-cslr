import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

R = Path("/home/su127/FYP/domain-bounded-cslr")
for p in (R / "tools/blank_gov", R / "external/TFNet", R / "src"):
    sys.path.insert(0, str(p))
import p78_train_official_tfnet as P78  # noqa

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
                map_location="cuda", weights_only=False)
mdl = P78.Net.moduleNet(1024, n + 1, "TFNet", "cuda", "CE-CSL", True)
mdl.load_state_dict(ck["model_state"], strict=True)
mdl = mdl.cuda().eval()

print("classifier11 is classifier22:", mdl.classifier11 is mdl.classifier22)
print("classifier33 is classifier44:", mdl.classifier33 is mdl.classifier44)

cap = {}
h = mdl.classifier11.register_forward_hook(
    lambda m, i, o: cap.setdefault("in", i[0].detach()))
for vid, tgt, tl, vl, tll, sids, ids in dl:
    with torch.no_grad():
        out = mdl(vid.cuda(), vl, False)
    print("vid", tuple(vid.shape), "vl", vl.view(-1).tolist())
    print("out len", len(out), "out0", tuple(out[0].shape),
          "out5", out[5])
    x = cap["in"]
    print("classifier11 input", tuple(x.shape), x.dtype)
    break
h.remove()
