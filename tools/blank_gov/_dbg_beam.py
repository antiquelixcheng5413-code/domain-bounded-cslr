import sys
from pathlib import Path
REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
import torch
import p78_train_official_tfnet as P
import p189_beam_eval as B

_, _, idx2word = P.DPM.Word2Id(str(P.CSV_DIR/"train.csv"), str(P.CSV_DIR/"dev.csv"),
                               str(P.CSV_DIR/"test.csv"), "CE-CSL")
labels = P.read_csv_split("dev.csv")
blob = torch.load(str(REPO/"artifacts/metrics/blank-gov/p188_logp_official.pt"),
                  map_location="cpu", weights_only=False)["items"]
for sid in sorted(blob)[:5]:
    rec = blob[sid]
    t = P._conv_len(rec["dl"])
    g = B.greedy(rec["logp"], t, idx2word)
    b = B.beam_decode(rec["logp"], t, idx2word)
    print(sid, "T=", rec["logp"].shape, "convlen=", t,
          "| ref=", len(labels[sid][1].split()), "greedy=", len(g), "beam=", len(b))
    print("  ref  :", labels[sid][1][:90])
    print("  greedy:", " ".join(g)[:90])
    print("  beam  :", " ".join(b[:30]))
