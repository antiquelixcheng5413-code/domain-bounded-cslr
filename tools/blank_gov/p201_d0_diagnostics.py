#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P201 / D0：数据侧诊断（纯推理，不训练）

1) train / dev 上的官方口径 greedy WER —— 判断泛化间隙（过拟合 vs 欠拟合）
2) 编辑分解：替换 / 删除 / 插入，并统计 top 替换词对 —— 决定是否做定向增广

用法：
  venv/bin/python tools/blank_gov/p201_d0_diagnostics.py \
      --ckpt artifacts/checkpoints/p184-tfnet-freeze-best.pt \
      --split train --img 160
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402

OUTDIR = REPO / "artifacts/metrics/blank-gov"


def log(*a):
    print("[D0 %s] %s" % (time.strftime("%H:%M:%S"),
                         " ".join(str(x) for x in a)), flush=True)


def align(ref, hyp):
    """编辑对齐（Levenshtein 回溯），返回 (subs, dels, ins)。"""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    subs, dels, ins = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + \
                (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                subs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels.append(ref[i - 1])
            i -= 1
        else:
            ins.append(hyp[j - 1])
            j -= 1
    return subs, dels, ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--max", type=int, default=0, help="0=全量；冒烟用小值")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    word2idx, word_set_num, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    split = "dev" if a.split == "validation" else "train"
    labels = P.read_csv_split(split + ".csv")
    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    _, tf = P.build_transforms(hflip=False, img_size=a.img)
    ds = P.RGBSeqDataset(split, labels, word2idx, tf, False)
    if a.max:
        ds.items = ds.items[:a.max]

    model = P.Net.moduleNet(1024, word_set_num + 1, "TFNet",
                            torch.device("cuda"), "CE-CSL",
                            True).to("cuda")
    ck = torch.load(str(a.ckpt), map_location="cpu", weights_only=False,
                    mmap=True)
    sd = ck.get("moduleNet_state_dict") or ck["model_state"]
    P80_adapt = getattr(P, "attach_conformer_adapters", None)
    if P80_adapt is not None and "config" in ck:
        P80_adapt(model, ck["config"], "cuda")
        P.attach_temporal_backbone(model, ck["config"], "cuda")
        ld = model.load_state_dict(sd, strict=False)
        log("strict=False：missing=%d unexpected=%d"
            % (len(ld.missing_keys), len(ld.unexpected_keys)))
    else:
        model.load_state_dict(sd, strict=True)
    model.eval()
    ls = torch.nn.LogSoftmax(dim=-1)

    refs, hyps = [], []
    subs_all, dels_all, ins_all = Counter(), Counter(), Counter()
    P.BLANK_BIAS = 0.0
    t0 = time.time()
    with torch.no_grad():
        for k in range(len(ds)):
            vid, _t, _tl, dl, _tl_len, sids, _ids = P.collate([ds[k]])
            o = model(vid.to("cuda"), dl, False)
            hyp = P.greedy_decode(ls(o[0]).squeeze(1).unsqueeze(1),
                                  dl, idx2word)[0]
            sid = str(sids[0])
            ref = labels[sid][1]
            refs.append(ref)
            hyps.append(hyp)
            r_tok = [r.strip() for r in ref.strip().strip("/").split("/") if r.strip()]
            h_tok = [h.strip() for h in hyp if h.strip()]
            s, dd, ii = align(r_tok, h_tok)
            subs_all.update(s)
            dels_all.update(dd)
            ins_all.update(ii)
            if (k + 1) % 1000 == 0:
                log("%d / %d  (%.0fs)" % (k + 1, len(ds), time.time() - t0))

    o = evaluate(refs, hyps)
    tot_ed = sum(subs_all.values()) + sum(dels_all.values()) + sum(ins_all.values())
    log("WER = %.2f%%  n=%d  编辑总数=%d（S/D/I = %d/%d/%d）"
        % (o["WER_official"], len(refs), tot_ed, sum(subs_all.values()),
           sum(dels_all.values()), sum(ins_all.values())))
    log("top 替换对：")
    for (r, h), c in subs_all.most_common(30):
        log("  %s -> %s   %d" % (r, h, c))

    dst = Path(a.out) if a.out else (
        OUTDIR / ("p201-d0-%s.json" % a.split))
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": str(a.ckpt), "split": a.split, "img": a.img,
        "n": len(refs), "WER_official": o["WER_official"],
        "edits": {"S": sum(subs_all.values()), "D": sum(dels_all.values()),
                  "I": sum(ins_all.values())},
        "top_subs": [[r, h, c] for (r, h), c in subs_all.most_common(80)],
        "top_dels": [[r, c] for r, c in dels_all.most_common(40)],
        "top_ins": [[h, c] for h, c in ins_all.most_common(40)],
    }
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    log("written -> %s（%.0fs）" % (dst, time.time() - t0))


if __name__ == "__main__":
    main()
