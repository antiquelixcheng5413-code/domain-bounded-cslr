#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P145：test split **最终验收**评估（只跑一次，结果直接归档）

🔴🔴 铁律（test 冻结）
  1. **不在 test 上扫描任何超参** —— bias 的值（0.75）是**在 dev 上选定**的，
     这里只做验证，不重选。扫描 ⇒ 等于在 test 上调参 ⇒ 数字作废。
  2. 只评估两个点：bias=0（基线口径）+ bias=0.75（dev 上选定的值）
  3. 结果直接归档，不回头再跑第二次

目的
  回答「dev 上的 blank bias 改善是否迁移到 test」：
    - dev: 55.84 → 54.75（−1.09 pp）
    - test: 待本脚本给出

用法
  python tools/blank_gov/p145_test_eval.py [max_n] [ckpt]
    默认：全量 500 条，ckpt = p123-official55bf16-best.pt
"""
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402

SPLIT = "test"
CK = REPO / (sys.argv[2] if len(sys.argv) > 2
             else "artifacts/checkpoints/p123-official55bf16-best.pt")
MAX_N = int(sys.argv[1]) if len(sys.argv) > 1 else 0     # 0 = 全量

# 🔴 只验证 dev 上选出的值，不在 test 上扫描
BIAS_LIST = [0.0, 0.75]
DEV_REF = {"bias0": 55.84, "bias075": 54.75}


def log(*a):
    print("[P145 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    dev = "cuda"
    log("split = %s   ckpt = %s" % (SPLIT, CK.name))
    log("🔴 只验证 dev 上选定的 bias，不在 test 上扫描")

    labels = P.read_csv_split("%s.csv" % SPLIT)
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d；%s 样本 %d 条" % (wordSetNum, SPLIT, len(labels)))

    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "键名变了：%s" % list(ck)[:8]
    P.OFFICIAL_PAD = bool(ck.get("config", {}).get("official_pad", False))
    log("official_pad = %s（取自 ckpt config）" % P.OFFICIAL_PAD)

    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s）" % (CK.name, ck.get("epoch")))

    _, tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset(SPLIT, labels, word2idx, tf, False)
    if MAX_N:
        ds.items = ds.items[:MAX_N]
    ls = torch.nn.LogSoftmax(dim=-1)

    # ---- 推理一次，缓存 logp ----
    log("推理并缓存 logp（%d 条）…" % len(ds))
    cache, lens, refs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            lp = ls(out[0])[:, 0, :]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            cache.append(lp[:L].to("cpu", torch.float16))
            lens.append(int(dl.view(-1)[0]))
            for b in sids:
                refs.append(labels[b][1])
    log("缓存完成 %.0fs，%d 条" % (time.time() - t0, len(cache)))

    def decode(bias):
        """与 p78 评估段逐字一致的写法"""
        hyps, ntok = [], 0
        P.BLANK_BIAS = float(bias)
        for lp16, L in zip(cache, lens):
            lp = lp16.float().unsqueeze(1)
            if P.BLANK_BIAS:
                lp = lp.clone()
                lp[..., 0] -= P.BLANK_BIAS
            toks = P.greedy_decode(lp, torch.LongTensor([[L]]), idx2word)
            hyps += toks
            ntok += sum(len(t) for t in toks)
        return hyps, ntok

    print()
    log("──── test 最终验收 ────")
    log("bias    WER        生成率    Δ vs bias=0")
    rows = {}
    base = None
    for b in BIAS_LIST:
        hyps, ntok = decode(b)
        o = evaluate(refs, hyps)
        gen = 100.0 * ntok / max(o["ref_tokens"], 1)
        rows["bias%.2f" % b] = {
            "WER": o["WER_official"], "gen": gen,
            "ref_tokens": o["ref_tokens"], "edits": o.get("edits"),
        }
        if base is None:
            base = o["WER_official"]
        log("%.2f    %6.2f%%     %5.1f%%    %+6.2f pp"
            % (b, o["WER_official"], gen, o["WER_official"] - base))

    print()
    log("──── 与 dev 对照（bias 是否在 test 上复现）────")
    log("  口径        dev      test     差")
    log("  bias=0      %5.2f    %6.2f   %+6.2f"
        % (DEV_REF["bias0"], rows["bias0.00"]["WER"],
           rows["bias0.00"]["WER"] - DEV_REF["bias0"]))
    log("  bias=0.75   %5.2f    %6.2f   %+6.2f"
        % (DEV_REF["bias075"], rows["bias0.75"]["WER"],
           rows["bias0.75"]["WER"] - DEV_REF["bias075"]))
    d_dev = DEV_REF["bias075"] - DEV_REF["bias0"]
    d_test = rows["bias0.75"]["WER"] - rows["bias0.00"]["WER"]
    log("  bias 收益   %+5.2f    %+6.2f   ⇒ %s"
        % (d_dev, d_test,
           "✅ 迁移成功（test 上也改善）" if d_test < 0 else "❌ 未迁移（test 上无改善/变差）"))

    print()
    log("──── 距官方基准（test）────")
    for nm, v in [("官方 VAC", 43.3), ("官方 TFNet", 41.9)]:
        log("  %-10s %.1f%%   我们(bias0.75) %.2f%%   差 %+.2f pp"
            % (nm, v, rows["bias0.75"]["WER"], rows["bias0.75"]["WER"] - v))

    out = REPO / "artifacts/metrics/blank-gov/p145-test-final.json"
    out.write_text(json.dumps({
        "split": SPLIT, "ckpt": CK.name, "n_samples": len(cache),
        "ref_tokens": rows["bias0.00"]["ref_tokens"],
        "results": rows,
        "dev_reference": DEV_REF,
        "bias_gain_pp": {"dev": d_dev, "test": d_test},
        "note": "test 最终验收；bias 在 dev 上选定，此处只验证不重选",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
