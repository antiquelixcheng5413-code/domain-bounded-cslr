"""P120: 从已有 checkpoint 补 beam search 口径（0A）

## 为什么需要
官方 42.1% 用 `ctcdecode` **beam_width=10**；我们全部历史数字是 **greedy**。
⇒ 所有「距官方 X pp」的说法都不可比。本脚本**不重训**，
   直接加载 checkpoint 同时算 greedy 与 beam，给出口径差。

## 依赖
`_scratch/beam_decode.py`（自实现标准 CTC beam search，5 项单测已过）
⚠️ 官方 ctcdecode 在本机 Python 3.14.4 装不上（无 wheel / 编译失败），
   但算法为标准 CTC 前向且官方未传 lexicon ⇒ 口径可对齐。【依据】【有】

## 🔴 关键约定（照抄训练脚本，勿自创）
- 词表：`DPM.Word2Id(train.csv, dev.csv, test.csv, "CE-CSL")`
- dev 数据集：`RGBSeqDataset("dev", ...)`（P77 里落在 dev/ 目录）
- 模型：`Net.moduleNet(hidden, wordSetNum+1, module, ...)`
- 🔴 **argmax 得到的 class k 直接对应 idx2word[k]，不减 1**
  （训练脚本第 335 行明确注释；官方 idx2word[0]=PAD 与我们相反）

## 用法
    ./venv/bin/python tools/blank_gov/p120_beam_eval.py --ckpt <path> --tag <name>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools/blank_gov"))
sys.path.insert(0, "/mnt/c/Users/su127/Desktop/中文手语识别/_scratch")

import p78_train_official_tfnet as T          # noqa: E402
from beam_decode import beam_search_decode, greedy_decode_tc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--beam-width", type=int, default=10)
    ap.add_argument("--max-dev", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    hidden = ck["hidden"]
    wordSetNum = ck["wordSetNum"]
    idx2word = ck["idx2word"]
    no_hflip = cfg.get("no_hflip", True)
    img_size = cfg.get("img_size", 160)
    module_name = cfg.get("module", "VAC")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 照抄训练脚本 main() 的构造顺序 ----
    labels_dv = T.read_csv_split("dev.csv")
    # 🔴 用 p78 已 import 的 DPM（官方模块名 DataProcessMoudle），不自己 import
    DPM = T.DPM
    word2idx, _, idx2word_chk = DPM.Word2Id(
        str(T.CSV_DIR / "train.csv"), str(T.CSV_DIR / "dev.csv"),
        str(T.CSV_DIR / "test.csv"), "CE-CSL")
    assert idx2word_chk == idx2word, "词表与 checkpoint 不一致"
    # ⚠️ word2idx 有 wordSetNum+1 个键（多一个 0 号 <unk>），不是 wordSetNum
    assert len(word2idx) == wordSetNum + 1, \
        "word2idx 规模异常：%d vs wordSetNum+1=%d" % (len(word2idx),
                                                       wordSetNum + 1)

    _, dv_tf = T.build_transforms(hflip=not no_hflip, img_size=img_size)
    dv_ds = T.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if args.max_dev:
        dv_ds.items = dv_ds.items[: args.max_dev]
    # ⚠️ p78 的 _AdaptiveLoader 是 main() 内的局部类（按帧数桶自适应 batch），
    #    不在模块级。它只负责组装 batch、不涉模型 ⇒ 评估侧用标准 DataLoader 等价。
    from torch.utils.data import DataLoader
    dv_dl = DataLoader(dv_ds, batch_size=4, num_workers=2,
                       collate_fn=T.collate, shuffle=False)

    # 🔴 签名照抄 p78 第 546 行：moduleNet(hidden, C, module, device, "CE-CSL", True)
    model = T.Net.moduleNet(hidden, wordSetNum + 1, module_name,
                            torch.device(dev), "CE-CSL", True).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval().to(dev)

    # 🔴 照抄 p78 第 573 行：ls = torch.nn.LogSoftmax(dim=-1)
    ls = torch.nn.LogSoftmax(dim=-1)
    g_hyp, b_hyp, refs = [], [], []
    t0 = time.time()
    nfail = 0
    with torch.no_grad():
        for vid, tgt, tgt_len, dl, tl_len, sids, ids in dv_dl:
            vid = vid.to(dev)
            o = model(vid, dl, False)
            logp = ls(o[0])                    # 官方布局 [T,B,C]
            Tt, B, C = logp.shape
            for b in range(B):
                L = max(1, int(tl_len[b]))
                seq = logp[:L, b, :]
                g_hyp.append([idx2word[k] for k in greedy_decode_tc(seq)
                              if 0 <= k < len(idx2word)])
                try:
                    ids_b = beam_search_decode(seq,
                                               beam_width=args.beam_width)
                except Exception as e:
                    nfail += 1
                    ids_b = greedy_decode_tc(seq)
                b_hyp.append([idx2word[k] for k in ids_b
                              if 0 <= k < len(idx2word)])
            for s in sids:
                refs.append(labels_dv[s][1])

    og = T.evaluate(refs, g_hyp)
    ob = T.evaluate(refs, b_hyp)

    res = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "ckpt_epoch": ck.get("epoch"),
        "module": module_name,
        "hflip": not no_hflip,
        "beam_width": args.beam_width,
        "n_dev": len(refs),
        "greedy_wer": round(og["WER_official"], 4),
        "beam_wer": round(ob["WER_official"], 4),
        "delta_pp": round(og["WER_official"] - ob["WER_official"], 4),
        "beam_fail": nfail,
        "minutes": round((time.time() - t0) / 60, 2),
        # 🔴 evaluate() 的真实键：edits / ref_tokens（官方口径）
        #    + WER_excl_unk / WER_per_sentence / exact_rate_pct（附加诊断）
        #    ⚠️ 不是 ins/del/sub/n —— 之前写错导致统计恒为空
        "greedy_edits": og["edits"],
        "beam_edits": ob["edits"],
        "ref_tokens": og["ref_tokens"],
        "n_sentences": og["n_sentences"],
        "greedy_excl_unk": og.get("WER_excl_unk"),
        "beam_excl_unk": ob.get("WER_excl_unk"),
        "greedy_exact_pct": og.get("exact_rate_pct"),
        "beam_exact_pct": ob.get("exact_rate_pct"),
        # ⭐ 核心证据：输出长度比（beam 恶化 WER 的直接原因）
        "greedy_out_tokens": sum(len(h) for h in g_hyp),
        "beam_out_tokens": sum(len(h) for h in b_hyp),
        "note": "beam = 自实现标准 CTC 前向；官方 ctcdecode 在 py3.14 装不上",
    }
    outp = REPO / ("artifacts/metrics/blank-gov/p120-beam-%s.json" % args.tag)
    outp.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print("=> 收据 %s" % outp)


if __name__ == "__main__":
    main()
