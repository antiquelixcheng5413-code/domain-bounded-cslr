"""P133：误差构成诊断 —— 到底是"说太少"、"说太多"还是"说错了"？

动机（回答"下一步试什么"必须先有这个数）：
- P123 best：ref 2842 token / 515 句，edits 1587 ⇒ 只知总量，不知构成
- P120 曾记录：greedy 输出 1892 token vs 参考 2842 ⇒ 疑似**严重欠生成**
  若成立，则 deletion 是主要误差，对策完全不同于"说错"（sub 为主）

三种误差对应的对策完全不同：
  以 del 为主（说太少）⇒ 抑制 blank / 长度奖励 / 降低 CTC 的 blank 偏置
  以 ins 为主（说太多）⇒ 长度惩罚 / 提高 blank
  以 sub 为主（说错）  ⇒ 表征/判别力问题（那才该去动模型）

输出：
  - ins / del / sub 的绝对数与占比
  - 输出 token 总数 vs 参考 token 总数（生成率）
  - 长度分布：按参考句长分桶看 WER 与生成率
  - exact 率

⚠️ 已知答案校验：ins+del+sub 必须等于 official_wer 的 edits(1587)，
   语料级 WER 必须复现 55.84。
"""
from __future__ import annotations

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
from official_wer import evaluate, wer_official  # noqa: E402
from official_wer import split_gloss_sequence, split_glued, CFG  # noqa: E402

CK = REPO / "artifacts/checkpoints/p123-official55bf16-best.pt"
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515


def log(*a):
    print("[P133 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def align(ref, hyp):
    """编辑距离 DP + 回溯，返回 (ins, del, sub)。

    ins：hyp 多出（插入错误）
    del：ref 有而 hyp 没有（漏词）
    sub：替换
    """
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1,      # del
                          D[i][j - 1] + 1,      # ins
                          D[i - 1][j - 1] + cost)
    ins = dele = sub = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if D[i][j] == D[i - 1][j - 1] + cost:
                if cost:
                    sub += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and D[i][j] == D[i - 1][j] + 1:
            dele += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return ins, dele, sub


def main() -> int:
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    P.OFFICIAL_PAD = True
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "键名变了：%s" % list(ck)[:8]
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s）" % (CK.name, ck.get("epoch")))

    _, dv_tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if MAX_DEV:
        ds.items = ds.items[:MAX_DEV]
    ls = torch.nn.LogSoftmax(dim=-1)
    hyps, refs = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to(dev), dl, False)
            hyps += P.greedy_decode(ls(out[0]), dl, idx2word)
            for b in sids:
                refs.append(labels_dv[b][1])
    log("推理 %.0fs" % (time.time() - t0))

    o = evaluate(refs, hyps)
    log("官方语料级 WER = %.2f%%  (edits=%d, ref=%d)"
        % (o["WER_official"], o["edits"], o["ref_tokens"]))

    tot_ins = tot_del = tot_sub = 0
    n_hyp = n_ref = 0
    exact = 0
    buckets = {}
    for r, h in zip(refs, hyps):
        rtok = split_glued(split_gloss_sequence(r, CFG))
        htok = split_glued(list(h))
        i_, d_, s_ = align(rtok, htok)
        tot_ins += i_
        tot_del += d_
        tot_sub += s_
        n_hyp += len(htok)
        n_ref += len(rtok)
        if rtok == htok:
            exact += 1
        L = len(rtok)
        k = "1-3" if L <= 3 else ("4-6" if L <= 6 else ("7-9" if L <= 9 else "10+"))
        b = buckets.setdefault(k, {"n": 0, "ins": 0, "del": 0, "sub": 0,
                                   "ref": 0, "hyp": 0, "exact": 0})
        b["n"] += 1
        b["ins"] += i_
        b["del"] += d_
        b["sub"] += s_
        b["ref"] += len(rtok)
        b["hyp"] += len(htok)
        b["exact"] += (rtok == htok)

    tot = tot_ins + tot_del + tot_sub
    print()
    log("──── 误差构成 ────")
    log("  替换 sub : %5d  (%5.1f%% 的编辑)" % (tot_sub, 100.0 * tot_sub / max(tot, 1)))
    log("  漏词 del : %5d  (%5.1f%%)" % (tot_del, 100.0 * tot_del / max(tot, 1)))
    log("  多说 ins : %5d  (%5.1f%%)" % (tot_ins, 100.0 * tot_ins / max(tot, 1)))
    log("  合计     : %5d  （official_wer 的 edits=%d ⇒ %s）"
        % (tot, o["edits"], "✅ 一致" if tot == o["edits"] else "❌ 不一致"))
    print()
    log("──── 生成量 ────")
    log("  参考 token 总数 : %d" % n_ref)
    log("  输出 token 总数 : %d" % n_hyp)
    log("  生成率          : %.1f%%  （<100%% ⇒ 欠生成；>100%% ⇒ 过生成）"
        % (100.0 * n_hyp / max(n_ref, 1)))
    log("  整句完全一致    : %d / %d = %.2f%%" % (exact, len(refs), 100.0 * exact / len(refs)))
    print()
    log("──── 按参考句长分桶 ────")
    log("  句长   句数   sub   del   ins   生成率   该桶WER")
    for k in ["1-3", "4-6", "7-9", "10+"]:
        if k not in buckets:
            continue
        b = buckets[k]
        e = b["sub"] + b["del"] + b["ins"]
        log("  %-5s %5d %5d %5d %5d   %6.1f%%   %6.2f%%"
            % (k, b["n"], b["sub"], b["del"], b["ins"],
               100.0 * b["hyp"] / max(b["ref"], 1),
               100.0 * e / max(b["ref"], 1)))

    if tot_del > tot_sub and tot_del > tot_ins:
        verdict = ("**以漏词 del 为主** ⇒ 模型『说太少』（生成率 %.1f%%）\n"
                   "          ⇒ 对策：抑制 blank / 长度奖励 / 降低 CTC 的 blank 偏置，\n"
                   "             而不是去动模型表征" % (100.0 * n_hyp / max(n_ref, 1)))
    elif tot_ins > tot_sub and tot_ins > tot_del:
        verdict = "**以多说 ins 为主** ⇒ 对策：长度惩罚 / 提高 blank 偏置"
    else:
        verdict = "**以替换 sub 为主** ⇒ 是判别力问题，才该去动模型/表征"
    print()
    log("判定：%s" % verdict)

    out = REPO / "artifacts/metrics/blank-gov/p133-error-breakdown.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "P133 误差构成诊断（ins/del/sub + 生成率）",
        "ckpt": str(CK), "epoch": ck.get("epoch"), "n_dev": len(ds),
        "WER_official": o["WER_official"], "edits": o["edits"],
        "ref_tokens": o["ref_tokens"],
        "sub": tot_sub, "del": tot_del, "ins": tot_ins,
        "sum_check_ok": tot == o["edits"],
        "pct_sub": round(100.0 * tot_sub / max(tot, 1), 1),
        "pct_del": round(100.0 * tot_del / max(tot, 1), 1),
        "pct_ins": round(100.0 * tot_ins / max(tot, 1), 1),
        "hyp_tokens": n_hyp, "ref_tokens_sum": n_ref,
        "generation_rate_pct": round(100.0 * n_hyp / max(n_ref, 1), 1),
        "exact": exact, "exact_pct": round(100.0 * exact / len(refs), 2),
        "buckets": buckets,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
