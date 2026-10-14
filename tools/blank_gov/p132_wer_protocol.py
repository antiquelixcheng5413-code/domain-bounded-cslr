"""P132：WER 口径核对 —— 语料级 vs 逐句等权（官方代码实际用的口径）。

🔴 发现（读官方源码得到，不是推测）：
  - `Train.py:97 / 99`：**`validLoader` 与 `testLoader` 的 batch_size = 1**
    （只有 `trainLoader` 是 config 里的 2）
  - `WER.py:145 WerScore(...)`：对 batch 内每个样本单独算 WER，返回 `werScoreSum/batchSize`
  - `Train.py`(验证循环)：`werScoreSum += werScore` … `werScore = werScoreSum / len(validLoader)`
  ⇒ batchSize=1 时 `len(validLoader)` == 样本数
  ⇒ **官方代码最终报的 WER = 每个句子单独算 WER，再算术平均（逐句等权）**

而本项目一直用的是**语料级**（CE-CSL 论文式 11：总编辑数 / 总参考 token）：
  MEMORY 1 明确记着「逐句等权 / exact / 剔 unk **降级为内部诊断**」
  ⇒ 也就是说我们一直把官方代码的口径当成"非官方"而只做内部诊断。

若两者差异显著，则 10.74 pp 缺口中**有很大一部分可能只是口径不同**，
而不是模型真的差那么多。

本脚本：同一个 checkpoint、同一批 dev，同时输出三种口径：
  A. 语料级（现行，MEMORY 认定的官方口径）
  B. 逐句等权（**官方代码实际口径**）
  C. 逐句等权（排除 ref 为空的句子，防除零）

⚠️ 已知答案校验：A 必须复现 **55.84**（P123 收据值），否则路径不一致。
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

CK = REPO / "artifacts/checkpoints/p123-official55bf16-best.pt"
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515


def log(*a):
    print("[P132 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def lev_edits(ref, hyp):
    """标准编辑距离，返回 (edits, ref_len)。ref/hyp 为 token 序列。"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return (m, 0)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + cost)  # substitution
        prev = cur
    return (prev[m], n)


def main() -> int:
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    P.OFFICIAL_PAD = True          # P123 是补帧训练的 ⇒ 评估同口径
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "ckpt 键名变了：%s" % list(ck)[:8]
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s，strict=True OK）" % (CK.name, ck.get("epoch")))

    _, dv_tf = P.build_transforms(hflip=False, img_size=160)
    ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if MAX_DEV:
        ds.items = ds.items[:MAX_DEV]
    log("dev %d 条" % len(ds))

    ls = torch.nn.LogSoftmax(dim=-1)
    hyps, refs = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _tgt, _tl, dl, _tl_len, sids, _ids = batch
            vid = vid.to(dev)
            out = model(vid, dl, False)
            logp = ls(out[0])
            hyps += P.greedy_decode(logp, dl, idx2word)
            for b in sids:
                refs.append(labels_dv[b][1])
    log("推理完成 %.0fs" % (time.time() - t0))

    # ---------- A. 语料级（现行口径）----------
    oA = evaluate(refs, hyps)
    log("A 语料级      WER = %.2f%%  (edits=%d, ref=%d)"
        % (oA["WER_official"], oA.get("edits", -1),
           oA.get("n", oA.get("ref_tokens", -1))))

    # ---------- B / C. 逐句等权（官方代码口径）----------
    # 🔴 必须复用 official_wer 的 token 化（split_gloss_sequence + split_glued 两侧拆），
    #    不能自己 split("/") —— 我第一版就是这么错的（自算语料级 57.67 ≠ 官方评估 55.84）
    per, per_edits, per_n = [], [], []
    skipped = 0
    for r, h in zip(refs, hyps):
        w_i, e_i, n_i = wer_official([r], [h])
        if n_i == 0:
            skipped += 1
            continue
        per.append(w_i)
        per_edits.append(e_i)
        per_n.append(n_i)
    B = sum(per) / len(per)
    A_self = 100.0 * sum(per_edits) / max(sum(per_n), 1)
    # 自实现（朴素 split("/")）只作对照，说明差异来源
    naive_e = naive_n = 0
    for r, h in zip(refs, hyps):
        rtok = r.split("/") if isinstance(r, str) else list(r)
        htok = h.split("/") if isinstance(h, str) else list(h)
        e, n = lev_edits(rtok, htok)
        naive_e += e
        naive_n += n
    log("  （对照）朴素 split('/') 语料级 = %.2f%% ⇒ 与官方评估的差来自分词/粘连拆分"
        % (100.0 * naive_e / max(naive_n, 1)))

    log("B 逐句等权    WER = %.2f%%  (n=%d 句，跳过 %d 句空参考)" % (B, len(per), skipped))
    log("A' 自算语料级 WER = %.2f%%  (edits=%d, ref=%d) ← 应与 A 一致"
        % (A_self, sum(per_edits), sum(per_n)))

    gap = oA["WER_official"] - B
    print()
    log("──── 结论 ────")
    log("  语料级（我们一直用的）      : %.2f%%" % oA["WER_official"])
    log("  逐句等权（官方代码实际用的）: %.2f%%" % B)
    log("  差（语料级 − 逐句等权）     : **%+.2f pp**" % gap)
    log("  官方 VAC dev 基线            : 45.10%")
    log("  若按逐句等权口径，我们距官方 : %.2f pp"
        % (B - 45.10))

    ok = abs(oA["WER_official"] - 55.84) < 0.02
    log("  已知答案校验：A 应=55.84 ⇒ %s" % ("✅ 通过" if ok else "❌ 不一致"))

    out = REPO / "artifacts/metrics/blank-gov/p132-wer-protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "P132 WER 口径核对：语料级 vs 逐句等权（官方代码实际口径）",
        "finding": "官方 validLoader/testLoader batch_size=1，WerScore 逐样本算 WER，"
                   "最后 werScoreSum/len(loader) ⇒ 实为逐句等权平均",
        "source": {"Train.py": "validLoader batch_size=1 (line 97)",
                   "WER.py": "WerScore line 145-161",
                   "Train.py_eval": "werScore = werScoreSum / len(validLoader)"},
        "ckpt": str(CK), "epoch": ck.get("epoch"), "n_dev": len(ds),
        "A_corpus_level_pct": oA["WER_official"],
        "B_sentence_mean_pct": round(B, 2),
        "A_selfcheck_pct": round(A_self, 2),
        "delta_A_minus_B_pp": round(gap, 2),
        "gap_to_official_45_10_if_B_pp": round(B - 45.10, 2),
        "n_sentences": len(per), "n_skipped_empty_ref": skipped,
        "known_answer_check_passed": ok,
        "note": "逐句等权给短句更大权重；若短句更易识别则该口径系统性偏低",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
