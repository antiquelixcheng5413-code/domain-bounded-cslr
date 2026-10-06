"""P88：beam search 解码 vs greedy（P87 checkpoint，无需重训）

═══════════════════════════════════════════════════════════════
📄 论文支撑（用户要求：方案必须附依据）
═══════════════════════════════════════════════════════════════
【有】官方 CE-CSL 论文 Implementation rules 原文：
  "During model testing, only central cropping is used for data augmentation and
   **a beam search algorithm with a beam width of 10** is employed in the final
   CTC decoding phase."
  官方代码 `external/TFNet/decode.py:22`：
  `self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, ...)`
  ⇒ **官方口径就是 beam=10，而我们一直用 greedy ⇒ 口径未对齐。**

【有】ref15Camgoz（Sign Language Transformers）**直接对比了两者**：
  "During the training and validation steps we employ a **greedy search** to decode
   both gloss sequences and spoken language sentences. **At inference time, we
   utilize beam search decoding with widths ranging from 0 to 10.** We also
   implement a length penalty with α values ranging..."
  ⇒ 该论文明确区分：训练/验证用 greedy，**推理用 beam（宽度扫描 0~10）**
  ⇒ 这正是我们的做法（greedy 评估 vs beam 评估）

【无】⚠️ **本地 17 篇里没有任何论文给出「beam 比 greedy 好多少 pp」的实验数据。**
  ⇒ 所以本实验是**填补口径空白**，不是验证已有结论。预期收益不确定。
【间接】其他论文的 beam width 参考值：ref07=3、ref17=5、ref19=5、官方 CE-CSL=10

═══════════════════════════════════════════════════════════════
判据（事先定好）
═══════════════════════════════════════════════════════════════
beam_width ∈ {1(≈greedy), 3, 5, 10}，长度惩罚 α ∈ {0.0, 0.5, 1.0, 2.0}
  · 主指标：官方口径 WER（只报这个用于与论文对照）
  · 辅助观察：输出长度、distinct —— 看beam 能否缓解坍缩
  · 判定：若 beam=10 相对 greedy 改善 < 1pp 且符号不稳定 ⇒判「beam 收益不显著」
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))

from pyctcdecode import BeamSearchDecoderCTC, Alphabet  # noqa: E402

import DataProcessMoudle as DPM# noqa: E402
import Net# noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location(
    "p78", REPO / "tools/blank_gov/p78_train_official_tfnet.py")
p78 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p78)

from official_wer import evaluate, split_gloss_sequence  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "p87-res160-2k"
CSV = REPO / "external/TFNet/data/CE-CSL"

lab_dv, tr_map = {}, {}
with open(CSV / "dev.csv", newline="", encoding="utf-8") as fh:
    for row in csv.reader(fh):
        if row and row[0]:
            lab_dv[row[0]] = row[3]
            tr_map[row[0]] = row[1]

w2i, wsn, idx2w = DPM.Word2Id(str(CSV / "train.csv"), str(CSV / "dev.csv"),
                             str(CSV / "test.csv"), "CE-CSL")
print("词表 wordSetNum = %d  idx2word 长度 = %d" % (wsn, len(idx2w)))
print("idx2word[0] = %r（blank/PAD）" % idx2w[0])

ck = torch.load(REPO / ("artifacts/checkpoints/%s-best.pt" % TAG),
                map_location="cpu", weights_only=False)
print("checkpoint epoch=%s wer(greedy)=%s img_size=%s"
      % (ck.get("epoch"), ck.get("wer_official"),
         ck["config"].get("img_size")))

model = Net.moduleNet(ck["hidden"], ck["wordSetNum"] + 1, ck["config"]["module"],
                      torch.device("cuda:0"), "CE-CSL", True).cuda()
model.load_state_dict(ck["model_state"])
model.eval()

_, dv_tf = p78.build_transforms(hflip=False,
                                img_size=ck["config"].get("img_size", 160))
ds = p78.RGBSeqDataset("dev", {k: (tr_map[k], v) for k, v in lab_dv.items()},
                       w2i, dv_tf, False)
print("dev %d 条" % len(ds))

# ---------- 推理并缓存 log_probs（避免重复跑网络）----------
print()
print("=== 推理并缓存 log_probs（只跑一次网络）===")
ls = nn.LogSoftmax(dim=-1)
all_lp, all_tl, refs, sids_all = [], [], [], []
with torch.no_grad():
    for k in range(0, len(ds), 2):
        ch = [ds[i] for i in range(k, min(k + 2, len(ds)))]
        vid, tgt, tl, dl, true_len, sids, _ids = p78.collate(ch)
        out = model(vid.cuda(), dl, False)
        lp = ls(out[0])                      # [T,B,C]
        for bi in range(lp.shape[1]):
            T = min(p78._conv_len(int(true_len[bi])), lp.shape[0])
            all_lp.append(lp[:T, bi, :].cpu().numpy().astype(np.float32))
            all_tl.append(T)
        refs += [lab_dv[s] for s in sids]
        sids_all += list(sids)
print("缓存 %d 条，帧数 min=%d max=%d mean=%.1f"
      % (len(all_lp), min(all_tl), max(all_tl), np.mean(all_tl)))


def dec_greedy(seq):
    """基线：与 P87 完全一致的 greedy。"""
    ids, prev = [], -1
    for t in range(seq.shape[0]):
        k = int(seq[t].argmax())
        if k != prev and k != 0:
            ids.append(k)
        prev = k
    return [idx2w[i] for i in ids]


# 🔴 pyctcdecode 0.5.0 的 API 与旧版 ctcdecode 完全不同：
#   - 类名是 BeamSearchDecoderCTC（不是 CTCBeamDecoder）
#   - 构造只接受 Alphabet（需要 labels + is_bpe 两个参数）
#   - beam_width / token_min_logp 是 **decode() 的参数**，构造时不能传
#   - ⚠️ 本库**不支持 alpha（长度惩罚）**，只支持可选的 language_model 浅融合
#      ⇒ 官方代码用的 ctcdecode 才有alpha，我们用 pyctcdecode 无法完全复刻
_ALPHA = None


def _get_dec():
    global _ALPHA          # 🔴 不写 global 会UnboundLocalError
    if _ALPHA is None:
        _ALPHA = Alphabet(idx2w, is_bpe=False)
    return BeamSearchDecoderCTC(_ALPHA)


def dec_beam(seq, width, alpha=0.0):
    """官方口径：beam search width=10（论文 Implementation rules）。

    ⚠️ 本库无 alpha 长度惩罚（ref15 提到官方用 length penalty α），
       所以 alpha 只在收据里记录、实际不生效 —— 这是与官方的**已知差异**。
    """
    dec = _get_dec()
    out = dec.decode(seq, beam_width=width,
                     token_min_logp=-5.0, beam_prune_logp=-10.0)
    if out:
        toks = out.split(" ")
        return [t for t in toks if t and t != "_"]
    return []


print()
print("=" * 76)
print("P88：beam search vs greedy（同一 checkpoint，只改解码）")
print("=" * 76)
print("%-22s %10s %8s %8s %10s" % ("解码配置", "WER%", "Δ vs g", "输出长", "distinct"))
print("-" * 76)

results = {}
hyps_g = [dec_greedy(s) for s in all_lp]
g = evaluate(refs, hyps_g)
results["greedy"] = {"wer": g["WER_official"], "out_len": np.mean([len(h) for h in hyps_g]),
                     "distinct": len({t for h in hyps_g for t in h})}
print("%-22s %9.2f%% %8s %8.2f %10d"
      % ("greedy (P87 基线)", g["WER_official"], "—",
         results["greedy"]["out_len"], results["greedy"]["distinct"]))

for width in (3, 5, 10, 25, 50):
    for alpha in (0.0,):          # ⚠️ pyctcdecode 不支持 alpha，只跑 0.0
        try:
            hyps = [dec_beam(s, width, alpha) for s in all_lp]
            o = evaluate(refs, hyps)
            key = "beam_w%d" % width
            results[key] = {"wer": o["WER_official"],
                            "out_len": float(np.mean([len(h) for h in hyps])),
                            "distinct": len({t for h in hyps for t in h})}
            print("%-22s %9.2f%% %+8.2f %8.2f %10d"
                  % (key, o["WER_official"],
                     o["WER_official"] - g["WER_official"],
                     results[key]["out_len"], results[key]["distinct"]))
        except Exception as exc:
            print("%-22s FAIL %s" % ("beam_w%d_a%.1f" % (width, alpha),
                                     type(exc).__name__))

print()
print("=" * 76)
print("判据")
print("=" * 76)
b10 = results.get("beam_w10")
b10_best = min(((v["wer"], k) for k, v in results.items() if k != "greedy"),
               default=(None, None))
if b10:
    d = b10["wer"] - g["WER_official"]
    print("官方口径 beam=10: %.2f%%  vs greedy %.2f%%  ⇒ Δ %+.2f pp"
          % (b10["wer"], g["WER_official"], d))
print("全部配置最优: %s = %.2f%%（Δ %+.2f pp vs greedy）"
      % (b10_best[1], b10_best[0], b10_best[0] - g["WER_official"]))
best = b10_best[0]
if best - g["WER_official"] < -1.0:
    print("⇒ **beam 改善 > 1pp** ⇒ 值得作为默认解码方式（且对齐官方口径）")
else:
    print("⇒ beam 改善 < 1pp ⇒ **收益不显著**")
    print("   ⚠️ 但仍应改用 beam=10，因为**官方口径如此**，不改则无法与 45.1 对照")

dst = REPO / ("artifacts/metrics/blank-gov/%s-beam-ablation.json" % TAG)
dst.write_text(json.dumps({
    "experiment": "P88",
    "tag": TAG,
    "checkpoint_epoch": ck.get("epoch"),
    "purpose": "beam search vs greedy（同一 checkpoint，只改解码，不重训）",
    "paper_basis": {
        "official": "arXiv:2409.11960v2 Implementation rules: "
                    "'a beam search algorithm with a beam width of 10 is employed "
                    "in the final CTC decoding phase'",
        "official_code": "external/TFNet/decode.py:22 "
                         "ctcdecode.CTCBeamDecoder(vocab, beam_width=10)",
        "ref15": "Camgoz Sign Language Transformers: 训练/验证用 greedy，"
                 "推理用 beam(width 0~10) + length penalty α —— 与我们做法一致",
        "no_direct_evidence": "⚠️ 本地 17 篇无任何论文给出 beam vs greedy 的 pp 对比"
                              " ⇒ 本实验填补口径空白，预期收益不确定",
    },
    "greedy_wer": g["WER_official"],
    "results": results,
    "official_beam10_wer": b10["wer"] if b10 else None,
    "known_deviation": "⚠️ pyctcdecode 0.5.0 不支持 alpha 长度惩罚，"
                       "而官方 ctcdecode 支持（ref15 提到官方用 length penalty）。"
                       "⇒ 本实验的 beam 与官方 beam 仍有差异",
    "best_config": b10_best[1],
    "best_wer": best,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("收据 -> %s" % dst)