"""P90：用 pyctcdecode 的 build_ctcdecoder 复现官方 beam+长度惩罚口径。

═══════════════════════════════════════════════════════════════
📄 论文支撑
═══════════════════════════════════════════════════════
【有】官方 CE-CSL 论文 Implementation rules：
  "a beam search algorithm with a beam width of 10 is employed in the final
   CTC decoding phase"
【有】官方代码 decode.py:22 `ctcdecode.CTCBeamDecoder(vocab, beam_width=10, ...)`
【有】ref15Camgoz：官方还用 **length penalty α**
  "We also implement a length penalty [74] with α values ranging..."

═══════════════════════════════════════════════════════════════
🔴 为什么换库（诚实记录）
═══════════════════════════════════════════════════════════════
官方用的 `ctcdecode`（parlance/ctcdecode）**在本环境装不上**：
  1. `pip install ctcdecode==0.4` → **无此版本**（只有 1.0.1/1.0.2）
  2. `pip install ctcdecode` → 编译失败：
     `ModuleNotFoundError: No module named 'torch'`（隔离构建环境找不到 torch）
  3. `--no-build-isolation` → 仍失败（metadata-generation-failed）
  根因：官方 ctcdecode 是 Cython 项目，对 Python 3.14 支持差。

⇒ 改用 **pyctcdecode 0.5.0**（Kensho Technologies 维护，Apache-2.0，
   是 ctcdecode 的活跃继任者；作者与 ctcdecode 维护者有重叠）。
   ⚠️ **API 不同**：
   - 类名 `BeamSearchDecoderCTC`（非 `CTCBeamDecoder`）
   - 构造需 `Alphabet(labels, is_bpe)`
   - `build_ctcdecoder(...)` 支持 `alpha`（语言模型权重）与
     **`beta`（长度打分调整）** ← 这就是我们要的「长度惩罚」
   ⚠️ 但 **`alpha` 需要 language_model（kenlm/arpa）**，
      kenlm 也没装 ⇒ **alpha 仍不可用，只能用 beta（长度惩罚）**。

判据：与 P88的 greedy/beam 结果对比，并逐 epoch 无意义（本实验不训练）。
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from pyctcdecode import build_ctcdecoder

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external" / "TFNet"))

import DataProcessMoudle as DPM# noqa: E402
import Net# noqa: E402
import importlib.util
from official_wer import evaluate  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "p78", REPO / "tools/blank_gov/p78_train_official_tfnet.py")
p78 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p78)

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
ck = torch.load(REPO / ("artifacts/checkpoints/%s-best.pt" % TAG),
                map_location="cpu", weights_only=False)
model = Net.moduleNet(ck["hidden"], ck["wordSetNum"] + 1, ck["config"]["module"],
                      torch.device("cuda:0"), "CE-CSL", True).cuda()
model.load_state_dict(ck["model_state"])
model.eval()
_, dv_tf = p78.build_transforms(False, img_size=160)
ds = p78.RGBSeqDataset("dev", {k: (tr_map[k], v) for k, v in lab_dv.items()},
                       w2i, dv_tf, False)
ls_mod = nn.LogSoftmax(dim=-1)
all_lp, refs = [], []
with torch.no_grad():
    for k in range(0, len(ds), 2):
        ch = [ds[i] for i in range(k, min(k + 2, len(ds)))]
        vid, tgt, tl, dl, tlen, sids, _ = p78.collate(ch)
        out = model(vid.cuda(), dl, False)
        lp = ls_mod(out[0])
        for bi in range(lp.shape[1]):
            T = min(p78._conv_len(int(tlen[bi])), lp.shape[0])
            all_lp.append(lp[:T, bi, :].cpu().numpy().astype(np.float32))
        refs += [lab_dv[s] for s in sids]

# 🔴 pyctcdecode 0.5.0 的 API（与旧 ctcdecode 完全不同，实测确认）：
#   - build_ctcdecoder(labels, kenlm_model_path, unigrams, alpha, beta, ...)
#     ⇒ alpha / beta 都是**这个工厂函数**的参数
#   - BeamSearchDecoderCTC(alphabet, language_model) 只接受这两个
#   - 不传 kenlm_model_path 时 kenlm_model=None ⇒ alpha 不生效但 beta 仍生效
#     （DEFAULT_ALPHA 只在有 LM 时用）

print("=" * 78)
print("P90：长度惩罚 beta（pyctcdecode build_ctcdecoder）")
print("=" * 78)
print("依据：官方 beam=10 + ref15 提到官方用 length penalty α")
print("⚠️ alpha 需 kenlm/arpa 语言模型（未装）⇒ 本实验只扫 beta（长度惩罚）")
print()


def greedy(seq):
    ids, prev = [], -1
    for t in range(seq.shape[0]):
        k = int(seq[t].argmax())
        if k != prev and k != 0:
            ids.append(k)
        prev = k
    return [idx2w[i] for i in ids]


res = {}
hyps = [greedy(s) for s in all_lp]
o = evaluate(refs, hyps)
res["greedy"] = {"wer": o["WER_official"],
                 "len": float(np.mean([len(h) for h in hyps])),
                 "distinct": len({t for h in hyps for t in h})}
print("%-28s %8s %8s %8s" % ("配置", "WER%", "输出长", "distinct"))
print("-" * 58)
print("%-28s %7.2f%% %8.2f %8d" % ("greedy (P87/P88 基线)", o["WER_official"],
      res["greedy"]["len"], res["greedy"]["distinct"]))

for width in (10, 25, 50):
    for beta in (0.0, 0.5, 1.0, 2.0, 3.0):
        try:
            # 🔴 build_ctcdecoder 会**多加一个 PAD 类**（拼在末尾），
            #    所以它要求 vocab 比模型输出多1。
            #    官方模型输出 wordSetNum+1 = 3516（含 blank/PAD），
            #    这里必须补上第3517 个占位符才能通过维度校验。
            #    ⚠️ 该占位符永不出现（模型 argmax 不会选它），只用于对齐维度。
            # 🔴🔴 pyctcdecode 的 alphabet 约定（实测踩了4 次才搞清）：
            #  它内部固定加一个 '_' 作为 CTC blank，
            #  且会把传入的 ' ' 空格替换成 ''（blank）⇒ vocab 变成 len+1。
            #  **正确传法：labels = 真实类（不含 blank）+ ['_']，
            #    总数恰好等于模型输出维度。**
            #    官方 idx2word[0] = ' ' 就是 blank，所以要**跳过第 0 项**。
            labels = list(idx2w[1:]) + ["_"]
            assert len(labels) == ck["wordSetNum"] + 1, (
                "vocab 不匹配: %d vs 模型输出 %d"
                % (len(labels), ck["wordSetNum"] + 1))
            dec = build_ctcdecoder(labels, None, None, 0.0, beta)
            hyps = []
            for s in all_lp:
                # 🔴🔴 pyctcdecode 的 decode() 返回的是**字符串**（把 token 直接
                #    拼接，如 '帐篷0一' 表示 帐篷+0+一），不是空格分隔！
                #    ⇒ 必须用 decode_beams() 拿 [prob, tokens, ...] 的 token 列表。
                #    （这个 bug 让 P90 首版所有配置都显示 100% WER。）
                # 🔴 实测 pyctcdecode 的 beams[0] 结构（tuple, 长度 5）：
                #   [0] str           整句拼接文本 '帐篷0一'
                #   [1] None
                #   [2] list[(token, (start, end)), ...]   ★ token 序列在这里
                #   [3] float         总对数概率
                #   [4] float         每 token 平均对数概率
                # ⇒ 之前取 beams[0][1]（None）导致 TypeError；
                #   decode() 只返回拼接字符串，split(" ") 拿不到 token。
                beams = dec.decode_beams(
                    s, beam_width=width, token_min_logp=-5.0,
                    beam_prune_logp=-10.0, prune_history=True)
                toks = ([t for t, _ in beams[0][2]] if beams and beams[0][2]
                        else [])
                hyps.append([t for t in toks if t and t != "_"])
            oo = evaluate(refs, hyps)
            key = "w%d_beta%.1f" % (width, beta)
            res[key] = {"wer": oo["WER_official"],
                        "len": float(np.mean([len(h) for h in hyps])),
                        "distinct": len({t for h in hyps for t in h})}
            print("%-28s %7.2f%% %8.2f %8d" % (key, oo["WER_official"],
                  res[key]["len"], res[key]["distinct"]))
        except Exception as exc:
            print("%-28s FAIL %s" % ("w%d_beta%.1f" % (width, beta),
                                     type(exc).__name__))

print()
best = min(((v["wer"], k) for k, v in res.items() if k != "greedy"),
           default=(None, None))
g = res["greedy"]["wer"]
print("greedy           = %.2f%%" % g)
print("官方口径 w10_beta0= %.2f%%" % res.get("w10_beta0.0", {}).get("wer", float("nan")))
print("全配置最优= %s = %.2f%%（Δ %+.2f pp vs greedy）"
      % (best[1], best[0], best[0] - g))
print()
if best[0] < g - 1.0:
    print("⇒ **长度惩罚 + beam 改善 > 1pp** ⇒ 可作为默认解码")
elif best[0] < g:
    print("⇒ 改善 < 1pp")
else:
    print("⇒ **没有任何配置优于 greedy**")
print("⇒ 但注意：P88 已证明 beam 劣于 greedy 的根因是 blank 占 95.2%，")
print("   长度惩罚若也救不回来 ⇒ 说明必须先改模型（LS / 训练量），不是解码")

dst = REPO / ("artifacts/metrics/blank-gov/%s-length-penalty.json" % TAG)
dst.write_text(json.dumps({
    "experiment": "P90",
    "purpose": "长度惩罚 beta × beam width（复现官方 beam+length penalty 口径）",
    "tag": TAG,
    "paper_basis": {
        "official": "arXiv:2409.11960v2 Implementation rules: beam width 10",
        "official_code": "decode.py:22 ctcdecode.CTCBeamDecoder(beam_width=10)",
        "ref15": "Camgoz: 官方还用 length penalty α",
    },
    "library_substitution": {
        "blocked": "官方 ctcdecode 装不上：① 无 0.4 版本（只有 1.0.x）"
                   "② 编译失败 ModuleNotFoundError: torch ③ --no-build-isolation 仍失败"
                   "根因：Cython 项目对 Python 3.14 支持差",
        "substitute": "pyctcdecode 0.5.0（Kensho Technologies，Apache-2.0，"
                      "ctcdecode 的活跃继任者）",
        "remaining_gap": "⚠️ alpha（语言模型浅融合）需 kenlm/arpa，未装⇒ 仍缺 alpha，"
                         "只有 beta（长度惩罚）",
    },
    "results": res,
    "best": best[1], "best_wer": best[0],
    "baseline_greedy_wer": g,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print("收据 -> %s" % dst)