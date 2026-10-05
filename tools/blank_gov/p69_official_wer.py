"""P69：对齐官方 WER 定义（arXiv:2409.11960v2 式 11）

官方原文：
    WER = 100% × (ins + del + subs) / sum
    "where ins represents the number of words to be inserted,
     del ... deleted, subs ... replaced,
     and sum represents the total number of words in the label."

关键点（与 P50 的"逐句等权"**不同**）：
  - 分母 sum = **整个语料的参考 token 总数**（corpus-level）
  - 分子 = 语料上全部编辑数之和
  - 这就是 **token 级 WER**，也是我一直报的 0.5211

⚠️ 因此 P50 那条"必须报逐句等权"的铁律需要修正：
   逐句等权不是官方口径，是**补充诊断指标**。
   官方判优只看 token 级 WER。

另一个关键：官方**没有定义**下列处理（论文里未提及）：
  - 是否移除标点
  - 是否移除重复 token
  - 是否大小写归一
  - 是否移除空格
⇒ 严格按官方口径 = 不做任何额外处理。

但官方 Table VII 的定性示例里含标点：
    Gloss ground truth: 他/小孩时间/开始/做/律师/希望/。
    Pred. (SEN): 他/时间/开始/。  WER 50.0
  ⇒ **标点保留在参考里**（`。` 被计入分母）
  验算：ref=7 token，hyp=4 token，
        编辑 = 删3 + 替1 = ... 让我实际算一下。

本脚本做三件事：
  1. 用官方式(11) 重算 P42 模型的 dev WER，与 0.5211 对齐验证
  2. 用 Table VII 的定性样例验证标点是否计入
  3. 确认官方口径与 P50 四件套的关系，给出新的报数规范
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

from cslr.recognition.gloss_sequence import (  # noqa: E402
    build_ordered_vocabulary, split_gloss_sequence, GlossSequenceConfig)
from p40_rgb_main import levenshtein, DualInputCTC, read_csv  # noqa: E402
import torch                                                # noqa: E402
from cslr.recognition.training import decode_batch           # noqa: E402

cfg = GlossSequenceConfig()
lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
FEAT = REPO / "artifacts/part3_features/validation"

print("=" * 74)
print("1. 用官方式(11) 验证标点是否计入分母")
print("=" * 74)
print("   官方 Table VII 定性样例（Test-Case-1，标注的 WER=50.0）")
ref = "他/小孩时间/开始/做/律师/希望/。".split("/")
hyp = "他/时间/开始/。".split("/")
e_full = levenshtein(ref, hyp)
ref_np = [t for t in ref if t != "。"]
hyp_np = [t for t in hyp if t != "。"]
e_nopunct = levenshtein(ref_np, hyp_np)
print("   ref = %s  (n=%d)" % (ref, len(ref)))
print("   hyp = %s  (n=%d)" % (hyp, len(hyp)))
print("   编辑数 = %d" % e_full)
print("   含标点：%d/%d = %.1f pct" % (e_full, len(ref),
                                  100 * e_full / len(ref)))
print("   去标点：%d/%d = %.1f pct" % (e_nopunct, len(ref_np),
                                  100 * e_nopunct / len(ref_np)))

# 官方标注 50.0 —— 反推真实 token 数
# 50.0% = 4/N  ->  N = 8   （4/8 = 50.0%）
#   ⇒ ref 实际是 **8** 个 token，不是 7
#   ⇒ 「小孩时间」应是两个 gloss：小孩 + 时间
print("\n   反推官方分母：编辑数=4，标注 50.0%% => 分母 N = 4/0.50 = **8**")
print("   而我按/ 切分得到 7 个⇒ 差1 个")
print("   ⇒ 官方把 `小孩时间` 视为 **两个 gloss（小孩 + 时间）**")
ref8 = "他/小孩/时间/开始/做/律师/希望/。".split("/")
hyp8 = "他/时间/开始/。".split("/")
e8 = levenshtein(ref8, hyp8)
print("\n   用 8-token 参考重算：")
print("     ref = %s  (n=%d)" % (ref8, len(ref8)))
print("     编辑 = %d-> %d/%d = **%.1f%%**  官方标注 50.0%%✓"
      % (e8, e8, len(ref8), 100 * e8 / len(ref8)))

print("\n   验证第二个样例（TFNet WER=0.0）")
ref2 = "他/小孩时间/开始/做/律师/希望/。".split("/")
hyp2 = "他/小孩时间/开始/做/律师/希望/。".split("/")
print("   编辑数 = %d -> %.1f%%" % (levenshtein(ref2, hyp2),
                              100 * levenshtein(ref2, hyp2) / len(ref2)))
print("   官方标注 0.0%% ✓")

print("\n   第三个样例（CorrNet WER=25.0）")
ref3 = "他/小孩时间/开始/做/律师/希望/。".split("/")
hyp3 = "他时间/开始/做/希望/。".split("/")
print("   hyp = %s  编辑 = %d -> %.1f%%"
      % (hyp3, levenshtein(ref3, hyp3), 100 * levenshtein(ref3, hyp3) / len(ref3)))

print("\n" + "=" * 74)
print("2. 用官方口径重算 P42 在 dev 上的 WER")
print("=" * 74)

voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                  max_tokens=300)
items = []
for sid in sorted(lab_dv):
    p = FEAT / (sid + ".landmark.npy")
    if not p.exists():
        continue
    toks = split_gloss_sequence(lab_dv[sid], cfg)
    ids = voc.encode(lab_dv[sid])          # 库路径（内部归一化）
    if not ids or len(ids) > 24:
        continue
    items.append({"sid": sid,
                  "lm": np.load(p).astype(np.float32),
                  "ref": voc.decode(list(ids))})   # 折叠后的参考

blob = torch.load(REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt",
                  map_location="cpu", weights_only=False)
c = blob["config"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = DualInputCTC(lm_dim=c["lm_dim"], rgb_dim=c["rgb_dim"], vocab=int(voc.size),
                 hidden=c["hidden"], layers=c["layers"], dropout=c["dropout"],
                 use_rgb=c.get("use_rgb", False), use_lm=c.get("use_lm", True),
                 mode=c.get("mode", "add")).to(dev)
m.load_state_dict(blob["model_state"])
m.eval()

recs = []
with torch.no_grad():
    for k in range(0, len(items), 32):
        ch = items[k:k + 32]
        lm = torch.from_numpy(np.stack([s["lm"] for s in ch])).to(dev)
        il = torch.full((len(ch),), lm.shape[1], dtype=torch.long, device=dev)
        lg = m(lm, il, None, None)
        lp = torch.log_softmax(lg.float(), dim=-1).cpu().numpy()
        ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
        dec, _, _ = decode_batch(lp, ol, 1)
        for r, seq in enumerate(dec):
            recs.append({"sid": ch[r]["sid"],
                         "ref": ch[r]["ref"],
                         "hyp": voc.decode(list(seq))})

# 官方式(11)：corpus-level
E = sum(levenshtein(r["ref"], r["hyp"]) for r in recs)
N = sum(len(r["ref"]) for r in recs)
wer_official = E / N
print("  样本 %d  参考 token 总数 %d  编辑总数 %d"
      % (len(recs), N, E))
print("  【官方式(11)】WER = %d/%d = **%.4f**" % (E, N, wer_official))
print("  收据里报的 token 级 = 0.5211")
print("  => %s" % ("**完全一致**" if abs(wer_official - 0.5211) < 0.0005
                else "⚠️ 有差异 %.4f" % wer_official))

# 补充诊断（P50 的其他口径，明确标注为非官方）
hit = sum(1 for r in recs if levenshtein(r["ref"], r["hyp"]) == 0)
avg_sent = float(np.mean([levenshtein(r["ref"], r["hyp"]) / max(len(r["ref"]), 1)
                          for r in recs]))
print("\n  【非官方 · 补充诊断】")
print("    逐句等权 WER = %.4f  （官方未定义，仅诊断用）" % avg_sent)
print("    exact 率= %d/%d = %.1f%%（官方未定义）"
      % (hit, len(recs), 100 * hit / len(recs)))
#剔 unk
E2 = N2 = 0
for r in recs:
    ref_iv = [t for t in r["ref"] if t != "<unk>"]
    if not ref_iv:
        continue
    E2 += levenshtein(ref_iv, [t for t in r["hyp"] if t != "<unk>"])
    N2 += len(ref_iv)
print("    剔 unk WER   = %.4f（官方未定义）" % (E2 / max(N2, 1)))

out = {
    "official_formula": "WER = 100% * (ins + del + subs) / sum  "
                        "[arXiv:2409.11960v2 式 11]",
    "official_denominator": "sum = 整个语料的参考 token 总数（corpus-level）",
    "punctuation_included": True,
    "punctuation_evidence": {
        "case1_ref_tokens": len(ref), "case1_edits": e_full,
        "computed": round(100 * e_full / len(ref), 1), "official": 50.0},
    "our_dev_wer_official_formula": round(wer_official, 4),
    "receipt_wer": 0.5211,
    "match": abs(wer_official - 0.5211) < 0.0005,
    "supplementary_diagnostics": {
        "per_sentence_average": round(avg_sent, 4),
        "exact_rate": round(hit / len(recs), 4),
        "no_unk_wer": round(E2 / max(N2, 1), 4),
        "note": "官方未定义，仅作诊断，不作为判优依据",
    },
    "revised_rule": "官方判优只看 token 级 WER（corpus-level）；"
                    "P50 的『四件套』降级为『主指标 + 补充诊断』",
}
q = REPO / "artifacts/metrics/blank-gov/p69-official-wer.json"
q.parent.mkdir(parents=True, exist_ok=True)
q.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % q)