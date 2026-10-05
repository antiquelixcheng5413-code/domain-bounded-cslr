"""P76：裁决 P50 的 2.8774 到底是不是错的

发现矛盾：
- 仓库里 docs/planning/OFFICIAL_EVAL_PROTOCOL.md 声称
  「P50 报『逐句等权 2.8774』是算错的，真实值是 0.5128」
- 但我今天在多个回复里都引用了 2.8774

必须用 dev 全量实测裁决，不能靠文档说什么。
用 jiwer（第三方库）交叉验证，确保不是我的实现问题。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402
from cslr.recognition.training import decode_batch             # noqa: E402
from p40_rgb_main import DualInputCTC, read_csv, levenshtein   # noqa: E402


def rd(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}


lab_tr = rd(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = rd(REPO / "data/raw/CE-CSL/label/dev.csv")
LM = REPO / "artifacts/part3_features"

voc, _ = build_ordered_vocabulary(lab_tr.values(), min_frequency=2,
                                 max_tokens=300)

# 重跑 P42 模型，拿到真实 hyp
ck = REPO / "artifacts/checkpoints/p42-lm_only-ep100.pt"
blob = torch.load(str(ck), map_location="cpu", weights_only=False)
cfg = blob["config"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = DualInputCTC(lm_dim=cfg["lm_dim"], rgb_dim=cfg["rgb_dim"],
                 vocab=int(voc.size), hidden=cfg["hidden"],
                 layers=cfg["layers"], dropout=cfg["dropout"],
                 use_rgb=cfg["use_rgb"], use_lm=cfg["use_lm"],
                 mode=cfg.get("mode", "add")).to(dev)
m.load_state_dict(blob["model_state"])
m.eval()

items = []
for sid in sorted(lab_dv):
    p = LM / "validation" / (sid + ".landmark.npy")
    if not p.exists():
        continue
    ids = voc.encode(lab_dv[sid])
    if not ids or len(ids) > 24:
        continue
    items.append({"sid": sid, "raw": lab_dv[sid],
                  "ids": ids, "feat": np.load(p).astype(np.float32)})

hyps = []
with torch.no_grad():
    for k in range(0, len(items), 32):
        ch = items[k:k + 32]
        lm = torch.from_numpy(np.stack([c["feat"] for c in ch])).to(dev)
        il = torch.full((len(ch),), lm.shape[1], dtype=torch.long, device=dev)
        lg = m(lm, il, None, None)
        lp = lg.log_softmax(-1).cpu().numpy()
        ol = np.full((lg.size(0),), lg.size(1), dtype=np.int64)
        dec, _, _ = decode_batch(lp, ol, 1)
        for r, s in zip(ch, dec):
            hyps.append((r["ids"], voc.decode(list(s))))

print("=" * 74)
print("裁决：'逐句等权 WER = 2.8774' 是否错误")
print("=" * 74)
print("  dev 样本 %d 条" % len(hyps))

# 🔴 第一版脚本的 bug：用 `voc.decode(ids)` 得到的是**原始 gloss**，
# 但官方口径（与 P40/P42 一致）用的是**折叠后**的参考 —— OOV 映射成 <unk>。
# 折叠后 hyp 与 ref 才有可比性；不折叠会让 hyp 的 OOV 词与 ref 的原词对不上，
# 编辑数暴涨（实测 2963 > token 总数 2838，WER 104% 明显荒谬）。
#⇒ 修正：参考与识别两侧都走 voc.decode（库路径，自带 index_of→unk 折叠）
pairs = []
for r_ids, h_txt in hyps:
    pairs.append((voc.decode(list(r_ids)), h_txt))

# 逐句编辑数与参考长度
per_edit = np.array([levenshtein(list(r), list(h)) for r, h in pairs])
ref_len = np.array([len(r) for r, _ in pairs])
E = int(per_edit.sum())
N = int(ref_len.sum())
n = len(pairs)

print("\n  原始量：")
print("    编辑总数 sum(edit_i) = %d" % E)
print("    参考 token 总数      = %d" % N)
print("    句子数               = %d" % n)

print("\n  各口径的实际值：")
rows = [
    ("① 官方口径（语料级）= sum(ed)/sum(len)", E / N),
    ("② 逐句等权 = mean(ed_i / len_i)", float(np.mean(per_edit / ref_len))),
    ("③ 逐句等权 = mean(ed_i)（错：除句数）", E / n),
    ("④ 逐句等权 = sum(ed)/n（就是③的写法）", E / n),
]
for nm, v in rows:
    print("    %-38s %.4f  (%.2f%%)" % (nm, v, 100 * v))

print("\n  文档声称：逐句等权 = 0.5128，P50 的 2.8774 是把 %d 除以句数 %d" % (E, n))
print("  实测：")
print("    ② mean(ed_i/len_i)          = %.4f  <- 这才是逐句等权" %
      float(np.mean(per_edit / ref_len)))
print("    ③ sum(ed_i) / 句数 %d         = %.4f  <- 这就是 2.8774" % (n, E / n))
print()
print("  ⇒ 文档是对的：2.8774 = %d / %d，**除错了分母**（用了句数而非 token 数）"
      % (E, n))
print("     真正的逐句等权 = %.4f，与 token 级 %.4f **相差仅 %.2f pp**，不是 5.5 倍"
      % (float(np.mean(per_edit / ref_len)), E / N,
         100 * abs(E / N - float(np.mean(per_edit / ref_len)))))

# jiwer 交叉验证
print("\n" + "=" * 74)
print("jiwer 交叉验证（第三方库，防止我的实现有问题）")
print("=" * 74)
try:
    import jiwer
    refs = [["w%d" % i for i in r] for r, _ in pairs]
    hs = [["w%d" % i for i in h] for _, h in pairs]
    num = den = 0
    for r, h in zip(refs, hs):
        o = jiwer.process_words(r, h)
        num += o.substitutions + o.deletions + o.insertions
        den += o.hits + o.substitutions + o.deletions
    print("  jiwer token级(语料) = %.4f" % (num / den))
    print("  我的 token 级       = %.4f" % (E / N))
    print("  一致 = %s" % (abs(num / den - E / N) < 1e-6))
except Exception as exc:
    print("  jiwer 不可用: %s" % exc)

print("\n" + "=" * 74)
print("⚠️ 必须撤回的结论")
print("=" * 74)
print("""
  ❌ 我今天多次说：「逐句等权 WER = 2.8774，与 token 级差 5.5 倍，
     说明短视频错得最狠」—— **这条完全错误**。
     2.8774 = 1479 / 514（编辑总数除以句数），是把分母用错了。
     真正的逐句等权是 0.5128，与 token 级 0.5211 只差 0.83 pp。

  ✅ 仍然成立的：「只看 token 级 0.5211 会掩盖样本级真相」——
     但**靠的是 exact 率 3.5%**（18/514 完全正确），不是逐句差 5.5 倍。

  ✅ 仍然成立的：输出短的句子 unk 占比高（P44 实测 ≥50% unk 的 129/514），
     因为 unk 挤占短句的 token 空间。这解释了用户「网页上看着很差」的直觉。

  ⚠️ 另一个受影响的地方：我写的 MEMORY.md 0.0005 段
     「逐句等权 WER 2.8774 vs token 级 0.5211（差 5.5 倍）」需要修正。
""")

out = {
    "verdict": "文档正确，P50 的 2.8774 是除错分母",
    "edit_total": E,
    "ref_token_total": N,
    "n_sentences": n,
    "official_corpus_wer": round(E / N, 4),
    "per_sentence_mean_ed_over_len": round(
        float(np.mean(per_edit / ref_len)), 4),
    "wrong_ed_div_n": round(E / n, 4),
    "gap_pp": round(100 * abs(E / N - float(np.mean(per_edit / ref_len))), 2),
    "retracted": "『token 级与逐句等权差 5.5 倍』不成立",
    "still_valid": [
        "只看 token 级 0.5211 会掩盖样本级真相 —— 但靠 exact 率 3.5%",
        "短句 unk 占比高（P44: >=50% unk 的 129/514）",
    ],
}
p = REPO / "artifacts/metrics/blank-gov/p76-arbitrate-2877.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)