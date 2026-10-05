"""P60：中科大/干净背景数据集能否与 CE-CSL 混用

分三层论证，每层都有可查证的事实而非推测：
  L1 词表层：标签体系是否对齐（决定有无共享 label space 的可能）
  L2 协议层：官方 benchmark 是怎么做的（决定混用会不会让数字不可比）
  L3 统计层：混合后 signer/背景/样本分布会变成什么样

注意：本地只有 CE-CSL，中科大/ CSL-Daily 数据**不在本地**，
所以 L1 只做 CE-CSL 侧的静态统计 + 引用论文报告的对方数字。
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "src"))

from cslr.recognition.gloss_sequence import build_ordered_vocabulary  # noqa: E402


def read_csv(p: Path) -> dict[str, str]:
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["Number"]: r["Gloss"] for r in csv.DictReader(fh)}


def toks(g: str) -> list[str]:
    return [t.strip() for t in g.split("/") if t.strip()]


lab_tr = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
lab_dv = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
lab_te = read_csv(REPO / "data/raw/CE-CSL/label/test.csv")

print("=" * 72)
print("本地数据实况（确认只有 CE-CSL）")
print("=" * 72)
raw = sorted((REPO / "data/raw").iterdir())
print("  data/raw/ 下: %s" % [d.name for d in raw])
print("  => 中科大 / CSL-Daily / SLR500 **本地都没有**")

tr_tok = collections.Counter()
for g in lab_tr.values():
    tr_tok.update(toks(g))

print("\n" + "=" * 72)
print("L1 词表层：标签体系能否对齐")
print("=" * 72)

# 论文报告的对方数字（arXiv:2409.11960v2 + USTC 官网）
others = {
    "CE-CNSL（中科大不是这个）": {
        "vocab": 3515, "dev_oov": 0, "modality": "RGB 连续句",
        "signers": None, "env": "70+ 复杂背景",
    },
    "CSL-Daily": {
        "vocab": 2000, "dev_oov": None, "modality": "RGB 连续句",
        "signers": 10, "env": "室内/相对干净",
        "note": "官方页面 home.ustc.edu.cn/~zhouh156/dataset/csl-daily/",
        "tfnet_wer": {"dev": 25.1, "test": 23.5},
    },
    "SLR500": {
        "vocab": 500, "modality": "RGB + 深度 + 3D 骨架",
        "task": "孤立词", "env": "实验室",
        "note": "需全职教师签协议，学生不可申请",
    },
    "NMFs-CSL": {
        "vocab": 1067, "task": "孤立词", "modality": "RGB",
        "env": "实验室",
        "note": "ustc-slr.github.io/datasets/2020_nmfs_csl/，同样需协议",
    },
}

print("\n  各数据集的任务类型与模态（决定能不能共享同一个 head）")
print("  %-26s %-8s %-8s %-22s %s"
      % ("数据集", "词表", "任务", "模态", "环境"))
for nm, d in others.items():
    print("  %-26s %-8s %-8s %-22s %s"
          % (nm, d["vocab"], d.get("task", "连续句"), d["modality"],
             d.get("env", "-")))

print("""
  ⇒ 三层不可混用的证据：

  (a) **任务不同**：CE-CSL / CSL-Daily 是连续句（需要时序对齐）；
      SLR500 / NMFs-CSL 是孤立词（只需分类，无时序）。
      孤立词数据无法提供"这个手语出现在句子什么位置"的监督。

  (b) **模态不同**：SLR500 有 Kinect 深度 + 25 个 3D 关节；
      CE-CSL 只有 RGB。混用要额外做模态对齐或补齐。

  (c) **词表不同**：3515 vs 2000 vs 500 —— 三套 label space 没有交集保证，
      联合训练需要一张联合词表（会膨胀到 5000+），且每类样本被摊薄。
""")

# CE-CSL 词表 vs 假设的 CSL-Daily 2000 词表的重叠无法直接算（数据不在本地），
# 但可以给出「若要联合，需要多少类」的上界估计
print("  联合词表规模上界估计（若强行混用）：")
print("    3515 + 2000 = 5515 类（假定零重叠，实际更少）")
n_tr = sum(len(toks(g)) for g in lab_tr.values())
print("    CE-CSL train token %d -> 每类平均 %.1f 个样本"
      % (n_tr, n_tr / 3515))
print("    5515 类时每类 %.1f 个 -> 比现在再少 %.0f%%"
      % (n_tr / 5515, 100 * (1 - 3515 / 5515)))
print("    ⇒ 长尾词样本量进一步稀释，加重「低频词学不动」问题")

print("\n" + "=" * 72)
print("L2 协议层：官方 benchmark 怎么做（决定混用会不会毁掉可比性）")
print("=" * 72)
print("""
  arXiv:2409.11960v2（CE-CSL 官方论文）实测确认：
  **按数据集分别训练、分别评估**，没有混合训练。

  原文依据：
  - "we conduct experiments on three other large-scale publicly available
     datasets, including RWTH, RWTH-T, and CSL-Daily"
  - 表 VI 为每个数据集分别列 Dev/Test WER，无混合训练结果
  - 各数据集用各自的官方划分（CE-CNSL 4973/515/500；
     CSL-Daily 18401/1077/1176）
  - 各数据集用各自的输出词表（CE-CNSL 3515；CSL-Daily 2000）
  - 消融实验在 RWTH 上单独做，不在混合集上做

  TFNet 各数据集 WER：
    CE-CNSL    dev 42.1 / test 41.9   <- 与我们同数据集
    RWTH       dev 18.7 / test 18.6
    RWTH-T     dev 18.0 / test 19.1
    CSL-Daily  dev 25.1 / test 23.5

  ⇒ 如果你混用 CE-CSL + CSL-Daily 训练，
    **42.1% 这个官方基准就不可比了** —— 那是在纯 CE-CSL 上训的。
    FYP 里报的数字无法与官方表格对照，失去意义。
""")

print("=" * 72)
print("L3 统计层：CE-CSL 自身的背景/signer 构成（混入会破坏什么）")
print("=" * 72)
# signer 从视频路径 A-L 推断（标签文件无 signer 字段）
vid_root = Path("/mnt/c/Users/su127/Desktop/csl视频")
split_counts = {}
for split in ("train", "dev"):
    d = vid_root / split
    if not d.exists():
        print("  (视频目录不可读: %s)" % d)
        break
    subs = sorted(p.name for p in d.iterdir() if p.is_dir())
    split_counts[split] = subs
    print("  %s 的 signer 子目录（=背景组）: %s" % (split, subs))

print("""
  CE-CSL 的 A-L 是**按背景/signer 分组**的 12 组。
  混入会引入一个新的维度（背景域），但**没有对应的标签**告诉你
  「这条视频属于哪个背景域」—— 因为 CE-CSL 的标签文件里没有这个字段。

  ⇒ 想用混合做「域泛化」，你需要先给 CE-CSL 补一个 domain id；
    官方论文没有这么做，也没有报告任何域泛化实验。
""")

print("\n" + "=" * 72)
print("结论")
print("=" * 72)
print("""
  **不能混用，而且有四层独立理由（任意一条都足够）：**

  1. **协议层（最致命）**：官方 42.1% 是在纯 CE-CSL 上训的。
     混用后你的数字与官方表格不可比，FYP 失去参照。

  2. **任务层**：CSL-Daily/连续句 vs SLR500/孤立词，
     孤立词数据给不了时序对齐的监督。

  3. **模态层**：SLR500 有 Kinect 深度+3D 关节，CE-CSL 只有 RGB。

  4. **词表层**：3515 vs 2000 vs 500 联合会膨胀到 5000+ 类，
     每类样本再稀释，加重已有的「低频词学不动」问题。

  **如果你的目的是「借干净背景数据提升 CE-CSL 表现」，
  正确做法不是混用训练集，而是：**
    - 只用 CE-CSL 训练 + 只用 CE-CSL 评估（保持可比）
    - 干净背景数据仅用于**特征提取器的调试/对照**，
      不进入训练与评估
    - 想借数据增强，优先用 CE-CSL 自身（时序增强 ±20%、
      随机裁剪 256→224 + 翻转 0.5 —— 论文明确用的这两招，
      而我们 P35 都没测过）
""")

out = {
    "local_data": {"only": "CE-CSL",
                   "ustc_csl_daily_slr500_present": False},
    "layer1_label_space": {
        "datasets": others,
        "reason": "任务不同（连续句 vs 孤立词）、模态不同（RGB vs RGB+D+骨架）、"
                  "词表不同（3515/2000/500/1067）",
        "joint_vocab_upper_bound": 5515,
        "per_class_samples_now": round(n_tr / 3515, 1),
        "per_class_samples_joint": round(n_tr / 5515, 1),
    },
    "layer2_protocol": {
        "official_protocol": "per-dataset train + per-dataset eval，无混合训练",
        "tfnet_wer": {"CE-CNSL": {"dev": 42.1, "test": 41.9},
                      "CSL-Daily": {"dev": 25.1, "test": 23.5},
                      "RWTH": {"dev": 18.7, "test": 18.6},
                      "RWTH-T": {"dev": 18.0, "test": 19.1}},
        "fatal_consequence": "混用后 42.1% 官方基准不可比 -> FYP 失去参照",
        "paper": "arXiv:2409.11960v2",
    },
    "layer3_domain": {
        "ce_csl_signer_groups": "A-L，12 组（按背景/signer 分组）",
        "labels_have_domain_field": False,
        "note": "标签文件无 domain 字段，无法直接做域泛化实验",
    },
    "verdict": "不能混用；四层独立理由，协议层最致命",
    "correct_approach": [
        "训练与评估都只用 CE-CSL，保持与官方基准可比",
        "干净背景数据仅用于特征提取器调试/对照，不进训练与评估",
        "若要数据增强，优先用 CE-CSL 自身的时序增强 ±20% + 随机裁剪翻转"
        "（论文明确使用，我们 P35 未测过）",
    ],
}
p = REPO / "artifacts/metrics/blank-gov/p60-dataset-mixing.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)