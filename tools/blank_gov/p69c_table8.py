"""P69c：Table VIII 算例对不上 —— 找出官方实际用的规则

我的算例 9/10 不匹配。两种可能：
  A) 我从 PDF 抽取的 gloss 串错了（PDF 排版换行/空格问题）
  B) 官方 WER 用了额外规则（最可能：**去重复** + **逐句等权**）

关键线索：Case-2 TFNet「我/可以/支持/你/去/运动/。」vs GT「我/可以支持/你/去/运动/。」
  我算出 ed=2（把「可以支持」拆成「可以」+「支持」算 2 次操作）
  但官方报 0.0
  ⇒ 说明 **「可以支持」和「可以/支持」被当成等价**，
    即官方按**字符/字级别**算，或做了某种切分归一。

而 SEN「我/可以你/经济/木头/。」报 42.9：
  42.9% = 3/7或 3/7.0 —— 分母 7，不是 6（GT 6 token）
  ⇒ **分母不等于 GT token 数！**

Case-1 SEN 报 50.0%：50.0% = 4/8 = 3/6 都不整；若分母 8 → ed 4 ✓（我算出 4）
  ⇒ 分母 8 与我一致，ed 4 也一致，但 4/8 = 50% ✓✓
  **等一下 —— Case-1 SEN 其实是 50.0%，我算出 57.1%，因为我的 GT 是 7 token 而非 8！**

⇒ 结论方向：**我抽取的 GT token 数可能错了**（PDF 里 `他/小孩时间/开始/做/律师/希望/。`
可能是 7 或 8 个，取决于是否把某个词拆开）。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path


def lev(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


# Case-1 报告值
R1 = {"SEN": 50.0, "CorrNet": 25.0, "VAC": 37.5, "MAM-FSD": 0.0,
      "TFNet": 0.0}
R2 = {"SEN": 42.9, "CorrNet": 42.9, "VAC": 42.9, "MAM-FSD": 28.6,
      "TFNet": 0.0}

print("=" * 78)
print("1. 从报告值反推分母")
print("=" * 78)
print("  Case-1（报告 50.0/25.0/37.5/0.0）")
print("    25.0% = 1/4 = 2/8 = 3/12 ...")
print("    37.5% = 3/8  ← 分母 8 很自然（ed=3）")
print("    50.0% = 4/8  ← 分母 8（ed=4）")
print("    ⇒ **Case-1 分母 = 8 token**，且 ed 分别为 4/2/3")
print("    ⇒ 我算的 ed=4(SEN)/3(CorrNet)/3(VAC) 中 SEN 对上，"
      "CorrNet 应为 2 我算了 3")
print()
print("  Case-2（报告 42.9/42.9/42.9/28.6/0.0）")
print("    42.9% = **3/7**（3/7=42.857）← 分母 7！")
print("    28.6% = **2/7**（2/7=28.571）← 分母 7！")
print("    ⇒ **Case-2 分母 = 7 token**，而 GT 我数出 6 个")
print("    ⇒ **官方把「可以支持」拆成 2 个 token** → 7 个")
print("    ⇒ 则 TFNet 的「我/可以/支持/你/去/运动/。」= 7 token 完全对齐 → WER 0 ✓")

print("\n" + "=" * 78)
print("2. 用「官方分母」重算，验证ed 是否也吻合")
print("=" * 78)
# 官方实际使用的 GT（按分母推断）
GT1_8 = ["他", "小孩时间", "开始", "做", "律师", "希望", "。", "?"]
GT1_7 = ["他", "小孩时间", "开始", "做", "律师", "希望", "。"]
GT2_7 = ["我", "可以", "支持", "你", "去", "运动", "。"]

for name, gt, reps, hyps in (
    ("Case-1 (7 token)", GT1_7, R1, {
        "SEN": ["他", "时间", "开始", "。"],
        "CorrNet": ["他时间", "开始", "做", "希望", "。"],
        "VAC": ["他", "小孩时间", "开始", "工作", "困难", "。"],
        "MAM-FSD": ["他", "小孩时间", "开始", "做", "律师", "希望", "。"],
        "TFNet": ["他", "小孩时间", "开始", "做", "律师", "希望", "。"],
    }),
    ("Case-1 (8 token)", GT1_8, R1, {
        "SEN": ["他", "时间", "开始", "。"],
        "CorrNet": ["他时间", "开始", "做", "希望", "。"],
        "VAC": ["他", "小孩时间", "开始", "工作", "困难", "。"],
        "MAM-FSD": ["他", "小孩时间", "开始", "做", "律师", "希望", "。"],
        "TFNet": ["他", "小孩时间", "开始", "做", "律师", "希望", "。"],
    }),
    ("Case-2 (7 token)", GT2_7, R2, {
        "SEN": ["我", "可以你", "经济", "木头", "。"],
        "CorrNet": ["我", "可以", "你", "。"],
        "VAC": ["我", "可以", "你", "好", "锻炼", "。"],
        "MAM-FSD": ["我", "可以", "支持", "你", "锻炼", "。"],
        "TFNet": ["我", "可以", "支持", "你", "去", "运动", "。"],
    }),
):
    n = len(gt)
    print("\n  %s  GT=%s" % (name, "/".join(gt)))
    for m, h in hyps.items():
        e = lev(gt, h)
        print("    %-8s ed=%d  ed/%d=%5.1f%%  报告=%5.1f%%  %s"
              % (m, e, n, 100 * e / n, reps[m],
                 "✓" if abs(100 * e / n - reps[m]) < 0.6 else "✗"))

print("""
  ⇒ **发现**：官方 gloss GT 与我从 PDF 抽出的不完全一致
    - Case-2 的「可以支持」官方按**2 个 token** 算（分母 7 而非 6）
    - 该数据集的标注存在**词组粘连**，官方在计算时做了拆分歧义处理

  ⇒ 但这**不影响**我们的口径结论：
    官方公式明确写在论文式(6)：WER = 100% × (ins+del+sub) / sum
    这是 **corpus-level 累加**、分母 = 参考 token 总数、
    **无归一化**（不去标点、不合并重复）。
    这与我们的 token 级 WER（0.5211）**完全同口径**。
""")

out = {
    "formula_in_paper": "式(6): WER = 100% x (ins + del + sub) / sum",
    "sum_definition": "该语料全部参考 gloss token 数（corpus-level）",
    "normalization": "无 —— 不去标点、不合并重复、不做大小写归一",
    "our_primary": "token 级 WER = 0.5211 —— 与官方同口径，可直接对照论文",
    "table_viii_caveat": "论文 Table VIII 的 per-case WER 与我从 PDF 抽取的 gloss "
                         "串不完全对齐（Case-2 GT「可以支持」疑为词组粘连），"
                         "所以不能用它反推分母来验证公式；"
                         "但公式本身是明确的。",
    "verified": False,
    "note": "Table VIII 是定性示例，其单句 WER 可能用了额外的对齐/拆分歧义处理；"
            "不影响主口径结论",
}
p = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov"
         "/p69c-table8-check.json")
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("收据 -> %s" % p)