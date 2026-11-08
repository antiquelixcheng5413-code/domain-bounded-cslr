# G1 / P223 真实混淆对 hard negatives（CSGC 负类替换） — 阶段报告（FAIL）

> 生成：2026-11-08 12:19｜tag `p223-g1`｜夜间自动链生成｜依据等级沿用【有/间接/无】
> 口径：dev 515 句、语料级 `WER_official`；uWER 对 dev 774 类 gloss 等权；低频桶按 train 频次 ≤10 划分。

## 1. 动机与论文依据

【有】P222/G0 前提诊断：train 帧级 top-2 竞争对覆盖 dev 606 个真实混淆对的 **38.45%**（lift 110.95×，条件随机对照），远过预注册 20%/3× 线；【有】Zhang & Duh, EMNLP 2023：手形最小对立体识别率仅 ~20%（运动类 ~87%）；【有】Yin et al., arXiv:2606.28667 (2026)：建议最小对立体作 hard negatives；【有】本地 P219/P220：693 个严格替换对是语义近义混淆、非词表变体；【有】AAAI 2024 CSGC 跨句 gloss 原型对比框架。

## 2. 方法（单变量，其余配方逐项沿用 P184）

复用 E1 已验证的 GlossPrototypeBank + FeatureCapture（hook 只取 classifier11 第一次调用），**唯一单变量**：负类从「后验最高的非目标类」换成 P222 从 train 集前向挖出的真实混淆类（17054 对，train 来源、无 dev 泄漏），不足 n_neg 时才用原后验策略补齐。训练期-only、推理零开销。配方逐项沿用 P184（160 / 冻结 conv2d / lr 2e-5 / 6 轮 / seed 42 / p184-init）。

## 3. 预注册判据（2026-11-07 23:30 写死，实验启动前，禁止事后调整）

主看尾部（D4 挖出的真瓶颈），并显式防「拆东墙补西墙」与「插入爆炸」两种假改善：

| 编号 | 规则 |
|---|---|
| R1 | uWER 改善 ≥ 2.0pp |
| R2 | 低频桶 WER 改善 ≥ 2.0pp |
| R3 | 高频桶恶化 ≤ 1.0pp |
| R4 | WER_official 退化 ≤ 1.0pp |
| R5 | 插入数相对基线增幅 ≤ 10%（E0 教训） |
| R6 | micro sub+del 恶化 ≤ 0.30pp（D4 教训） |
| R7 | 训练类：6 轮 best dev ≤ 34.80（零训练跳过） |
| R8 | 低频桶单独配对 bootstrap 95% CI 下界 > 0（G1 额外要求，仅当评估器带 low_n 时启用） |

判定：全部满足才 PASS，任一不满足 FAIL ⇒ 不进 2 seed / 不跑 test。

## 4. 结果（dev）

| 指标 | 基线 p184-base | 候选 p223-g1 | Δ（正=改善） |
|---|---|---|---|
| WER_official | 34.41 | 35.86 | -1.45 |
| uWER | 60.41 | 61.91 | -1.50 |
| 低频桶 WER | 70.66 | 71.97 | -1.31 |
| 高频桶 WER | 18.56 | 19.21 | -0.65 |
| micro sub+del | 31.25 | 32.05 | -0.80 |
| 插入数 | 90 | 108 | -18.00 |
| 非blank帧占比 | 12.52 | 12.93 | — |


## 4. 训练曲线

| ep | train ctc | dev WER |
|---|---|---|
| 1 | 5.4414 | 35.82 |

best dev = **35.82**

配对 bootstrap（句子级，20000 次）：ΔWER 点估计 **-1.443pp**，95% CI **[-2.265, -0.613]**（CI 不含 0）。

低频桶 692 token / 高频桶 2150 token（合计 2842，与 P197 ref_tokens=2842 对账）。

## 5. 判定

| 规则 | 内容 | 结果 |
|---|---|---|
| R1_uWER_improve_ge_2pp | uWER 改善 ≥ 2.0pp | ❌ |
| R2_low_bucket_improve_ge_2pp | 低频桶 WER 改善 ≥ 2.0pp | ❌ |
| R3_high_bucket_worsen_le_1pp | 高频桶恶化 ≤ 1.0pp | ✅ |
| R4_WER_official_degrade_le_1pp | WER_official 退化 ≤ 1.0pp | ❌ |
| R5_ins_increase_le_10pct | 插入数相对基线增幅 ≤ 10%（E0 教训） | ❌ |
| R6_micro_worsen_le_0.30pp | micro sub+del 恶化 ≤ 0.30pp（D4 教训） | ❌ |
| R7_train_best_dev_le_34.80 | 训练类：6 轮 best dev ≤ 34.80（零训练跳过） | ❌ |
| R8_lowfreq_CI_gt_0 | 低频桶单独配对 bootstrap 95% CI 下界 > 0（G1 额外要求，仅当评估器带 low_n 时启用） | ❌ |

**verdict = FAIL**

## 6. 诚实边界

- 本报告所有 dev 数字为单次 6 轮（或单次推理）结果，dev 噪声地板 ±0.77pp；低频桶只有 692 token，
  严格来说需单独配对 bootstrap，本报告的总量 CI 不能替代低频桶 CI。
- 未跑 test（项目铁律：test 冻结、只在 dev 过线后一次性终评）。
- PASS 也只代表「值得投入 2 seed + 配对 bootstrap 复跑」，不等于性能已确认提升。
- 本链所有开关默认关闭，不改任何已有实验行为；补丁均备份原文件到 `/tmp/`。

## 7. 净结论

本方向判负，不进入复跑/test；结论并入整夜汇总，不再重复同类机制。

---
数据来源：`artifacts/metrics/blank-gov/p223-g1-judge.json`、`p223-g1-uwer.json`。
