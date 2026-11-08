# N2-S3 / K1 seed 43 复跑 — 阶段报告（FAIL）

> 生成：2026-11-09 05:19｜tag `p238-k1-s43`｜夜间自动链生成｜依据等级沿用【有/间接/无】
> 口径：dev 515 句、语料级 `WER_official`；uWER 对 dev 774 类 gloss 等权；低频桶按 train 频次 ≤10 划分。

## 1. 动机与论文依据

（未登记）

## 2. 方法（单变量，其余配方逐项沿用 P184）

（未登记）

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

| 指标 | 基线 p184-base | 候选 p238-k1-s43 | Δ（正=改善） |
|---|---|---|---|
| WER_official | 34.41 | 34.52 | -0.11 |
| uWER | 60.41 | 60.26 | +0.15 |
| 低频桶 WER | 70.66 | 69.65 | +1.01 |
| 高频桶 WER | 18.56 | 20.6 | -2.04 |
| micro sub+del | 31.25 | 32.55 | -1.30 |
| 插入数 | 90 | 56 | +34.00 |
| 非blank帧占比 | 12.52 | 11.45 | — |


## 4. 训练曲线

| ep | train ctc | dev WER |
|---|---|---|
| 1 | 7.8671 | 35.36 |
| 2 | 7.1201 | 35.36 |
| 3 | 6.8911 | 34.73 |
| 4 | 6.8770 | 35.01 |
| 5 | 6.5206 | 34.52 |
| 6 | 6.2968 | 34.80 |

best dev = **34.52**

配对 bootstrap（句子级，20000 次）：ΔWER 点估计 **-0.106pp**，95% CI **[-1.092, +0.888]**（CI 跨 0 = 无差别）。

低频桶 692 token / 高频桶 2150 token（合计 2842，与 P197 ref_tokens=2842 对账）。

## 5. 判定

| 规则 | 内容 | 结果 |
|---|---|---|
| R1_uWER_improve_ge_2pp | uWER 改善 ≥ 2.0pp | ❌ |
| R2_low_bucket_improve_ge_2pp | 低频桶 WER 改善 ≥ 2.0pp | ❌ |
| R3_high_bucket_worsen_le_1pp | 高频桶恶化 ≤ 1.0pp | ❌ |
| R4_WER_official_degrade_le_1pp | WER_official 退化 ≤ 1.0pp | ✅ |
| R5_ins_increase_le_10pct | 插入数相对基线增幅 ≤ 10%（E0 教训） | ✅ |
| R6_micro_worsen_le_0.30pp | micro sub+del 恶化 ≤ 0.30pp（D4 教训） | ❌ |
| R7_train_best_dev_le_34.80 | 训练类：6 轮 best dev ≤ 34.80（零训练跳过） | ✅ |
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
数据来源：`artifacts/metrics/blank-gov/p238-k1-s43-judge.json`、`p238-k1-s43-uwer.json`。
