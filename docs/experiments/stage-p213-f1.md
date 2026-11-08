# F1 / P213 训练期 logit adjustment（LA-CTC） — 阶段报告（FAIL）

> 生成：2026-11-08 04:51｜tag `p213-f1`｜夜间自动链生成｜依据等级沿用【有/间接/无】
> 口径：dev 515 句、语料级 `WER_official`；uWER 对 dev 774 类 gloss 等权；低频桶按 train 频次 ≤10 划分。

## 1. 动机与论文依据

【有】Menon et al., ICLR 2021《Long-tail Learning via Logit Adjustment》：训练时在 logits 上减 τ·log π_c 可得到Fisher-consistent 的平衡错误率；【有】Ren et al., NeurIPS 2020 Balanced Softmax（同类先验校正）；【有】本地 E0（P211）前提诊断 corr(log freq, 平均 logp)=0.8756 ⇒ TFNet 分类器确实编码了 train prior；【有】本地 E0 已证伪 post-hoc τ 调整（插入爆炸），明确留下「训练期补偿」这一未验证口子；【有】FGXM（PRL 2025）指出低频实词识别差源于 CTC 训练过程与数据不平衡。

## 2. 方法（单变量，其余配方逐项沿用 P184）

训练时对全部 CTC 头的 log-prob 施加 la(lp)=log_softmax(lp − τ·log π_c)，blank 类不动；推理时完全不调整（模型输出原样），故零推理开销。π_c 走训练实际路径 DPM.PreWords→word2idx 统计，与 D4 频次表同口径。与 D4 的 per-sample 加权、E0 的 post-hoc 解码均为不同机制：本项是 per-class 先验解耦，且模型有机会在 6 轮内自行补偿，避免 E0 的插入爆炸。配方逐项沿用 P184，单变量仅加 --la-tau。

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

判定：全部满足才 PASS，任一不满足 FAIL ⇒ 不进 2 seed / 不跑 test。

## 4. 结果（dev）

| 指标 | 基线 p184-base | 候选 p213-f1 | Δ（正=改善） |
|---|---|---|---|
| WER_official | 34.41 | 35.54 | -1.13 |
| uWER | 60.41 | 65.76 | -5.35 |
| 低频桶 WER | 70.66 | 76.59 | -5.93 |
| 高频桶 WER | 18.56 | 20.51 | -1.95 |
| micro sub+del | 31.25 | 34.17 | -2.92 |
| 插入数 | 90 | 39 | +51.00 |
| 非blank帧占比 | 12.52 | 11.37 | — |


## 4. 训练曲线

| ep | train ctc | dev WER |
|---|---|---|
| 1 | 5.7772 | 35.89 |
| 2 | 5.3343 | 35.57 |
| 3 | 5.2166 | 35.78 |
| 4 | 5.1129 | 35.47 |
| 5 | 5.0638 | 36.42 |
| 6 | 4.9301 | 36.24 |

best dev = **35.47**

配对 bootstrap（句子级，20000 次）：ΔWER 点估计 **-1.126pp**，95% CI **[-1.879, -0.384]**（CI 不含 0）。

低频桶 692 token / 高频桶 2150 token（合计 2842，与 P197 ref_tokens=2842 对账）。

## 5. 判定

| 规则 | 内容 | 结果 |
|---|---|---|
| R1_uWER_improve_ge_2pp | uWER 改善 ≥ 2.0pp | ❌ |
| R2_low_bucket_improve_ge_2pp | 低频桶 WER 改善 ≥ 2.0pp | ❌ |
| R3_high_bucket_worsen_le_1pp | 高频桶恶化 ≤ 1.0pp | ❌ |
| R4_WER_official_degrade_le_1pp | WER_official 退化 ≤ 1.0pp | ❌ |
| R5_ins_increase_le_10pct | 插入数相对基线增幅 ≤ 10%（E0 教训） | ✅ |
| R6_micro_worsen_le_0.30pp | micro sub+del 恶化 ≤ 0.30pp（D4 教训） | ❌ |
| R7_train_best_dev_le_34.80 | 训练类：6 轮 best dev ≤ 34.80（零训练跳过） | ❌ |

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
数据来源：`artifacts/metrics/blank-gov/p213-f1-judge.json`、`p213-f1-uwer.json`。
