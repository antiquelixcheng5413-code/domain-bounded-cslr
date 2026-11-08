# H1 / P224-P226 检索增强 kNN 后验精修（RAC 分布插值） — 阶段报告（FAIL）

> 生成：2026-11-08 17:58｜tag `p226-h1`｜夜间自动链生成｜依据等级沿用【有/间接/无】
> 口径：dev 515 句、语料级 `WER_official`；uWER 对 dev 774 类 gloss 等权；低频桶按 train 频次 ≤10 划分。

## 1. 动机与论文依据

【有】IEEE 2025 泰语指语多模态检索增强识别：train 派生知识库 + kNN 后验精修，signer-independent 显著改善、野外 WER 降 >30%，并明确要求 data-leakage-controlled；【有】CSLR² (Raude et al., ECCV 2024, arXiv:2405.10266)：句级检索与 CSLR 互益，BOBSL CSLR-TEST WER 65.5 vs 71.7，检索式分类对同义词/视觉相似手语更宽容；【有】RAC 通用框架：P(y|x)=λP_knn+(1−λ)P_cls 分布插值；【有】本地 E3 1-NN 27.44%（随机 44×）与 G0 train 混淆表（38.45% 覆盖 / lift 111×）；【无】gloss 级稠密检索修正无直接先例，属自建协议。

## 2. 方法（单变量，其余配方逐项沿用 P184）

零训练、参数无关。记忆库只由 train 构建：CTC greedy 路径切成连续相同非 blank 段，每段取 classifier11 输入帧特征均值 + L2，PCA128 后建 cosine 索引（K=32）。dev 段检索得 P_knn，对段内每帧做 log((1−λ)exp(lp)+λP_knn) 融合后 greedy 解码。主干/适配器/分类头全部不动，不花 6 轮训练预算。λ 选择规则与 R1–R8 判据见 `docs/planning/H1-PREREG-2026-11-08.md`。

## 3. 预注册判据（2026-11-08 16:50 写死于 `H1-PREREG-2026-11-08.md`；R1–R8、λ 网格与选择规则均在看到结果前固定）

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

| 指标 | 基线 p226-h1-base | 候选 p226-h1 | Δ（正=改善） |
|---|---|---|---|
| WER_official | 34.34 | 34.31 | +0.03 |
| uWER | 60.28 | 60.21 | +0.07 |
| 低频桶 WER | 70.52 | 70.38 | +0.14 |
| 高频桶 WER | 18.51 | 18.47 | +0.04 |
| micro sub+del | 31.18 | 31.1 | +0.08 |
| 插入数 | 90 | 91 | -1.00 |
| 非blank帧占比 | None | None | — |


配对 bootstrap（句子级，20000 次）：ΔWER 点估计 **+0.035pp**，95% CI **[-0.071, +0.144]**（CI 跨 0 = 无差别）。

低频桶 692 token / 高频桶 2150 token（合计 2842，与 P197 ref_tokens=2842 对账）。

## 5. 判定

| 规则 | 内容 | 结果 |
|---|---|---|
| R1_uWER_improve_ge_2pp | uWER 改善 ≥ 2.0pp | ❌ |
| R2_low_bucket_improve_ge_2pp | 低频桶 WER 改善 ≥ 2.0pp | ❌ |
| R3_high_bucket_worsen_le_1pp | 高频桶恶化 ≤ 1.0pp | ✅ |
| R4_WER_official_degrade_le_1pp | WER_official 退化 ≤ 1.0pp | ✅ |
| R5_ins_increase_le_10pct | 插入数相对基线增幅 ≤ 10%（E0 教训） | ✅ |
| R6_micro_worsen_le_0.30pp | micro sub+del 恶化 ≤ 0.30pp（D4 教训） | ✅ |
| R8_lowfreq_CI_gt_0 | 训练类：6 轮 best dev ≤ 34.80（零训练跳过） | ❌ |

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
数据来源：`artifacts/metrics/blank-gov/p226-h1-judge.json`、`p226-h1-uwer.json`。


## 附录：口径与审计说明（2026-11-08 18:00）

- **基线口径**：本报告用「同 p188 后验、同句序的 λ=0 派生基线」（`p226-h1-base-uwer.json`，
  WER 34.34），而非 `p184-base-uwer.json`（34.41）。两者差 0.07pp，
  源于 p188 落盘后验与评估器重新前向的数值/解码差异；用同后验基线可让配对 bootstrap
  真正逐句配对，消除这个系统偏差。
- **两次被拦下的无效判定**：17:37 首次判定因 dev dump 跳过 1 条超长视频（514 vs 515 句）
  导致配对失败；17:44 又发现 `per_item` 的句序索引 `ii` 被 `align_ops` 的插入数覆盖，
  515 句只写出 4 个 sid。修法：以 p188 的 515 句为准、缺段特征的句子不融合、
  句序索引改用独立变量 `ix`，并给 `night_judge.py` 的 `paired_bootstrap` 加
  「双方 sid 非空时必须逐句同序」断言。**只有 17:58 这次 FAIL 有效**。
- 判据与 λ 网格/选择规则见 `H1-PREREG-2026-11-08.md`（16:50 写死，早于任何结果）。
- 未跑 test；FAIL 后不调 λ、不换 K 重跑（预注册规则）。
