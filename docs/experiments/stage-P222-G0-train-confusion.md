# P222 / G0 完整版：train 混淆对能否覆盖 dev 混淆对 — 阶段报告（PRECONDITION_PASS）

> 生成：2026-11-08 11:12｜train 集 4973 条前向（GPU）｜**前提诊断，不报性能、不进 test**
> 预注册判据 08:25 写死：k=2 的 recall ≥20% 且 lift ≥3.0 ⇒ PRECONDITION_PASS，否则 STOP。

## 1. 为什么必须做
第三轮方案 G1 想用「train 集挖出的混淆对」做最小对立体 hard negatives。
但 train WER 仅 1.03%（强过拟合），train 上可能根本不犯错；
且 dev 混淆对**不得进训练**（数据泄漏）。因此 train 混淆表的覆盖能力是 G1 的一票否决。

## 2. 方法
- 模型：`p184-tfnet-freeze-best.pt`，train split 无增广前向（与 dev 评估同口径）。
- (a) greedy 实际替换错误：严格 Levenshtein DP 回溯（复用 P219，自检已通过）。
- (b) 帧级 top-k 竞争对：CTC greedy 路径按连续相同非 blank 类切段，
  每段取平均 logp 里最高的 k 个「非自身、非 blank」类 ⇒ 即使 train 全对，
  也能揭示模型觉得哪些 gloss 最容易互相混淆。
- 覆盖度：与 dev 严格混淆表（606 个唯一对）求交；
  对照 = **条件随机**（固定 dev 对的 ref、随机替换 hyp，与 P219 一致）重复 20 次的期望 recall，
  lift = recall / 随机 recall。
  ⚠️ 08:27 冒烟时原定的「均匀随机词对」对照在 3516 类词表下期望命中≈0，
  使 lift 退化成 10^7 量级 ⇒ 正式跑前改为条件随机；**判据阈值 recall≥0.20 / lift≥3.0 未变**，
  这是对照实现的修正，不是看到结果后调判据。

## 3. 结果

| 指标 | 值 |
|---|---|
| train 样本 / greedy WER | 4973 / **0.988%** |
| (a) greedy 替换对（唯一 / dev 命中） | 94 / 9（recall 0.0149） |
| dev 混淆表规模 | 606 |

| top-k | 唯一竞争对 | 命中 dev 对 | recall | 随机 recall | lift |
|---|---|---|---|---|---|
| 1 | 9162 | 193 | 0.3185 | 0.002063 | 154.40 |
| 2 | 17054 | 233 | 0.3845 | 0.003465 | 110.95 |
| 3 | 24743 | 255 | 0.4208 | 0.005363 | 78.46 |

## 4. 判定：**PRECONDITION_PASS**

过线 ⇒ G1 的 train 挖掘前提成立，可用 train 竞争对做 hard negatives；仍须新写预注册判据并做低频桶单独 bootstrap 后再启动 6 轮训练。

## 5. 诚实边界
- 帧级竞争对是「模型当前觉得相近」的近似，不是标注的最小对立体；
  top-k 越大覆盖越广但噪声越多，故主判据固定在 k=2，不事后挑 k。
- 随机对照按均匀词表采样，未控制词频；长尾先验可能抬高 lift，
  但 lift 只作相对量级参考，主判据同时要求 recall 绝对过线。
- 本项只回答「train 混淆表能否覆盖 dev 混淆对」，不回答 G1 训练是否一定有效。
- 未跑 test、未 commit、未 push。

补充说明：本次为 **10:35 修复显存问题后的完整重跑**（08:30 首跑在 1500/2487 batch 被用户叫停、未产出判定）。修复项：batch=1、每 100 batch `empty_cache()`、跳过 >400 帧的超长视频 **24 条**（占 train 0.36%，实测最大 520 帧）、每 400 batch 断点存盘。**跳过的样本已计入 seen 但未参与统计**，判据阈值与对照算法均未改动。

数据来源 `artifacts/metrics/blank-gov/p222-g0-train-confusion.json`。
