# P233 / K1 语料合成 — 阶段报告

> 日期：2026-11-09 01:24｜判定：**FAIL**｜预注册：`docs/planning/K1-PREREG-2026-11-08.md`（20:45 写死）
> 上位方案：`OPT-DIRECTIONS-6-2026-11-08.md`（J1 片段库前提已过线）

## 1. 做了什么
本项目**第一次改训练数据的组织方式**：train 全量 CTC Viterbi 强制对齐切出
per-gloss 片段库 -> 统计 bigram + 低频优先生成 1500 条新 gloss 句 ->
随机取片段硬链接拼接成新 RGB 训练样本 -> 按 manifest 注入 `labels_tr`，
其余逐项沿用 P184 配方 6 轮 PEFT。**模型/损失/解码/评估管线零改动**。

## 2. 判据核查（统一 R1–R8）
| R1_uWER_improve_ge_2pp | ✗ |
| R2_low_bucket_improve_ge_2pp | ✗ |
| R3_high_bucket_worsen_le_1pp | ✗ |
| R4_WER_official_degrade_le_1pp | ✓ |
| R5_ins_increase_le_10pct | ✓ |
| R6_micro_worsen_le_0.30pp | ✗ |
| R7_train_best_dev_le_34.80 | ✓ |
| R8_lowfreq_CI_gt_0 | ✗ |

## 3. 结论
**FAIL** ⇒ 按预注册不调比例、不换生成器、不重跑、不进 test；累计 15 连败下接受天花板收尾

## 4. 诚实边界
- 生成器是**本地统计 bigram**，非文献的 LLM（启动前写死的工程妥协，证据等级【无】）；
  判负时无法区分「语料合成无效」与「统计生成质量不足」。
- 文献收益全部在 SLT（BLEU）测得；CTC gloss 识别的条件独立假设可能使
  「新上下文」收益打折，这是本方向最大的机制风险。
- 用 P184 切分 + 继续 P184 PEFT，存在文献承认的边界偏差自我强化风险。
- 合成样本只来自 train、只进 train；dev/test 目录零改动；合成句与 dev/test
  无 gloss 级精确重复；词表 `wordSetNum` 未变（strict load 校验）。
- 未跑 test；判据/数量/种子/比例/混合方式均未事后修改。

## 5. 依据
【有】arXiv:2606.11925、ACM MM20 arXiv:2010.05264、arXiv:2401.05336、
arXiv:2506.09643、本地 J1；【无】统计 bigram 替代 LLM；
【有】反证据：仅重组原句 −1.34 BLEU-4、合成数据不可用于预训练、L2 边界平滑负收益。

## 6. 产物
`p230_k1_clips.json` / `p231_k1_sentences.json` / `p232_k1_manifest.json` /
`p233-k1-judge.json`；日志 `artifacts/logs_P233_k1_train.log`。

## 7. 补跑与审计说明（2026-11-09 01:24）
- **训练完整**：20:48→01:03 跑满 6 轮，best **34.45%@ep6**，`p233-k1-best.pt` 675M 已落盘。
- **01:05 首次评估失败**：`CUDA driver error: device not ready / Failed to create GPU mapping`
  （WSL 驱动在 4h 训练后未恢复），K1 链因此 ABORT、未产出 judge。接力链 run1 于 01:06
  因 judge 缺失一度标 `K1_INVALID` 并跳过 seed43，该中间状态已由本次补跑纠正。
- **01:19 等待 150s 后**真实 TFNet 单样本前向探针通过；**01:22 用 `--batch 1` 补跑评估成功**
  （n=515、WER 34.45、uWER 60.53、micro 32.13），**未重训、未改任何权重**。
- **01:24 配对审计**：`p184-base-uwer.json` 的历史句序（`dev-00002, dev-00001, ...`）与
  `night_eval_full.py` 的自然句序仅前两句交换，被 `night_judge.paired_bootstrap` 的
  sid 同序断言拦下。用 `n2_align_peritem.py` 按 sid 重排 K1 的 `per_item`，
  逐项求和自检通过（edits 979 / ref_n 2842 / ins 66 / sub 670 / del 243 /
  low_n 692 / low_err 484 全部不变）⇒ **只改配对顺序，不改指标、不改判据**。
- 最终 verdict 以本报告 §2 的 R1–R8 为准；汇总报告已重生成。
