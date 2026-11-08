# N3 收官诊断预注册（2026-11-09 05:22 写死）

> 背景：夜间接力 #2（`N2-PREREG-2026-11-08.md`）已在 05:19 完整跑完，
> K1 seed42 / seed43 均 FAIL、L1 FAIL、L2 OK。距 09:00 硬停剩 3h40m，
> **不足以再投一个 6 轮训练**（实测 3h46m + 评估收尾），故只做纯 CPU 收官诊断。
> 本文写于跑之前，禁止事后调整。

## 1. 性质（最重要）

**探索性诊断，不判 PASS/FAIL、不产生新的性能主张、不改任何已有判据**。
只把已有收据/产物的统计关系算清楚，用于回答两个描述性问题：

- Q1：K1 的 FAIL 在 2 个 seed 上是否**稳健**（同号、合并配对 CI、逐句一致性）？
- Q2：统计 bigram 生成的 1500 条合成句**多样性如何**（是否为 K1 的已知局限提供量化线索）？

## 2. 输入（只读，不重新前向、不跑 test、不改任何收据）

- `p184-base-uwer.json`、`p233-k1-uwer.json`（seed42）、`p238-k1-s43-uwer.json`（seed43）
  —— 三者 `per_item` 均已按 base 的 sid 顺序对齐（01:24 手动 / 05:19 链自动）。
- `p230_k1_clips.json`、`p231_k1_sentences.json`、`p232_k1_manifest.json`
- `p235-n2-k1-audit.json`（L2 已算的分布/覆盖数字，只引用不重算）

## 3. 统计项（写死）

1. **两 seed 合并配对 bootstrap**：bootstrap 20000 次、seed 42、句子级配对、
   逐句同序断言；分别报 seed42、seed43、「两 seed 逐句 edits 取平均」三种 ΔWER 点估计与 95% CI。
2. **逐句一致性**：按 `sign(edits_seed − edits_base)` 分四桶
   （两 seed 同时改善 / 同时恶化 / 分歧 / 至少一方持平），报句数与占比。
3. **分桶与编辑类型**：三者 WER_official / uWER / 低频桶 / 高频桶 / ins-sub-del / micro，
   只做对照表，不重新判定 R1–R8。
4. **合成语料多样性**：生成句的 token 总数、不同 gloss 数、
   top-1 / top-10 gloss 占 token 比例、最大单 gloss 复用次数、
   不同句子数、不同 gloss bigram 数、与 train 原句的精确重复数（应为 0）、低频覆盖率（引用 L2）。

## 4. 输出与边界

- 产物：`p239-n2-closure.json`、`stage-P239-N2-closure.md`（WSL + 桌面镜像），
  追加到总汇总 `OVERNIGHT-2-RUN-2026-11-09.md` 的一节，commit（**不 push**）。
- **诚实边界**：Q2 的数字只能说明「这批合成句长什么样」，**不能**据此断言
  「换成 LLM 生成就会有效」或「K1 判负完全归因于生成质量」——
  K1-PREREG 已写明 CTC 条件独立假设这一更大的机制风险。
- 严禁：push、test、删除/修改任何已有收据与判据、启动新训练、事后调阈值。
- 结论表述必须以已完成的 R1–R8 verdict（两 seed 均 FAIL）为准，不重新判定。
