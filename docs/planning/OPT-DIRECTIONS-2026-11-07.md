# 剩余优化方向（2026-11-07 文献核查，仅方案，不启动实验）

> 输入：用户自跑 P184–P206 全系列记录（`stage-P184-tfnet-peft.md`、`DATA-SIDE-PLAN-2026-11-06.md`）
> + 本地论文库（参考论文/papers 17 篇、新参考论文 7 篇、`LIT_PLAN_论文逐篇对照.md`）
> + 2026-11-07 外部文献检索（ACL Anthology / arXiv / ScienceDirect / IEEE / NeurIPS）
> 依据等级（用户约定）：【有】能指出是哪篇哪句｜【间接】论文讲相近机制｜【无】无支撑，我拍的
> 状态：**仅方案**。任何方向启动前须按项目规范写死判据。

---

## 0. 一句话结论

P184–P206 已把「微调 / 蒸馏 / 换主干 / 通用增广 / 解码 n-gram / 集成」全部证伪，
但记录里留了**两个尚未开采的硬数字**：**P197 的 8-best oracle 25.44**（greedy 34.41，
差 8.97pp）与 **D0 的 33.3pp 泛化间隙**（train 1.03 / dev 34.34）。
剩余方向的价值，取决于它是否直接打这两个数字之一；其余都是噪声内抛光。

## 1. 三个诊断锚点（决定方向取舍）

| 锚点 | 数字 | 推论 |
|---|---|---|
| P191 误差构成 | 替换 **67.5%** / 删除 24.5% / 插入 8.1%；生成率 94.6%、exact 13.0% | 主矛盾是**同类手势细粒度区分**，不是长度/欠生成/解码 ⇒ 解释了 blank bias、多尺度 CTC、WE/WC、LM 重排为何全无效 |
| D0 泛化间隙 | train WER **1.03%** vs dev **34.34%** = **33.3pp** | 强过拟合；但 E1/E2 增广实测失败 ⇒ 不是「数据不够多样」，是「子模型输出不一致 / CTC 峰值塌缩」 |
| P197 解码上界 | greedy 34.41 → **8-best oracle 25.44** | **正确句子已在候选里**，缺的是打分器 ⇒ 唯一可能不重训就拿到 pp 级收益的口子 |

> ⚠️ **25.44 的准确含义（2026-11-07 核查 `p197_beam_lm.py` L233–246 + `official_wer.wer_official()`）**：
> 它是 **dev 集语料级 `WER_official` = 25.44%**（`723 / 2842`），与 greedy 的 `978 / 2842 = 34.41%`
> **同一分母、同一 `split_glued` 两侧处理**，故 −8.97pp 的差值真实可比。
> 但它是 **oracle WER（作弊上界）**：每句用**参考 gloss** 从 8 个候选里挑 Levenshtein 最小的
> （`best = min(levenshtein(rt, ...) for pre in nb)`），**用了标签选句**，实际不可达。
> 同收据里真实解码：beam 1-best 34.34 ≈ greedy，bigram LM λ=1.0 恶化到 37.12。
> ⇒ 25.44 只能读作「候选集合的信息上限」，**不是「有 8.97pp 可拿」**。
> 另：这是 dev、语料级；官方口径（P189）是 test、逐句均值，两者不可直接比较。

补充：D0-b 显示替换对**不集中**在单一手势族（标点互换≈5%、近义变体 gloss≈8%、真实语义混淆为主）
⇒ 定向增广 / focal 重加权须按稀有实词而非手形词族设计。

## 2. 候选方向（按性价比排序）

### D1. N-best 用预训练语言模型重排 —— 优先级 1，零训练

- **依据**：【有】AutoSign（arXiv:2507.19840, 2025）"bypassing traditional alignment mechanisms...
  directly learning the textual dependencies in the glosses"，用预训练 AraGPT2 解码器，
  Isharah-1000 上 WER 改善最高 6.1%；【有】本地 `ref15_Camgoz_Sign_Language_Transformers` 与
  官方 TFNet 评测均用 beam10 + 词表约束；【有】SpellRing（CHI 2025）实测
  "phrase-level WER improved with use of a language model"（0.099）。
  【间接】CVT-SLR（CVPR 2023）视觉-文本对齐、ACMC（2025）多粒度视觉-文本对比。
- **与已判死项的区别（关键）**：P197/P166 判死的是**从 4974 句统计出的 bigram/词频先验**，
  其失败主因是语料稀疏、未登录回退惩罚压过声学项；**预训练 LM 自带语言知识，不吃这份稀疏统计**。
  两者机制不同，不构成对 D1 的反证。
- **零成本先行诊断**：P197 已落盘 8-best 候选，直接复用，用本地已接入的 Qwen2.5-1.5B
  （commit fc9be7c）对 gloss 序列打分（长度归一 logP / PPL），在 **dev** 上测
  「重排后 WER 能把 25.44 的 oracle 吃掉多少」。纯推理、不动模型、不碰 test。
  ⚠️ 诚实风险：gloss 序列 ≠ 自然语言，中文预训练 LM 对其困惑度未必有判别力 ⇒ 先诊断再决定是否投入。
- **成本**：诊断 1–2 h（纯 CPU/GPU 推理）；若有效，仍是零训练、推理零改动。
- **判据（预注册）**：dev WER_official ≤ **33.6**（比 34.34 好 ≥0.7pp）才进入 2 seed + 配对 bootstrap；
  否则判负，不跑 test。同时报 oracle 利用率 = (34.41−重排WER)/(34.41−25.44)。

> **判据溯源（2026-11-07 14:30 核查补注，不改动判据数值）**：
> 33.6 来自 `DATA-SIDE-PLAN-2026-11-06.md` §3 E1 判据「dev ≤ 33.6（比 34.34 提升
> ≥0.74pp，刚好越过噪声地板）」，本项目沿用该线以保持与 E1/E2/P206 可比。
> 构成：基线 **34.34**（P184@160 dev WER_official，语料级）− **0.74pp** ≈ 噪声地板
> **±0.77pp**（`stage-P184-tfnet-peft.md` §5：dev 仅 515 句的最小可分辨效应）。
> 目的：单次实验的 best 会受 seed/CUDA 非确定性漂移 0.3–1.7pp（P83/P85/P89 曾三次骗过），
> 故要求提升接近噪声地板才值得投 2 seed 复跑 + 配对 bootstrap + 唯一一次 test。
> ⚠️ **两处口径瑕疵（诚实记录）**：(1) 上方括号写「≥0.7pp」，但 34.34−0.70=**33.64**，
> 而 33.6 实为 **0.74pp**；本文把硬约束 §4.1 的 0.7pp 与方案 §3 的 33.6 混写，
> 差 0.04pp（33.6 更严，不影响判负）。(2) 该线原为 **6 轮训练筛选**设计，
> D1 是**零训练纯推理**、无 seed 漂移，套用偏保守；更贴切的主判据应是配对 bootstrap CI。
> 实际 D1 同时要求「dev≤33.60 **且** CI 下界>0」，两道门；best 33.88 提升仅 0.46pp
> 且 CI [−0.035, 1.091] 跨 0，**换任何合理判据线结论都是判负**（见 §5 敏感性核算）。
- **预期量级**：【无】我拍的——大概率 33–34（噪声内）；若能到 30 以下即为本项目首个大幅突破。

### D2. E3 多视图一致性正则 / CR-CTC —— 优先级 2，计划内唯一未跑项

- **依据**：【有】R-Drop（NeurIPS 2021, arXiv:2106.14448）原文"especially when applied to
  fine-tuning large-scale pre-trained models... can yield substantial improvements"，
  18 数据集普遍有效；【有】CR-CTC（Yao et al.，语音）"enforcing agreement pushes the model
  toward the average of the two branches' predictions... **suppressing the extremely peaky
  CTC distributions, which reduces overfitting and improves generalization**"，
  在 LibriSpeech / Aishell-1 / GigaSpeech 把 plain CTC 提到接近 transducer。
  【间接】本地 `ref12_Wu_CrossModal_Consistency_SLR`（CCL-SLR 跨模态一致性）。
- **与已判死项的区别（关键）**：E1(P202)/E2(P203) 是**加大增广强度**⇒ 收敛变慢、逐轮落后；
  D2 是**用两个视图做一致性约束**（增广只是产生视图，不改变标签分布），是正则项不是数据增强。
  故不受「数据增广这条线关闭」结论约束。`DATA-SIDE-PLAN` §3 的 E3 列了优先级 3 但从未执行。
- **对症**：同时打 D0 的 33.3pp 过拟合 + CTC 峰值对齐（本地 blank 率 87.4%，
  本地 `06_SMART` 与 CR-CTC 都指出峰值对齐是 CTC 固有缺陷）。
- **实现**：两次前向（不同 dropout / 轻增广视图），逐帧 CTC logits 对称 KL，λ≈0.3；
  推理零开销。⚠️ **KL 必须按非 blank 帧加权**，否则被 ~87% 的 blank 主导（计划文档已提示）。
  沿用 P184 配方（160 / 冻结 conv2d / 6 轮 / lr 2e-5 / seed 42）单变量对照。
- **成本**：6 轮 ≈ 4 h（两次前向）。
- **判据（预注册）**：dev ≤ **33.8** 且 train–dev 间隙缩小；否则判负。
- **预期量级**：【无】我拍的——0 ~ −1pp；R-Drop 在微调预训练模型上常见 0.5–2pp，
  但本项目已有强增广+冻结主干，收益可能被压缩到噪声内。

### D3. 密集帧级监督 / Sign Spotting 辅助 —— 优先级 3，工程量大

- **依据**：【有】本地 `06_SMART`（arXiv:2608.25493）"CTC often produces peaky alignments...
  limited supervision for modeling fine-grained temporal dynamics"，CSFormer 用 sign spotting
  的密集帧级监督补 CTC，PHOENIX14-T test WER **19.50**（优于 AdaptSign 19.80）；
  【有】EMNLP 2026《Beyond CTC Spikes》(zhu-etal-2026-beyond-ctc, ACL Anthology 2026.emnlp-main.578)
  "dense temporal supervision substantially alleviates alignment collapse... **converges
  significantly faster**"，GTM 用 LLM 生成 gloss-to-frame 区间监督；
  【间接】EnCTC / EsCTC（最大条件熵 / 等间距约束抑制 peaky，场景文字识别）；
  【间接】本地 `07_TS²-TFD` 提供**无参数、CPU 可跑**的 TFD 边界检测，
  可替代 LLM 直接生成伪帧级区间标签 ⇒ 大幅降低 GTM 落地成本。
- **与已判死项的区别**：P185 多尺度 CTC、P196 WE/WC 都是**序列级 / 逐帧 max 的弱辅助**；
  D3 是**持续 gloss-to-frame 区间监督**，梯度信号密度不同。
- **风险**：① 与 P191 主矛盾（替换 67.5%）只是间接相关，改善对齐未必改善细粒度区分；
  ② 需生成边界标签 + 改损失，约 1 天工程；③ 6 轮预算内未必收敛。
- **成本**：高（P3）。**先做零成本探针**：用 TFD 在 train 上出边界，
  统计「伪边界与 CTC spike 的偏离度」，若两者高度重合则 D3 信息量≈0，直接放弃。
- **判据（预注册）**：dev ≤ 33.6 且 CTC 非 blank 帧占比上升；否则判负。

### D4. 低频实词 / 长尾重加权（focal CTC 或视觉-语言一致性）—— 优先级 4

- **依据**：【有】FGXM（Ke/Liu/Feng, Pattern Recognition Letters 191, 2025）
  "current CSLR methods perform well with high-frequency glosses... **struggle to accurately
  recognize low-frequency content words**... arising from limitations in existing datasets
  and the CTC training procedure"，并提出 uWER 无偏指标；
  【有】PGFLoss（IEEE MLSP 2025）pose-guided focal，PHOENIX14 24.1→**21.1**。
- **本地吻合证据**：P57 显示 train 频次 ≤10 的 dev token 占 **26.4%**；
  P191 高频漏词 `喜欢`(10)/`最`(7)/`介绍`(7)/`而且`(7)；D0-b 近义变体 gloss 混淆≈8%。
- **与已判死项的区别**：label smoothing（P142）是全局软化（已判死）；
  focal / class-balanced 是对难例与稀有类重加权，机制不同，**focal CTC 从未测过**。
- **判据（预注册）**：不能只看总 WER —— 须同时报 **uWER 与低频桶 WER**；
  dev ≤ 33.6 且低频桶明显改善才算正向，否则判负（避免又被噪声骗）。
- **成本**：中（改损失 + 6 轮）。

### D5. 低优先 / 不建议（有依据但本地反证据或算力不匹配）

| 方向 | 依据 | 不推荐理由 |
|---|---|---|
| CLIP/SigLIP 冻结骨干 + adapter（AdaptSign / SMART / ACMC / MMCST） | 【间接】多文 2025–2026 报告 SOTA | 本地 CLIP 线性探针 Cohen d **0.1006** 已判死（`LIT_PLAN` §A ref12）；8G + 6 轮预算不匹配 |
| 生成式 / diffusion CSLR（Pattern Recognition 2026）、decoder-only 全替换 | 【间接】跨数据集 SOTA | 需从零训练、3516 词表、4973 条，成本远超预算；P200 已证随机初始化大时序模块在 6 轮内拟合失败 |
| 224 / 更大容量、Conformer、全参微调、蒸馏 | 【有】本项目实测 | P186/P193/P199/P200/P171/P172/P174/P175/P195 全部判死 |
| 通用增广 E1/E2、landmark 融合 E4 | 【有】本项目实测 | P202/P203/P206 判死；P206 若续跑须先补 **test landmark**（当前无） |
| E5 易混词定向增广 | 【无】条件触发 | D0-b 已证替换对不集中，触发条件不成立 |

## 3. 一个不要漏掉的「口径红利」（零成本）

P189 已确认官方口径 = **逐句均值 + beam10**：官方 @224 test 从语料级 33.09 → 逐句均值
**32.74** → beam10 **32.63**（README 32.46，残差 0.17pp）。
但 **P184@160 的 test 33.98 报的是语料级**，从未在同一官方逐句均值口径下复评。
⇒ 建议先做一次纯推理复评（不训练、不碰 test 调参），把双轨数字统一到同一口径；
这不会改变模型，但会改变「对外数字」与后续所有判据线的位置。
⚠️ 注意：这只影响报数，不构成性能提升，不得当作实验收益。

## 4. 推荐执行顺序（全部预注册判据后再启动）

1. **D0'** P184@160 官方逐句口径复评（0.5 h，纯推理）→ 统一判据基线
2. **D1** N-best oracle 诊断：Qwen2.5-1.5B 打分重排（1–2 h，纯推理，零训练）
   dev ≤ 33.6 → 2 seed + 配对 bootstrap → 才考虑 test；否则判负（保留 oracle 利用率作诊断结论）
3. **D2** E3 / CR-CTC 一致性正则（6 轮 ≈ 4 h，唯一需训练的高性价比项）
4. **D3** TFD 伪边界探针（CPU，零训练）—— 有信息量才投入 GTM / Sign Spotting
5. **D4** focal CTC + uWER（改损失 + 6 轮）

任一步落在 dev 噪声地板 ±0.77pp 内即按项目规范判「无差别」，不进入 test。

## 5. 来源清单

- 本地：`docs/experiments/stage-P184-tfnet-peft.md`（§8–§15）、`docs/planning/DATA-SIDE-PLAN-2026-11-06.md`、
  `docs/planning/LIT_PLAN_论文逐篇对照.md`、`docs/planning/LIT_PLAN_论文方法实施计划.md`（§6）、
  `新参考论文/说明.txt`、`artifacts/metrics/blank-gov/p197-beam-lm-dev.json`
- 外部（2026-11-07 检索）：R-Drop arXiv:2106.14448 (NeurIPS 2021)；
  CR-CTC / EnCTC / EsCTC 见 josiehong.github.io CTC 综述 (2026-08)；
  AutoSign arXiv:2507.19840；CVT-SLR CVPR 2023；
  FGXM Pattern Recognition Letters 191 (2025) DOI 10.1016/j.patrec.2025.02.017；
  PGFLoss IEEE MLSP 2025 DOI 10.1109/MLSP62443.2025.11204296；
  SMART arXiv:2608.25493；Beyond CTC Spikes EMNLP 2026 (2026.emnlp-main.578)；
  ACMC Pattern Recognition 2025 (S0262885625002100)；MMCST IEEE SPL 2026；
  SpellRing CHI 2025；Signer-Invariant Conformer arXiv:2508.09372。
