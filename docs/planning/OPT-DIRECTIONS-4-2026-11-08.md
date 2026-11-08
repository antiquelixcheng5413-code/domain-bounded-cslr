# 第四轮优化方向（2026-11-08 15:55，仅方案，未启动实验）

> 输入：第三轮收官结果（`ROUND3-SUMMARY-2026-11-08.md`）+ 2026-11-08 系统性文献检索
> + 本地论文库（参考论文 18 篇、新参考论文 7 篇）。
> 依据等级沿用用户约定：【有】能指出是哪篇哪句｜【间接】相近机制｜【无】无支撑。
> 状态：**仅方案**。未跑 test、未 commit（本轮方案文档尚未提交）、未 push。

## 0. 一句话结论

「冻结主干 + 6 轮 + 4973 条」内**加轻量训练项 / 解码后处理 / 先验校正 / 原型对比**
已被 D1–D4、E0/E1、F0/F1/F2、G1 系统证伪。第四轮必须换**信息源或推理范式**，
而不是继续往 loss 里加东西。文献核查给出一个很关键的事实：

> **「冻结视觉主干 + 有限训练预算 + CSLR（gloss 识别）」这个组合，
> 在测试时自适应、检索增强、密集监督三条线上都基本是空白**
> （TTA 已有工作全部落在 SLT 翻译、检索增强只有指语/句级、密集监督全部在全量微调设定下）。

这既是机会也意味着没有现成基线可抄，必须自己定义评价协议。下面三条按**性价比**排序，
全部附预注册判据；其中 H1 是唯一「零训练、参数无关」的路线。

## 0.6 H1 执行结果（2026-11-08 18:00 更新）

- **H1 / P224–P226 检索增强 kNN 后验精修 = FAIL（有效判定）**。
  记忆库 28010 条 train 段实例（PCA128 / K=32 / cosine），dev 2735 段查询，零训练、主干全冻结。
  λ 网格 0/0.05/0.1/0.2/0.3/0.5，预注册选择规则取 **λ=0.05**（护栏内 uWER 最低）。
- 同后验基线 34.34 → H1 **34.31**（Δ +0.03pp，配对 CI [−0.071, 0.144] 跨 0）、
  uWER 60.28→60.21（+0.07）、低频桶 70.52→70.38（+0.14，单独 CI [0.000, 0.447]）、
  高频桶 18.51→18.47、micro 31.18→31.10、插入 90→91。
  R1✗ R2✗ R3✓ R4✓ R5✓ R6✓ R8✗ ⇒ **FAIL**。
- **λ>0.1 单调恶化**（λ=0.5 时 WER 35.36 / uWER 62.10）⇒ 外部记忆提供的信号
  与 CTC 后验高度重合，插值只会稀释，没有带来新信息。
- ⇒ H1 路线判死；按 §3 顺序下一步是 **H2-a TTA 前提诊断**（dev 加 corruption 测 WER 跌幅，
  跌幅 <2pp 即 H2 STOP），成本约 30 min 纯推理、零训练。
  按累计 13+ 判负与本文 §4 的诚实预期，**接受天花板收尾**仍是合理选项。
- 详见 `stage-p226-h1.md`（含两次无效判定的审计说明）。

## 1. 现状锚点（决定取舍）

| 锚点 | 数字 |
|---|---|
| 累计判负 | D1–D4 / E0 / E1 / F0 / F1 / F2 / E3 / P218 / G1（12+） |
| 主矛盾 | P191 替换 67.5%；P219/P220 693 个严格替换对是**语义近义混淆**（cos 0.1974 vs 条件随机 −0.001，变体对=0），低频 ref 占 56.28% |
| 长尾 | 3515 类中 48% 只有 1 个 train 样本；P221 ref 频次中位 8 → hyp 31（低频实词被换成高频近义词） |
| 弱监督 | D3 每词仅 **1.06 帧**被监督；非 blank 帧占比 12.5%；blank 率 87.4% |
| 唯一正向 | P184 冻结主干 PEFT（dev 36.81→34.34 / test 35.66→**33.98**） |
| 天花板 | 官方 @224 零样本 test **32.74**（逐句均值/beam10）、P184@160 test 33.98、P150 53.73 |
| 硬预算 | 5060 8G｜160 输入｜6 轮 ×3h｜test 冻结只跑一次｜dev 噪声地板 ±0.77pp |

## 2. 候选方向（按性价比）

### H1. 检索增强 / kNN 后验精修（Retrieval-Augmented Classifier）—— 优先级 1，零训练

- **依据**
  - 【有】IEEE 2025《Multimodal Retrieval-Augmented Recognition for Thai Sign Language Fingerspelling》：
    构建「train 派生原型 + 文本 gloss 描述 + 音系属性」的多模态知识库，推理时检索 top-k 原型，
    用 **kNN 后验精修**（不是生成文本）；明确强调 **data-leakage-controlled knowledge base**，
    signer-independent 下显著改善、野外条件 WER 降超 30%。
  - 【有】CSLR²（Raude et al., ECCV 2024, arXiv:2405.10266）：手语—口语联合嵌入空间，
    句级检索与 CSLR 互益，BOBSL CSLR-TEST WER **65.5 vs 基线 71.7**；
    并演示**检索式分类**——不在 one-hot 上查表，而在文本嵌入空间找最近邻，
    「对同义词和视觉相似手语更宽容」。
  - 【有】RAC（Retrieval-Augmented Classifier）通用框架：
    标准融合式 `P(y|x) = λ·P_knn(y|x) + (1−λ)·P_cls(y|x)`（分布插值），
    记忆库可为实例记忆 / 标签模式记忆 / 混合，索引用 FAISS/HNSW。
  - 【有】本地 E3：landmark 样本级描述子留一 1-NN 预测主 gloss **27.44%**（随机 0.617%，44×）；
    但 **test 集没有 landmark**（P206 已记）⇒ 必须改用 RGB 帧特征，不能照搬 E3 的模态。
  - 【有】本地 G0：train 帧级竞争结构与 dev 真实混淆重合（38.45% 覆盖 / lift 111×），
    说明 train 特征库携带可判别信息。
  - 【无】**gloss 级稠密检索修正没有直接先例**；检索增强用于「连续」手语识别（而非孤立指语）
    的证据极薄。这是要自己定义协议的地方，不得当作已有结论引用。
- **与已判死项的区别（关键）**：D1 是 LM 重排、F0/P218 是 logit 校正、E1/G1 是训练期原型对比；
  H1 是**参数无关的外部记忆**，不动主干、不动适配器、不花 6 轮训练预算，
  推理期才查库 ⇒ 天然适配「冻结主干 + 小预算」。
- **本地资源与成本**（已估算）
  - train 帧级特征库：4973 句 × ~60 帧 × 1024 维 = float32 **1.14GB** / fp16 0.57GB /
    **PCA128+fp16 仅 ~70MB**；类原型库仅 13.7MB。8G 显存与 WSL 内存都放得下。
  - `p188_logp_p184.pt`（dev 后验 163MB）、`p222_confusion_map.json`（train 混淆表）、
    `p214_vta_target.npy`（gloss 文本嵌入）已落盘。
  - **第一步**：改造已验证的 G0 前向脚本，额外导出 train 帧级特征 → PCA128 → 记忆库
    （**GPU 前向 20–30 min，不是训练**）；再做 dev 上 λ 扫描（CPU，~10 min）。
- **⚠️ 硬约束**：记忆库只能来自 **train**，dev/test 样本不得回流；检索标签用 `DPM.PreWords→word2idx`
  与训练 target 同口径。
- **判据（预注册）**：统一 R1–R8（uWER/低频桶主看 + 插入/micro/高频桶护栏 +
  低频桶单独配对 bootstrap），另报记忆库大小、单次 kNN 查询延迟、λ 敏感性；
  **λ 选择规则必须在看到结果前写死**（护栏内取 uWER 最低，否则判负），防 p-hacking。
- **风险**：① 检索粒度——句级检索解决不了 gloss 级稠密对齐，需在帧/段后验上修正；
  ② 48% 单样本类记忆条目极稀疏，kNN 可能偏向高频类（与 P221 的偏置同向）；
  ③ 【无】增益幅度纯属推测，检索结果可能只是把错误从一个近义词搬到另一个。

### H2. 测试时自适应（参数自适应版 TTA）—— 优先级 2，需先做前提诊断

- **依据**
  - 【有】SAME（Yang et al., ACL 2026）：自称**首个 SLT 中的 TTA**，signer-aware MoE 门控，
    动机明确写「现有无监督域适应需要 batch statistics 或长时间适应，实时部署做不到」。
  - 【有】TRANSLATE（Wang & Wang, IEEE TMM 2026, DOI 10.1109/TMM.2026.3730967）：
    SLT 中**腐败诱导分布偏移**的 TTA；参数无关的时序方差解耦（动态手势特征 vs 时序稳定非手势特征）
    + 时序双正则 + 运动感知增强下的文本一致性正则。参数无关这点非常适合小预算。
  - 【有】TENT（ICLR 2021）熵最小化通用范式；【间接】CTC 边际概率/置信度作无监督目标
    （比直接压 entropy 更不容易退化成极短输出）。
  - 【有】系统性检索结论：**CSLR（gloss 识别）层面的 TTA 未见，冻结主干下未见**。
  - 【有】本地 P187 的 TTA 已判死 —— 但那是 **hflip / 多尺度 / 分辨率等输入变换**，
    与「参数自适应」机制不同，不构成对 H2 的反证。
- **⚠️ 本地最大障碍（必须先证伪）**：CE-CSL 的 dev 与 test 是**同分布**（同一采集协议），
  没有 signer shift / corruption ⇒ TTA 的收益前提可能根本不成立。
  ⇒ **H2-a 零成本前提诊断**：在 dev 上人为加 corruption（高斯模糊 / JPEG 压缩 / 亮度 / 遮挡），
  测 WER 掉多少；**若掉 < 2pp，H2 直接 STOP**，不投实验。
- **风险**：序列输出上的持续熵最小化极易坍塌（输出变短 / 词汇多样性下降）；
  必须监控「空输出句子占比」「不同 gloss 数」两个 collapse 指标，并在判据里设为护栏。
- **成本**：前提诊断 ~30 min 纯推理；若过线，参数自适应实验 1–2h（每样本少量前向+反传）。
- **判据**：统一 R1–R8 + 「空输出率不上升、输出 gloss 多样性不下降」；
  只更新 BN/LN affine 或 PEFT 模块，**conv2d 主干保持冻结并每样本 `.eval()`**。

### H3. 密集帧级监督 / Gloss-Temporal Module —— 优先级 3，成本最高

- **依据**
  - 【有】EMNLP 2026《Beyond CTC Spikes: LLM-Driven Gloss-Temporal Supervision for CSLR》
    （Zhu, Jin, Fang, 2026.emnlp-main.578）：GTM 用 LLM 融合 RGB 多路信息构造
    **持续 gloss-to-frame 区间监督**，帧级+序列级联合优化 + 课程学习；
    原文结论「dense temporal supervision substantially alleviates alignment collapse …
    **converges significantly faster**」。收敛更快对「有限预算」是直接利好。
  - 【有】DPLR（Jang et al., ICASSP 2023）：与 GT gloss 序列比对，长度一致就替换错误 gloss
    再填满 blank，差 1 只 densify、差 ≥2 丢弃样本；VAC+DPLR 在 PHOENIX14 test **21.6% WER**。
  - 【有】本地 `06_SMART`（CSFormer, arXiv:2608.25493）：直指 CTC blank 率 97%、每词仅少数帧预测，
    主张 Sign Spotting 密集帧级监督；本地 D3 已实测**每词仅 1.06 帧被监督**、非 blank 12.5%。
  - 【间接】CLIP-SLA（arXiv:2504.01666）在**冻结 CLIP 编码器**上加辅助 CLIP-CTC 损失，
    是「冻结主干 + 密集监督」的先例；但本地 CLIP 线性探针 Cohen d 0.1006 判死，不能照搬骨干。
- **与已判死项的区别**：P185 多尺度 CTC、P196 WE/WC、D2 一致性正则都是**序列级/逐帧 max 的弱辅助**；
  H3 是**持续区间监督**，梯度密度不同。
- **⚠️ 反证据**：本地 D3 已证 TFD 无参数边界质量差（corr 0.29、lift 1.17）、
  P206 landmark 融合判负 ⇒ **边界来源**是 H3 最大障碍。
  避开 LLM 的本地替代：用 CTC 强制对齐 + DPLR 式迭代伪标签生成区间（无需外部模型/标注）。
  另一个真问题：密集监督的经典机理是「改造特征提取器」，主干冻结后边际收益是否成立未知
  ⇒ 主指标可改为「达到目标准确率所需 GPU-hours / step 数」，而非只看最终 WER。
- **成本**：高（生成伪标签 + 改损失 + 6 轮训练，约 1 天工程）。
- **判据**：统一 R1–R8 + 非 blank 帧占比上升 + 达到 34.80 所需 step 数下降。

### H4. 不推荐 / 已排除

| 方向 | 不推荐理由 |
|---|---|
| 换 CLIP / AdaptSign / CLIP-SLA 骨干 | 【有】本地 CLIP 线性探针 Cohen d **0.1006** 判死；且改 3516 词表会破坏全部可比性；8G+6轮预算不匹配。AdaptSign(arXiv:2404.08226)/CLIP-SLA(arXiv:2504.01666) 本身有效，但「验证冻结 CLIP 可行」已被别人做透 |
| 再加轻量损失项 / 重排 / 增广 / 蒸馏 / 先验校正 / 原型对比 | D1–D4、E0/E1、F0/F1/F2、P218、G1 全部判负 |
| 全量微调、更长训练、Conformer、224 输入 | P171/P199 灾难性遗忘、P184-cont 无增益、P198/P200 判负、P186 判负 |
| 分句 / 分词方向（本地 04_SignShift、05_TLAS、07_TFD、03_Wójcicka） | 属 SLT 句子边界任务，与 CSLR 的 WER 主任务不同；且 D3 已证本地边界质量不达标 |
| 生成式 / decoder-only 全替换 | P200 已证随机初始化大时序模块 6 轮内拟合失败 |

## 3. 推荐执行顺序（全部预注册后再启动）

```
H0 ✅ 已完成（本方案）：本地论文库无重叠核查 + 记忆库规模估算（PCA128+fp16 ~70MB 可行）
H1-a 建 train 帧级特征记忆库（改造 G0 前向脚本导出特征，GPU 20–30 min，非训练）
H1-b kNN 后验精修 λ 扫描（CPU ~10 min）→ R1–R8 判定；λ 选择规则先写死
      └─ 过线才考虑 2 seed；不过线则 H1 判负，不调 λ
H2-a TTA 前提诊断：dev 加 corruption 测 WER 跌幅（纯推理 ~30 min）
      └─ 跌幅 <2pp ⇒ H2 STOP（同分布下 TTA 无空间）
H2-b 仅当 H2-a 过线：参数自适应 + collapse 监控
H3  密集监督（DPLR 式伪标签，成本最高，最后投）
```

任一步落在 dev 噪声地板 ±0.77pp 内即判「无差别」，不进 test。
**所有新实验仍必须同时报 WER_official + uWER + 低频桶 + 高频桶 + ins/sub/del + micro，
低频桶单独配对 bootstrap（R8）。**

## 4. 诚实预期

【无】三条都需要新工程（记忆库 / 自适应循环 / 伪标签），都不是「再加一个轻量损失项」；
且 CE-CSL dev/test 同分布会让 H2 的收益前提大概率不成立。
按第三轮 12 连败的轨迹，H1 落进噪声内的概率同样不低。
但 H1 是**唯一零训练、参数无关、不占 6 轮预算**的路线，且直接打
「近义 gloss 细粒度混淆 + 48% 单样本类」这两个已定位的真瓶颈，
因此优先级最高。若 H1/H2 仍无信号，合理结论是**接受天花板收尾**，
而不是继续找新损失项（与 `OPT-DIRECTIONS-3` §6 一致）。

## 5. 来源清单

- 外部（2026-11-08 检索）：
  - AdaptSign, arXiv:2404.08226（冻结 CLIP + 轻量模块，3.2% 额外计算）
  - CLIP-SLA, arXiv:2504.01666（SLA-LoRA / SLA-Adapter，PHOENIX14 18.8% WER，PEFT 优于全量微调）
  - SAME, ACL 2026（SLT 首个 TTA，signer-aware MoE）
  - TRANSLATE, IEEE TMM 2026, DOI 10.1109/TMM.2026.3730967（SLT 腐败偏移 TTA）
  - Multimodal Retrieval-Augmented Recognition for Thai Sign Language Fingerspelling, IEEE 2025
    （kNN 后验精修、data-leakage-controlled 知识库）
  - CSLR², Raude et al., ECCV 2024, arXiv:2405.10266（句级检索 + CSLR 互益）
  - RAC 通用框架（分布插值 / 实例记忆 / FAISS-HNSW）
  - Beyond CTC Spikes, Zhu/Jin/Fang, EMNLP 2026, 2026.emnlp-main.578（GTM 密集区间监督）
  - DPLR, Jang et al., ICASSP 2023（迭代伪标签密集化）
  - TENT, ICLR 2021（熵最小化 TTA 通用范式）
- 本地：`ROUND3-SUMMARY-2026-11-08.md`、`OPT-DIRECTIONS-3-2026-11-08.md`、
  `stage-P219-subpair-semantics.md`、`stage-P220-variant-split.md`、
  `stage-P221-g0-prior.md`、`stage-P222-G0-train-confusion.md`、
  `新参考论文/说明.txt`（04/05/06/07 篇）、`LIT_PLAN_论文方法实施计划.md`
