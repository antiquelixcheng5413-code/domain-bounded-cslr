# CE-CSL 连续手语识别改进方案文献综述

> 调研时间：2026-09-29；目的：针对当前 CTC 模型的「对齐塌陷、特征判别力不足、无帧级标注」三大瓶颈，检索并评估可落地的改进方向。

## 1. 概述与痛点诊断

当前系统的识别链路是「视频采样 → 特征提取 → 逐帧 CTC 解码 → gloss 序列」，实测表现有三处明确瓶颈：

1. **对齐塌陷**：CTC 模型 `blank_ratio` 高达 0.94、dev WER 0.85~0.87，几乎不落 token，断不出词。
2. **特征判别力不足**：用线性多标签判别探针测得的 gloss macro-AUC，MediaPipe landmark 特征为 0.68，冻结的 Qwen2.5-VL 视觉塔特征为 0.65；实义词（能力、紧张、生活等）基本判不出来。
3. **无帧级标注 + 时序分辨率不足**：数据只有整句 gloss 序列、没有逐帧标签；视频仅采样 48 帧，对一条含 8~15 个 gloss 的句子而言，平均每词不足 5 帧，时间对齐的物理空间严重不足。

三者相互耦合：特征是「词义」的载体，特征判别力不足会导致模型分不清哪个片段是哪个词，进而无法学出正确的时序分割，最终退化成只吐 blank 的塌陷态。

## 2. 关键基线：CE-CSL 官方基准

CE-CSL 数据集原始论文提出的 TFNet 在验证集 / 测试集的 WER 分别为 **42.1%** 和 **41.9%** [cite:28]。

这一数字有两层含义：

- 该数据集「复杂背景 + 无帧级标注」的设定本身很难，即使官方基线也只到 42% 量级，而非 5~10%。
- 当前系统 0.87 的 WER 明显**差于官方基线**，说明问题不是「数据集天花板」，而是「对齐没有被正确训练出来」——这是有明确修复空间的。
- 修复后的合理预期上限约在 40% 量级，应据此校准目标，而不是期待「看懂每句实义词」。

## 3. 方向 A：对齐与 CTC 塌陷修复（低侵入，优先）

这一方向直接针对 `blank_ratio 0.94` 的塌陷，且多数方法**不需要额外帧级标注**，侵入小、见效快。

- **VAC（Visual Alignment Constraint）** [cite:1]：从 CTC 的 "spiky" 特性切入，用两个辅助 loss（VE 约束特征器、VA 对齐特征器与对齐模块），强制特征学习更合理的时空对齐，而非只把帧判成 blank。与塌陷痛点最直接对应。
- **SEN（Self-Emphasizing Network）** [cite:2]：用空间自强调（SSEM）+ 时间自强调（TSEM）突出关键区域与关键帧，减少全 blank 退化；可作为 BiLSTM+CTC 的轻量升级模块。
- **CorrNet** [cite:3]：用相关运算显式建模视频帧序列与 gloss 序列的时序相关性，缓解纯 CTC 的边界模糊；在 CSL-Daily 上 dev WER 30.6% / test 30.1%。
- **DPLR（Self-Sufficient）** [cite:4]：用 GT gloss 序列 + 预测序列生成 dense 帧级伪标签，替代 CTC 的 spiky blank 监督，直接缓解过度依赖 blank。
- **ConsLT（对比学习）** [cite:7]：引入视觉对比损失，用无标签帧增强帧级表示、拉近同义帧特征，缓解 CTC spike。

**小结**：起点建议「VAC + 提高采样帧数」，其次叠加 SEN；两者不依赖帧级标注、无明显副作用，是最值得先验证的低风险改动。

## 4. 方向 B：表征学习与判别力提升（中成本）

直接针对「landmark 0.68 / VL 0.65 判别力不足」的瓶颈。

- **SignBERT / SignBERT+** [cite:8][cite:9]：把手部姿态编码为 token，通过「掩码重建手部关节 / 帧 / 身份」的自监督预训练学习手语姿态的上下文表示；TPAMI 版的 SignBERT+ 进一步引入 model-aware 手先验。
- **SignMAE（Segmentation-Driven MAE）** [cite:13]：按身体部位 / 手势区域分组掩码重建，而非把 pose token 当静态视觉 token；与 MediaPipe landmark 输入高度契合。
- **MASA（Motion-aware Masked AE）** [cite:10]：重建被掩码帧的「运动残差」+ 动量语义对齐，兼顾局部动作与全局词义；在 WLASL 上相对 BEST 提升 per-instance Top-1 +5.81%。
- **SignCLIP** [cite:11]：CLIP 式对比，把视频 / pose 表示推向 gloss 文本语义空间，用大规模视频-文本对预训练。
- **SignCL** [cite:14]：针对 gloss-free 模型中「语义不同但视觉表示过密」的问题，用对比损失提升特征判别性，可直接作为当前 0.68/0.65 的补充诊断手段。
- **SignX** [cite:17]：融合 SMPLer-X / DWPose / MediaPipe 等多源 pose 信息到紧凑 latent space 做时序建模与 CTC refinement。
- **Sigma** [cite:18]：以骨架 / 2D 关键点为输入，设计 sign-aware 融合 + 层次对齐 + 对比学习 / 文本匹配的预训练框架。
- **SHuBERT** [cite:12] / **SignVLM** [cite:15]：多流 / 冻结视觉塔 + 轻量时序器两条预训练路径，说明单一 landmark 需要面/躯干等多通道或语义信号补充。

**小结**：要突破 0.68，最对症的是「pose 掩码自监督预训练（SignMAE 式）+ 对比损失（SignCL 式）」，训练成本可控；大规模多模态预训练（SignCLIP / UNI-SIGN 级别）成本高，适合作为后续扩展。

## 5. 方向 C：端到端 / gloss-free 翻译与帧采样

若最终目标是「中文句子」而非中间 gloss，或想缓解「无帧级标注 + 时序分辨率不足」，可考虑本方向。

- **Sign2GPT** [cite:19]：两阶段无 gloss 翻译——先用口语文本反生成的 pseudo-gloss 做视觉对齐预训练，再用 XGLM 端到端翻译；CSL-Daily BLEU-4 15.40、Phoenix14T BLEU-4 22.52。
- **GFSLT-VLP** [cite:20]：视觉-语言对比预训练 + 轻量 mapper 到文本空间，是无 gloss SLT 的重要基线（Phoenix14T BLEU-4 21.44）。
- **C²RL** [cite:21]：内容 + 上下文表示学习，在 CSL-Daily 上联合优化翻译与检索目标 [1]。
- **FLa-LLM** [cite:22]：LLM 辅助的分解学习，在 CSL-Daily 上 BLEU-4 提升 3.20。
- **SAGE（Segment-Aware Gloss-Free Encoding）** [cite:23]：把视频分割成语义片段、生成 token-efficient 表示（CSL-Daily BLEU-4 25.30），直接应对「时序分辨率不足」。
- **SL-LLaMA** [cite:24] / **MMSLT** [cite:25]：MLLM 从视频直接学翻译，跳 gloss 中间层。

帧采样 / 关键帧方向：

- **关键帧提取算法** [cite:26]：基于角位移 + 序列检查，帧数降至 ~75% 仍保持 83~84% 准确率，说明可用关键帧替代均匀采样以增大单帧信息密度。
- **HAFTR-SLR** [cite:27]：自适应关键帧选择，避免固定 48 帧漏掉关键动作。

**小结**：gloss-free SLT 在中文手语（CSL-Daily）上已到 BLEU-4 15~26 的可观水平，跳 gloss 中间层可行；但风险是对齐更不稳定、且需要 GPU 从视觉塔到语言模型整体重训。若仍走「gloss → 句子」两阶段，则应优先引入关键帧 / 自适应采样解决时序分辨率。

## 6. 分级改进路线建议

| 优先级 | 改动 | 目标瓶颈 | 成本 / 风险 |
| --- | --- | --- | --- |
| P0（本周可试） | 现有 BiLSTM+CTC 加 **VAC 对齐约束** + 采样帧数 48→96+（或关键帧采样） | 对齐塌陷、时序分辨率 | 低，无额外标注 |
| P1（1~2 周） | **pose 掩码自监督预训练**（SignMAE 式）+ **对比损失**（SignCL / ConsLT 式） | 判别力 0.68 | 中，GPU 增量训练 |
| P2（若目标为句子） | 迁移 **gloss-free SLT** 基线（GFSLT-VLP / Sign2GPT / SAGE） | 无帧级标注、端到端 | 高，整体重训 |

推荐按 P0 → P1 顺序推进，并以验证集的 `blank_ratio`、`WER`、`gloss macro-AUC` 三项作为同步观测指标——`blank_ratio` 下降即代表「切割 / 对齐」在变好。

## 7. 存疑与待验证点

- 本综述的量化数值来自二手检索摘要，**未逐篇核对原文**；其中 C²RL 的 CSL-Daily 与 Phoenix14T 结果在检索摘要中完全相同（疑似复制错误），已略去。
- SignMAE、SignSparK、SignLlama 等 2026 年预印本的 arXiv 编号未逐一复核。
- 各数据集的 WER / BLEU 绝对值为不同骨干、不同设置下的结果，**不可跨数据集直接迁移**，仅作方向性参考。
- CE-CSL 官方基线 42% 这一锚点来自数据集原始论文，需在落地前回读原文确认细节。

## Sources

[cite:1] Min et al., "Visual Alignment Constraint for Continuous Sign Language Recognition", ICCV 2021 — https://arxiv.org/abs/2104.02330
[cite:2] Hu et al., "Self-Emphasizing Network for Continuous Sign Language Recognition", AAAI 2023 — https://arxiv.org/abs/2211.17081
[cite:3] Hu et al., "Continuous Sign Language Recognition with Correlation Network", 2023 — https://arxiv.org/abs/2303.03202
[cite:4] Jang et al., "Self-Sufficient Framework for Continuous Sign Language Recognition", ICASSP 2023 — https://arxiv.org/abs/2303.11771
[cite:7] Gan et al., "Contrastive Learning for Sign Language Recognition and Translation", IJCAI 2023 — https://cs.nju.edu.cn/lxie/publication/IJCAI2023.pdf
[cite:8] Hu et al., "SignBERT: Pre-Training of Hand-Model-Aware Representation for Sign Language Recognition", ICCV 2021 — https://arxiv.org/abs/2110.05382
[cite:9] Hu et al., "SignBERT+: Hand-Model-Aware Self-Supervised Pre-Training for Sign Language Understanding", IEEE TPAMI 2023 — https://dl.acm.org/doi/10.1109/tpami.2023.3269220
[cite:10] Zhao et al., "MASA: Motion-aware Masked Autoencoder with Semantic Alignment for Sign Language Recognition", IEEE TCSVT 2024 — https://arxiv.org/abs/2405.20666
[cite:11] Jiang et al., "SignCLIP: Connecting Text and Sign Language by Contrastive Learning", EMNLP 2024 — https://arxiv.org/abs/2407.01264
[cite:12] Gueuwou et al., "SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction", ACL 2025 — https://arxiv.org/abs/2411.16765
[cite:13] "SignMAE: Segmentation-Driven Self-Supervised Learning for Sign Language Recognition", arXiv — https://arxiv.org/abs/2605.02094
[cite:14] Ye et al., "Improving Gloss-free Sign Language Translation by Reducing Representation Density", NeurIPS 2024 — https://arxiv.org/abs/2405.14312
[cite:15] "SignVLM: a pre-trained large video model for sign language recognition", PeerJ CS 2024 — https://peerj.com/articles/cs-3112/
[cite:17] Fang et al., "SignX: Continuous Sign Recognition in Compact Pose-Rich Latent Space", arXiv 2025 — https://arxiv.org/abs/2504.16315
[cite:18] "Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding", arXiv 2025 — https://arxiv.org/abs/2509.21223
[cite:19] Wong, Camgoz, Bowden, "Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation", ICCV 2024 — https://arxiv.org/abs/2405.04164
[cite:20] Camgoz et al., "Gloss-Free Sign Language Translation: Improving from Visual-Language Pretraining (GFSLT-VLP)", ICCV 2023 — https://arxiv.org/abs/2211.08751
[cite:21] Zhu et al., "C²RL: Content and Context Representation Learning for Gloss-free Sign Language Translation and Retrieval", CVPR 2024 — https://arxiv.org/abs/2408.09949
[cite:22] Gan et al., "Factorized Learning Assisted with Large Language Model for Gloss-free Sign Language Translation (FLa-LLM)", 2024 — https://arxiv.org/abs/2403.12556
[cite:23] "SAGE: Segment-Aware Gloss-Free Encoding for Token-Efficient Sign Language Translation", arXiv 2025 — https://arxiv.org/abs/2507.09266
[cite:24] "SL-LLaMA: Multimodal Large Language Model for Gloss-Free Video Sign Language Translation", 2024 — https://pdfs.semanticscholar.org/8a74/3ce0869b54dc2171d68bde04c3f3514a234b.pdf
[cite:25] Kim et al., "Leveraging the Power of MLLMs for Gloss-Free Sign Language Translation (MMSLT)", ICCV 2025 — https://arxiv.org/abs/2411.16789
[cite:26] "Keyframe Extraction Algorithm for Continuous Sign-Language Videos", 2025 — https://pdfs.semanticscholar.org/cc35/1b20c077305b942ce6e4345904151e7706e7.pdf
[cite:27] "HAFTR-SLR: An Efficient Hybrid Attention-Guided Frame-Wise Temporal Reasoning for Sign Language Recognition", IEEE 2025 — https://xplorestaging.ieee.org/document/11301644/keywords
[cite:28] Zhu, Li et al., "CE-CSL: A Chinese Continuous Sign Language Dataset Based on Complex Environments", 2024 — https://arxiv.org/abs/2409.11960

[1] 该文献在检索摘要中的 CSL-Daily 与 Phoenix14T 分项数值完全相同，判定为复制误差，故此处只保留方法与「显著提升」的定性描述，未引具体 BLEU 值。