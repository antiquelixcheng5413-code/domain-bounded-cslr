# 第二轮优化方向（2026-11-07 22:00，仅方案，不启动实验）

> 立足点变了：D1–D4 全部判负，但 D4 的 uWER 评估暴露了真正的靶子。
> 输入：D1–D4 汇总报告、2026-11-07 文献检索（domain_search + WebSearch）、
> 本地 `part3-spamo-slt.md` §18/§20 判别力分析、本地特征资源核查、train 词频重算。
> 依据等级（用户约定）：【有】能指出是哪篇哪句｜【间接】相近机制｜【无】我拍的
> 状态：**仅方案**，未启动任何实验。

---

## 0. 一句话结论

之前所有实验（含 D1–D4）都在优化被高频功能词主导的 `WER_official`；
而 **train 中 48.0% 的 gloss 类只出现 1 次、90.4% 出现 ≤10 次**（中位频次 2），
这些类在 dev 上的错误率是 **70.66%**（高频桶仅 18.56%）。
⇒ 剩余方向必须同时满足两条：**直接作用于长尾类别的表征/决策边界**，
且**判据换成 uWER + 低频桶**（否则会像 D4 一样被高频词稀释、把重分配误判为增益）。

---

## 1. 两个新查明的硬事实

### 1.1 极端长尾（本次重算，走训练实际路径 `DPM.PreWords → word2idx`）

| train 频次 | 类数 | 占比 |
|---|---|---|
| = 1 | **1688** | **48.0%** |
| ≤ 2 | 2246 | 63.9% |
| ≤ 5 | 2849 | 81.1% |
| ≤ 10 | 3179 | **90.4%** |
| ≤ 50 | 3453 | 98.2% |

- 3515 类，28272 token，**中位频次 2**，max 3576；
- **top-10 类占 34.0% token**，前 1% 类占 46.9% token；
- dev 仅覆盖 774 类（词表 3515）。

⇒ D4 的「按句稀有度给 CTC loss 加权」在 48% 一类一样本的分布下几乎无力：
加权改变不了这些类**没有足够样本形成判别边界**这一事实。这解释了 D4 为何只是
把错误从低频搬到高频。

### 1.2 判别力探针早已给出互补证据（`part3-spamo-slt.md` §20，此前未与主线打通）

四类特征逐帧均值池化 + 线性探针的 per-gloss macro-AUC（dev 173 个有正例的 gloss）：
landmark 0.6821 / VL48 0.6459 / rgb 0.6265 / motion 0.5354。

> 实词（房子、休息、行李、告诉（我））在所有特征族 AUC **0.93–1.0**；
> 语法功能词（了、又、好不好、决定、走）**0.02–0.25**。

⚠️ 与 D4 的 uWER 结论**表面矛盾、实则互补**：探针用的是「train 频次 ≥5 的 top-200 高频词」
且是**多标签线性探针**；D4 的低频桶是 3516 类 CTC 识别。
合起来的解释是：**实词视觉上可分（AUC 高），但低频实词样本太少学不到；
虚词视觉上不可分，靠 CTC 上下文先验猜对**。
⇒ 长尾问题的性质更接近「**决策边界欠定**」而非「表征无信息」——
这正好是原型/解耦方法的适用前提（LT-SignDiff 2026 的原话）。

---

## 2. 候选方向（按性价比排序）

### E0. 分类头先验诊断 + 解耦重标定 —— 优先级 1，零训练起步

- **依据**：【有】Menon et al. (ICLR 2021) logit adjustment：`ẑ_c = z_c − τ·log π_c`，
   grounded in balanced-error consistency；post-hoc 形式**无需重训**。
  【有】Kang et al. (ICLR 2020) decoupled cRT / τ-normalization：
  "representation learns best from natural distribution, classifier best from rebalanced view"，
  冻结表征后只重训/重标定分类头即可恢复大部分尾部性能。
  【有】LT-SignDiff (Inteligencia Artificial 29(78), 2026)：原型空间生成优于像素空间，
  Macro-F1 +12.4（3782 类、64% 类单样本场景，与本项目 48% 单样本同构）。
- **与已判死项的区别**：D4 是**训练期改 CTC loss 权重**（动表征，判负）；
  E0 是**训练后只动分类器决策边界**，表征冻结 ⇒ 不触碰 D4 已证伪的路径。
- ⚠️ **必须排除的重合项**：P47b「词频校准解码」已判死，原文结论是
  「不是高频词分数高，是其它词分数更低，减 log P 只会把随机词推上来」
  （`。` 平均 logprob −50.97 vs 其它非 blank 类 −7.8）。
  ⇒ **不能**直接照搬 logit adjustment；先做零成本诊断验证前提是否成立。
- **零成本第一步（纯 CPU，用已落盘 `p188_logp_p184.pt`）**：
  1. 算 `corr(train 类频次, 该类平均 logp)` 与 `corr(频次, 分类头权重范数)`；
     若前者不显著为正、后者不随频次增长 ⇒ **P47b 结论在 TFNet 上仍成立，E0 直接判死**，
     不投任何训练。
  2. 若成立 ⇒ 扫 τ∈[0,0.2,…,1.0]，在 **dev** 上报 `WER_official + uWER + 低频桶 WER`
     + 配对 bootstrap；**判据主看低频桶**。
- **成本**：诊断 ~10 min CPU；若通过，τ 扫描零训练；cRT 重训分类头 <1 h。
- **判据（预注册）**：低频桶 WER 改善 ≥2pp 且 uWER 改善 ≥1pp，
  同时 `WER_official` 退化 ≤0.3pp（允许小幅总量换尾部，但必须显式报出）。

### E1. 跨句 gloss 原型对比（CSGC）—— 优先级 2

- **依据**：【有】Rao et al., AAAI 2024, `10.1609/aaai.v38i5.28265`：
  "a prototype is maintained for each gloss category ... gloss contrastive loss to pull
  samples to their prototype and push away from others"，CSL-Daily **−5.7%** /
  PHOENIX14 −1.6% / PHOENIX14-T −2.4% WER。
- **机制**：跨句为每类维护动量原型 + 辅助相似度分类器（ASFS）。
  与 D4 的 loss 加权、D2 的双视图一致性、P196 的逐帧 max 辅助**都不同**：
  它改的是**表征几何**，且对单样本类可从跨句同类/近邻借力，直击 48% 单样本问题。
- **成本**：中（3515×1024 原型 ≈ 3.6M 参数，8G 可放；加对比损失 + 6 轮）。
- **风险**：论文从头训 55 轮；本项目冻结主干 + 6 轮，原型动量可能来不及收敛。
  ⇒ 先跑 **12 轮小档**（约 6–8 h）看低频桶是否动，再决定要不要全量。
- **判据（预注册）**：uWER 改善 ≥2pp 且低频桶改善 ≥2pp，配对 bootstrap CI 不含 0；
  仅 `WER_official` 改善不算数。

### E2. 视觉-语言细粒度一致性（FGXM）—— 优先级 3

- **依据**：【有】Ke/Liu/Feng, Pattern Recognition Letters 191 (2025) 23–30,
  `10.1016/j.patrec.2025.02.017`：正是提出 uWER 的那篇，明确指出
  "perform well with high-frequency glosses ... struggle to accurately recognize
  low-frequency content words"，FGXM 是 **training-time only、推理零开销、不需额外数据或新模态**
  的即插即用模块，作用于潜在特征序列而非输出层。
  【间接】ACMC (2025)、MMCST (IEEE SPL 2026) 均为跨模态细粒度对齐。
- **与已判死项的区别**：D1 是**解码后用 LM 重排**（判负）；E2 是**训练期约束表征**。
  P174/P175 是软标签 KD（判死）；E2 是特征级跨模态一致性，不是 KD。
- **可用资源**：本地 Qwen2.5-1.5B（3.09 GB）可作中文 gloss 文本编码器；
  本地 `ce-csl-qwenvl48`（5488 条 `[48,2048]`）与 landmark96（5487 条）已落盘。
- ⚠️ **反证据**：本地 CLIP RGB 线性探针 Cohen d **0.1006** 判死（`LIT_PLAN` §A ref12），
  说明冻结 CLIP 特征对 gloss 身份判别力极弱。但那是**冻结特征判别力**，
  不等于**训练中对齐**无效，两者机制不同，仍需验证。
- **成本**：中高（文本侧编码 + 细粒度对齐损失 + 6 轮）。
- **判据（预注册）**：同上，主看 uWER / 低频桶。

### E3. 手形音系感知辅助监督 —— 优先级 4

- **依据**：【有】Zhang & Duh, EMNLP 2023 Handshape-Aware SLR：手形最小对立体识别率
  仅 **~20%**（运动类 ~87%），帧级 CE 伪标签 + 双编码器在 PHOENIX14T 达 20.3/21.8% WER。
  【有】Yin et al., arXiv:2606.28667 (2026)：建议把最小对立体作 **hard negatives**
  用于对比学习；姿态流强于手形、像素流强于位置 ⇒ 双流互补。
  【有】本地 `01_HandsOn_FG2025`：SignBank >60% 手语词只用单一手形。
- **机制**：用本地 MediaPipe 21 点手部 landmark 做**无监督手形聚类**出帧级伪标签，
  加轻量辅助头 → 给 CTC 补密集监督（不是找边界，与 D3 的 TFD 不同任务）。
  直击 P191「替换 67.5%、同类手势细粒度区分」。
- ⚠️ **反证据**：D3 证实 landmark48 的 TFD 边界质量差（corr 0.29）、
  P206 landmark 融合判负。手形聚类是否也失效未知，需先探针。
- **零成本第一步**：对 landmark 手部点做 k-means，算聚类纯度 /
  与 gloss 类的互信息；若聚类与 gloss 几乎独立 ⇒ 直接放弃。
- **成本**：高（需实现手形头 + 帧级伪标签 + 6 轮）。

### E4. 不推荐 / 超出预算

| 方向 | 依据 | 不推荐理由 |
|---|---|---|
| CR-CTC 完整版（time-masking + masked prediction） | 【有】ICLR 2025 小米 | D2 只做了对称 KL 已判负；虽未实现 masked prediction，但同族群，先验下调 |
| 换解码范式（permutation Transformer decoder / AutoSign） | 【有】ICCVW 2025 / arXiv:2507.19840 | 需从零训 decoder；P200 已证随机初始化大模块在 4973 条 / 6 轮拟合失败（40.04） |
| Qwen-VL48 特征融合进 TFNet | 【间接】本地已落盘 5488 条 | P206 landmark 残差融合判负；且 part3 的 VL48 全量实验是 cap300 词表，与 3516 词表主线不可比 |
| HaMeR 手部网格 / 面部韵律分句 | 【有】本地 01/03 | 需装新模型 + 重新提特征；且分句不是当前 WER 瓶颈 |
| 通用增广 / 蒸馏 / 换主干 / 解码重排 | 【有】本项目实测 | P185/P196/P198/P202/P203/P206/P174/P175/P195/P200/D1/D2 全判负 |

---

## 3. 方法论硬约束（必须由 D4 的教训固化）

1. **判据换靶**：后续任何实验除 `WER_official` 外**必须**同时报
   `uWER`（类别等权）与 `低频桶 WER`（train freq ≤10）。仅总量改善不构成正向。
2. **口径**：`WER_official` 沿用 `official_wer.evaluate`（语料级，可比 34.34）；
   uWER/桶 WER 用 `p210_d4_eval.py` 的 `align_ops`；低频桶 692 token，
   **需单独配对 bootstrap**，不能套 ±0.77pp（那是整句 515 的噪声地板）。
3. **防重分配**：必须核对高频桶不恶化、按 token 加权的 micro 指标不恶化，
   否则就是 D4 式的「拆东墙补西墙」。
4. **test 冻结**：dev 过线后才考虑一次性 test；本轮方向一律不碰 test。
5. **零成本优先**：每条方向都先设「前提是否成立」的 CPU 诊断，
   不成立就直接判死，不投 GPU（D3 的 TFD 探针就是这么省下 1 天工程的）。

## 4. 推荐执行顺序

```
E0-诊断  类频次 vs logp / 分类头范数（10 min CPU，用已落盘后验）
  ├─ 前提不成立（P47b 在 TFNet 上复现）──► E0 判死，直接跳 E1
  └─ 成立 ──► τ 扫描（零训练）──► cRT 重训分类头（<1 h）
E1  CSGC 原型对比（先 12 轮小档）── 主看 uWER/低频桶
E2  FGXM 视觉-语言一致性
E3  手形聚类纯度探针 ── 有信息才做辅助头
```

⚠️ **诚实预期**：D1–D4 已把「冻结主干 + 6 轮 + 4973 条」预算内的增量路线系统穷尽。
E0/E1/E2 里真正可能突破的，都需要**动决策边界 / 表征几何 / 训练轮数**，
而不是再加一个轻量损失项。若 E0 诊断判死、E1 小档无信号，
合理结论是**接受天花板**，而不是继续找第 5 个轻量增量。

## 5. 来源清单

- 本地：`stage-D1-D4-summary-2026-11-07.md`、`stage-P210-D4-class-balance.md`、
  `part3-spamo-slt.md` §18/§20、`LIT_PLAN_论文方法实施计划.md` §4（P47b）、
  `p188_logp_p184.pt`、`p210_d4_eval.py`、`ce-csl-qwenvl48/`、`ce-csl-landmark96/`
- 外部（2026-11-07 检索）：Menon et al. ICLR 2021（logit adjustment）；
  Kang et al. ICLR 2020（decoupled cRT/τ-norm）；LT-SignDiff, Inteligencia Artificial
  29(78) 2026, DOI 10.4114/intartif.vol29iss78pp215-232；
  Rao et al. AAAI 2024, DOI 10.1609/aaai.v38i5.28265（CSGC）；
  Ke/Liu/Feng PRL 191 2025, DOI 10.1016/j.patrec.2025.02.017（FGXM/uWER）；
  Zhang & Duh EMNLP 2023（Handshape-Aware）；
  Yin et al. arXiv:2606.28667（最小对立体审计）；
  domain_search 综合（含 PhonSSM arXiv:2604.08761、RAIDAL、diffusion CSLR）。
