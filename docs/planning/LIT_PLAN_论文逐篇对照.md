# 论文逐篇对照表（读论文时直接查这里）

> 配套文件：`LIT_PLAN_论文方法实施计划.md`
> 图例：✅ 已应用有效 / ❌ 已应用无效 / 🔴 应用中发现 bug / ⚠️ 作废 / ⏳ 未应用

---

## A. 参考论文/papers/（17 篇）

### ref11_Yan_ST-GCN — Spatial-Temporal Graph Convolution Network
**核心主张**：直接用关节坐标、不建模空间关系是不够的（"do not explicitly exploit the spatial relationships among the joints"）。

| 我们的动作 | 结果 | 收据 |
|---|---|---|
| 把 landmark 拍平成 126 维向量（**丢掉拓扑**） | 这是我们当前状态 | — |
| 实现 ST-GCN 骨架拓扑 | ❌ train 完美 dev 0.6938；22.24M 参数 dev **0.7311** | `p29` |
| 结论 | 拓扑信息有用，但**加速记忆训练集**，不是泛化瓶颈 | — |
| **仍值得做** | 复用它的**邻接矩阵定义**，配合 §2.4 的 motion 段加权 | §6 P2-9 |

**⚠️ 铁律**：我们只把 21 点手骨拍平，丢了骨骼连接拓扑。这是 P28「特征到顶」最可能的结构原因，有明确文献依据 + 可实施 + 不需重提特征。

---

### ref23_Arib_SignFormer-GCN — ST-GCN + I3D 双流
**核心主张**：骨架流与 RGB 流**相加**融合（式 7：`Fused = Z_t + L_t`）。

| 我们的动作 | 结果 |
|---|---|
| RGB + landmark 相加 | ❌ rgb+lm **0.5314** > lm_only **0.5211** |
| RGB 单独 | ❌ rgb_only 0.6949 |

**含义**：在 CE-CSL 上 landmark 路线优于 RGB，双流相加没带来收益。
**注意**：论文原话"most works rely solely on RGB features"是**反向证据**（说明 RGB 也有用），但对我们不适用。

---

### ref07_Zhou_Sign_Back-Translation — SignBT
**核心方法**：Back-Translation（用 gloss-free SLT 生成伪 gloss）+ CTC + beam search。

| 我们实验过的部分 | 结果 |
|---|---|
| Random shift + discard/copy 20% frames | ❌ 最好 combo **−0.0060**（`p35`） |
| CTC + beam + 词表约束 | ⏳ 未跑完（`p47b` 太慢）|
| Back-Translation 本身 | ⏳ 未尝试 —— **需要先有 gloss-free SLT 模型** |

**仍值得做**：beam search width=10（官方 TFNet 也用）→ §6 P1-6

---

### ref15_Camgoz_Sign_Language_Transformers — SLT 基础
**核心方法**：CTC + beam search + 词表约束。

**我们的相关实验**：
- ✅ 受控解码实验（P47）—— 词频先验**无效**，但机制有价值：
  正确词 top-1 仅 **10.6%**、top-5 **44.8%**，所以重排无从下手
- ⏳ beam search width=10 未跑完

---

### ref12_Wu_CrossModal_Consistency_SLR — CCL-SLR 跨模态对比预训练
**核心方法**：跨模态一致性对比预训练（RGB + 骨架两个视图）。

| 我们的动作 | 结果 |
|---|---|
| CLIP RGB 线性探针 | ❌ Cohen d **0.1006** 极弱；heldout 0.0222 < 基线 0.1833（`p36`） |
| Spatial-temporal augmentation | ❌ `p35` |
| 跨模态对比 | ⏳ 未尝试 |

**关键判断**：CLIP 特征对 gloss 身份几乎无判别力 → 这条路的**前提就不成立**。

---

### ref03_DeCoster_MT_signed_to_spoken — 综述
**核心观点**：gloss 标注**无统一标准**；小数据上 data-driven SL 系统「cannot reach high qualities and be generalizable」。

**用途**：FYP 报告引用 —— 解释为什么 CE-CSL 官方数据集本身难。

---

### ref20_Desai_biases — 101 篇元研究（biases in sign AI）
**用途**：FYP 报告可引用的**领域批评**，用于讨论标注偏差与评估口径。

---

### ref07/ref15 词表约束解码汇总
详见本文档 §4「受控解码」。

---

### ref08_Lugaresi_MediaPipe — MediaPipe
**我们正在用的工具**。今天发现的三个 bug 全部与它的 VIDEO 模式有关：
- 时间戳必须单调递增（跨调用复用会抛异常）
- `handedness[i]` 必须按索引取，不能恒取 `[0]`
- 新旧 API 的 handedness **语义相反**（图像视角 vs 解剖学）

---

### ref09_Cao_OpenPose / ref10_Moryossef_Pose_Estimation — 姿态估计
**备用特征提取器**。⚠️ 我们的 CE-CSL **只有上半身**（无腿脚），OpenPose 的价值有限。

---

### ref13/ref14 — 轻量 SLR / 双向 Reservoir
未系统测试。`p18-config-c-char.json` 曾试 char 层级 target。

---

## B. 新参考论文/（7 篇，全部未应用）

### 06_SMART — ⭐⭐ 与我们的问题最同源
**核心诊断**：CTC 产生**峰值对齐**（peak alignment）——每个 gloss 仅由少数帧预测，
其余帧分给 blank。这种**弱时序监督**限制密集帧级表示学习。

| 我们的实测 | 对照 |
|---|---|
| 87.4% 的帧 argmax 是 blank | 论文说 blank 97%（**我们已降到 90.47%**）|
| 非 blank 帧/句 **6.1**，参考均长 5.52 词 | → **帧数:词数 = 8.7:1 过于宽裕** |

**论文方法**：CSFormer = 边界感知时序分割骨干，联合执行识别 + Sign Spotting，
用识别产生的 gloss 证据注入定位网络。

**⇒ 计划 §6 P3-13**。这篇**直接针对我们的 blank 率问题**，是最相关的未应用论文。

---

### 07_TS²-TFD — 自监督时序分词
**核心方法**：TFD（Temporal Feature Difference）= 当前帧特征与局部时序平均特征的 L2 距离；
边界 = TFD 信号超自适应百分位阈值的局部最大值。
**复杂度 O(N)，CPU 可跑，无可训练参数**。免训练版就超过所有对比的无监督方法。

**计划 §6 P3-12** —— **最适合先试**，因为零训练成本。

---

### 01_HandsOn_FG2025 — 手形作为边界信号
**关键定量证据**：SignBank 中 **>60% 的手语词只使用单一手形** → 手形变化本身就是强边界线索。
**方法**：HaMeR 手部网格恢复 + 3D 骨骼角度；用 **BIO 标注**处理相邻片段间无 O 帧的情况。

**我们的现状**：⚠️ **Kinect 25 关节不含手指**，MediaPipe 21 点是唯一手部来源。
**计划 §6 P3-10**（HaMeR）+ P3-14（BIO）

---

### 02_LinguisticallyMotivated_EMNLP2023 — 分词+分句联合建模
**核心主张**：分词和分句**联合建模**而非独立任务；用 **BIO 替代 IO**，论文明确说 BIO 对手语边界是必要的。
**韵律线索**：短语分割靠停顿、手语时长延长、面部表情；探索用光流显式建模。

**计划 §6 P3-14**

---

### 03_Wojcicka_LREC2026 — 面部韵律 ⭐⭐
**定量结论**：融合面部韵律比单模态基线**提升 13.6 个百分点**。
**方法**：MS-TCN + 通道注意力，融合 MediaPipe 的手/身体/面部骨骼特征。
**结果**：Segmental F1 75.43%（IoU=0.10）、57.52%（IoU=0.50）。
**非手动韵律标记**：眨眼、头部倾斜/摇头、面部表情、目光转移、身体倾斜。

⚠️ **但我们 P34 把面部从 8 点扩到 128 点只提升 1.01x**。
⇒ 差异可能在于：他们用**韵律动态**（眨眼等时间模式），我们用的是**静态坐标**。

**计划 §6 P3-11**

---

### 04_SignShift — Vis-SSLS 无字幕分句
**核心修正**：很多句子边界**没有显式停顿**。手语中的句子转换通常是**平滑且视觉模糊**的，
缺显式停顿或姿态重置，静态帧表示无法捕捉。
**方法**：时序差异模块，融合全帧+面部+手部线索，用**帧间差分**捕捉局部运动学和全局语义过渡。
另有**片段数量预测模块**缓解过/欠分割。

**⇒ 对我们直接相关**：分句/分词不能只靠运动速度阈值。

---

### 05_TLAS_ALVR2026 — 流式分句工程方案
**关键参数**（可在自有数据上重新校准）：
- 句子内**词间间隔 300–650 ms**
- 句子之间**停顿 2–7 s**
**方法**：时序停顿检测器（EMA 追踪词间隔）+ 语言就绪度评估器（冻结 T5 上的神经头）+ 自适应融合门控。
**主动超时**：词间隔超阈值就在下一个词到达前主动触发分句，**不需预言边界信息**。

**计划 §6 P3-15** —— 成本最低的可落地方案

---

## C. CE-CSL 官方论文（arXiv:2409.11960v2）

| 我们采用的做法 | 官方原文依据 | 我们的实测 |
|---|---|---|
| **随机裁剪 256→224 + 翻转 0.5** | 训练设置明确列出 | ⏳ **从未测过**（P35 只测 landmark 扰动）|
| **时序增强 ±20%** | "视频长度随机增加或减少 ±20%" | ⏳ **从未测过** |
| **beam search width=10** | 评价设置 | ⏳ P47b 未跑完 |
| **词表 3515** | 报告 Vocabulary 3515 | ⏳ 正在扩（当前 300）|
| Adam / lr 1e-4 / batch 2 / 55 epoch | 训练设置 | 我们用 AdamW / 1e-3 / batch 16 / 100 epoch |

**⚠️ 协议红线**：官方是**按数据集分别训练、分别评估**，无混合训练。
我们混用任何其它数据集都会让 42.1% 这个基准**不可比**（P60 四层论证）。

**⚠️ 官方 6 个模型全部用 RGB，landmark 路线无文献对照** —— 这是本 FYP 的立足点。

### ⭐ 官方论文自己引的两篇（2026-10-06 补检，本地原缺）

**ref24 Hu et al. — CorrNet, CVPR 2023** ✅ 已下载
`参考论文/papers/ref24_Hu_CorrNet_CVPR2023.pdf` · 官方代码 `github.com/hulianyuyy/CorrNet`

原文（直指我们的困境）：
> "current methods in CSLR usually **process frames independently, thus failing to
> capture cross-frame trajectories** to effectively identify a sign... a **correlation
> module** is first proposed to **dynamically compute correlation maps between the
> current frame and adjacent frames** to identify trajectories of all spatial patches."

| 我们采用的做法 | 官方原文依据 | 我们的实测 |
|---|---|---|
| **帧间轨迹建模（CorrNet）** | 上述原文 | ⏳ **从未测过**。我们把 48 帧压成 1 个 mean-pool 向量 = 「process frames independently」的极端形态 |
| CE-CSL 47.2/46.5 的基线 | Table VII | CorrNet 是 TFNet 的直接对照，官方认可其为强基线 |

⚠️ 官方 `external/TFNet/Module.py` 里**已实现 `resNet18Corr`**，`Net.py` 里已有 `"CorrNet"` 分支
⇒ **不必自己写**，直接 `moduleChoice="CorrNet"` 即可。

**ref25 Hu et al. — AdaSize, Pattern Recognition 2024** ⛔ 全文拿不到
DOI `10.1016/j.patcog.2023.109903` · Vol.145, 109903
⛔ ScienceDirect 反爬（返回 HTML 非 PDF）/ ⛔ arXiv 无预印本 / ⛔ S2 `openAccessPdf=CLOSED`

仅可引abstract：
> "**spatial redundancy** in CSLR... **not all frames are equally important**... lightweight
> 2D CNN first browses input frames under a **low resolution (e.g. 112×112)**"
> 结果：**0.38×计算 / 0.41×显存 / 1.25×吞吐**，精度与 SOTA 相当。
验证集：PHOENIX14 / PHOENIX14-T / CSL-Daily / CSL（**不含 CE-CSL**）

⚠️⚠️ **它调的是每帧的空间分辨率，帧数不变 ⇒ 不能作为「抽稀到 48 帧」的依据。**

### ⭐ 官方论文关于帧的全部原文（2026-10-06 逐字核查）

关键词全文命中：`fps` **1 次**（仅 Table I 表头）· `frame rate` **0** · `frames per second` **0**
· `number of frames` **0** · `temporal resolution` **0** · `frame stride` **0** · `resize` **0**

- Table I：CE-CSL 的 `FPS` 与 `Resolution` 列均写 **`varying`**（其他数据集是 25/30）
- §A：「video lengths vary widely, from the shortest at **39 frames** to the longest at **530 frames**」
- Table II：train frames **930,841** ← 与我们 P77 提取出的帧数**完全一致**（交叉验证提取完整）
- Implementation rules：**只谈超参，一个字都没提帧数上限或采样策略**

⚠️ **官方代码也不抽帧**：`DataProcessMoudle.sample_indices(n) = np.linspace(0, n-1, num=n)`。

---

## D. 按你的需求查表

| 你想… | 看这里 |
|---|---|
| 降 blank 率 | §B 的 **06_SMART**（直指 CTC 峰值对齐）+ §2.6 blank 96.98%→90.47% |
| 提升识别准确率 | §6 P1-4/5/6（裁剪翻转 + 时序增强 + beam，都**从未测过**）|
| 换更好的特征 | §6 P3-10（HaMeR）+ ref11（拓扑）+ ref09（OpenPose，收益有限）|
| 分词/分句 | §6 P3-12（TFD，零训练）+ P3-15（TLAS 主动超时）+ 03_Wojcicka（面部韵律）|
| 理解为什么效果不好 | §5 作废实验 + §10 审计铁律 + §1 差距分解 |
| 写 FYP 的立足点 | §1 + 附录 B（landmark vs RGB 无文献对照）|
| **建模帧间关系（2026-10-06 新增）** | **§C 的 ref24 CorrNet —— 官方代码已有实现，`moduleChoice="CorrNet"`** |
| 想知道还有哪些坑 | `ROADMAP_2026-10-06.md` §8（今天踩的 7 类错误）|
| 避免重复踩坑 | §4（8 项无效）+ §5（2 项作废）+ §7（7 项已排除）|