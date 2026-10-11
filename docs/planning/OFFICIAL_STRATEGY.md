# 官方策略档案 — CE-CSL / TFNet

> **本文用途**：记录官方（论文作者 Zhu Qidan）对 CSLR 各瓶颈的应对策略与完整配置，
> 作为后续实验设计、复现对齐、差距归因的依据。
>
> **来源与核实方式**（三级，逐项标注）：
> - 【论文】= 从本地 PDF 提取的原文（`_ce-csl-paper.pdf`，arXiv:2409.11960v1，10 页 / 51,179 字符）
> - 【代码】= `external/TFNet/` 逐行读源码
> - 【README】= 官方仓库 README
> - 【推断】= 我方推理，无原文直接支撑
>
> **核对时间**：2026-10-10
> **配套文档**：`OFFICIAL_TFNET_SPEC.md`（代码结构核对，2026-10-06）、
> `OFFICIAL_EVAL_PROTOCOL.md`（评测口径）

---

## 1. 官方对四类瓶颈的应对策略

### 1.1 帧级边界缺失 → **时域/频域双分支，用架构先验替代边界监督**

【论文 Fig.3 + 式(1) 起】

```
输入 V = (x_1, ..., x_T) ∈ R^{T×C×H×W}
  ↓
F_f  帧级特征提取（CNN）→ f_frame ∈ R^{T×C'}
  ↓
  ├─ 时域分支：f_frame → F_temporal（1D CNN + BiLSTM）
  └─ 频域分支：f_frame → DFT → F_spectral（1D CNN + BiLSTM）
  ↓
两分支输出合并 → 分类器 → 句子预测
```

【论文原文】*"the model is designed with two independent yet complementary branches.
In one branch, we model the temporal sequence using a combination of 1D CNN and
Bidirectional Long Short-Term Memory Networks (Bi-LSTM) to capture the temporal
dynamics of sign language actions. In the other branch, we transform the temporal
features into the frequency domain via Discrete Fourier Transform (DFT) and
similarly model the frequency-domain sequence…"*

【代码】两分支的 sequence extractor 结构完全相同（都是 TemporalConv + BiLSTM），
区别仅在于频域分支前多一步 `fft(framewise, dim=-1)` → `abs()`。

**实际增益（⚠️ 消融在 RWTH 上做，不是 CE-CSL）**：

【论文 Table IX】
| 域类型 | Dev(%) | Test(%) |
|---|---|---|
| Temporal domain only | 19.5 | 19.1 |
| Frequency domain only | 19.7 | 20.0 |
| **Time-frequency domain** | **18.7** | **18.6** |

⇒ **时频融合只值 0.8 pp（Dev）/ 0.5 pp（Test）。**

【推断】DFT 把时序转到频域后，卷积在频域对应"周期匹配"，理论上更易捕捉全局节奏；
但实测增益很小，不应把时频双域当作主要收益来源。

### 1.2 复杂背景 → **MAM（Motor Attention Module）**

【论文标题即 "Based on Complex Environments"】
论文动机：*"most publicly available datasets are limited to laboratory environments
or television program recordings, resulting in a single background environment with
uniform lighting, which significantly deviates"*（实际场景）

【代码】`Module.py` 中 `ResNet34MAM`：
- `conv1 = Conv3d(kernel=(1,7,7))` ⇒ **时间维在第 0 位**，输入布局 `[B,C,T,H,W]`
- `MotorAttention` kernel = `(k,1,1)` ⇒ 卷积核第 0 位是**时间**
- `forward` 返回 **4 元组** `(x, outData1, outData2, outData3)`，第 0 个是最终特征 `[B*T, 512]`
- 内部加载 **ImageNet 预训练 resnet34**：`model_zoo.load_url(model_urls['resnet34'])`，
  并把 2D 卷积核 `unsqueeze(2)` 升成 3D

⚠️ **三个静默出错点**（不实测就会存错数据）：
1. 输入是 `[B,C,T,H,W]` 而非 `[B,T,C,H,W]`
2. 逐帧推理必须给 `T=1`，否则报「got 16 channels」
3. `forward` 返回 tuple，直接 `np.save` 会存成多尺度特征图

### 1.3 低频词 → **官方完全未处理，也未讨论**

【论文】`grep -c -iE 'rare word|low.frequen|unseen gloss|rare gloss|low.resource'` = **0**

⇒ 全文无任何低频词相关论述。**这是官方与我们的共同盲区。**

### 1.4 数据量 → **官方用的与我们完全相同**

【论文原文第 271 / 754 行】
> "there are **4,973** sign language videos in the training set, **515** in validation"

| split | 数量 |
|---|---|
| train | **4,973** |
| validation | **515** |
| test | **500** |

词表：**3,515**（train+dev+test 全收、无截断）

⇒ **我们的数据量与官方逐项一致。**
⇒ **「数据密度低（7.4 token/词）不是我们的宿命** —— 官方在同样数据上做到 42.1 %。
⇒ 与官方的 18.84 pp 差距**不是数据量造成的**。

---

## 2. 官方模型对照表（CE-CSL 主表）

【论文 Table VII】

| 模型 | Dev(%) | Test(%) |
|---|---|---|
| MSTNet | 54.4 | 53.0 |
| CorrNet | 47.2 | 46.5 |
| SEN | 46.5 | 45.3 |
| **VAC**（我们当前主线） | **45.1** | 43.3 |
| MAM-FSD | 44.9 | 44.7 |
| **TFNet** | **42.1** | 41.9 |

【README】另称提供的权重 test WER 可达 **32.46 %**（与表格 41.9 不一致，来源不明，勿混用）。

### ⚠️ 关键推论：差距的性质
- 官方从 VAC(45.1) → TFNet(42.1)，**全部架构改进仅 2.8 pp**
- 我们用 VAC 得 **60.94 %**，比官方 VAC(45.1) **差 15.84 pp**
- **15.84 ÷ 2.8 = 5.6 倍**

⇒ **问题不是「我们少了时频双域」，而是「我们的 VAC 复现本身远差于官方的 VAC」。**
⇒ 追架构（TFNet/CorrNet）不可能补上这个洞，**应先查训练策略与复现细节**。

---

## 3. 官方完整配置清单（可直接照做）

### 3.1 训练

| 项 | 官方 | 我们 | 对齐？ |
|---|---|---|---|
| epoch | **55** | 20 | ❌ |
| lr 衰减 | **35 / 45 epoch 各降 80 %** | `lr_milestone_ep=13`（按总轮数折算） | ❌ |
| batch | 2 | 2 | ✅ |
| 优化器 | Adam | Adam | ✅ |
| **水平翻转** | **hflip p=0.5** | **`no_hflip=True`（关闭）** | ❌ |
| 时序增强 | **±20 %** | 无 | ❌ |
| 图像增强 | RandomCrop 256→224 | — | 待核 |
| 词表 | 3515 全收 | 3515 全收 | ✅ |
| seed | 未说明 | 42 | — |

【README】55 epoch，lr 在 35/45 降 80 %，batch 2，Adam
【代码】`videoAugmentation.py`：随机裁剪 256→224、水平翻转 p=0.5、时序 ±20 %

### 3.2 推理

| 项 | 官方 | 我们 |
|---|---|---|
| 解码 | **beam search, width 10** | **greedy** |
| 库 | `ctcdecode==0.4`（`parlance/ctcdecode`） | 未装 |

⚠️ **在补上 beam search 之前，我们所有「距官方 X pp」的说法都不可信。**

### 3.3 数据预处理

【代码】`CE-CSLDataPreProcess.py`：逐帧 `cv2.resize(image, (256,256))` 存 jpg
⇒ **不做任何关键点提取**，纯 RGB。

---

## 4. 评测口径

【论文式 11，引用 Koller et al.】
```
WER = 100% × (ins + del + subs) / sum
  sum = 整个语料的参考 gloss token 总数（corpus-level，非逐句平均）
```

### ⭐ 词组粘连：参考与识别**两侧都要拆**
【论文 Table VIII 反推】标注含粘连词（`可以支持`）：

| | 序列 | token 数 |
|---|---|---|
| Gloss ground truth | `我/可以支持/你/去/运动/。` | 6 |
| 拆开后 | `我/可以/支持/你/去/运动/。` | 7 |

【论文 Table VIII Case-2 验证】`我/可以支持/你/去/运动/。` → `我/可以/你/经济/木头/。`
WER 42.9 = 3/7 ⇒ 分母是 **7**（拆开后的长度）⇒ **两侧都拆** ✅ 5/5 精确吻合。

### 标点计入分母
【已验证】`他/小孩时间/…/。` + hyp `他/时间/…/。` → 官方标 50.0 %
4/0.50 = **8 token** ⇒ 标点「。」计入。

**落地模块**：`tools/blank_gov/official_wer.py`

---

## 5. 消融数据（⚠️ 全部在 RWTH/PHOENIX-2014T 上做）

【论文第 683 行原文】
> "In this section, we conduct ablation experiments on the **RWTH dataset** to further
> validate the effectiveness of the model components."

⚠️ **官方没有在 CE-CSL 上做任何组件消融**，只报了主表对比。
⚠️ RWTH 数字（18.7~22.3）**绝不可与 CE-CSL 的 42.1 混用**。

### Table IX — 域序列特征
| 域 | Dev(%) | Test(%) |
|---|---|---|
| Temporal only | 19.5 | 19.1 |
| Frequency only | 19.7 | 20.0 |
| Time-frequency | **18.7** | **18.6** |

### Table X — 序列特征提取器（原 vs 时频）
| Model | 原 Dev/Test | 时频 Dev/Test |
|---|---|---|
| VAC | 21.2 / 22.3 | 19.9 / 20.2 |
| MSTNet | 20.3 / 21.4 | 19.7 / 19.8 |
| SEN | 19.5 / 21.0 | 19.4 / 19.3 |
| CorrNet | 18.8 / 19.4 | 19.2 / 19.1 |
| MAM-FSD | 19.2 / 18.8 | **18.7** / **18.6** |

### Table VIII — 定性案例（个别 case，非整体指标）
| 模型 | Case-1 | Case-2 |
|---|---|---|
| SEN | 50.0 | 42.9 |
| CorrNet | 25.0 | 42.9 |
| VAC | 37.5 | 42.9 |
| MAM-FSD | 0.0 | 28.6 |
| **TFNet** | **0.0** | **0.0** |

---

## 6. 我们 vs 官方：差异清单（待办，按优先级）

| # | 差异 | 我们的值 | 官方值 | 优先级 |
|---|---|---|---|---|
| 1 | **水平翻转** | `no_hflip=True` ❌ | p=0.5 | **最高** |
| 2 | **beam search** | greedy ❌ | width 10 | **最高** |
| 3 | 时序增强 ±20 % | 无 ❌ | 有 | 高 |
| 4 | epoch | 20 ❌ | 55 | 中（我们已饱和，存疑） |
| 5 | lr 衰减点 | 按总轮数折算 | 固定 35/45 | 中 |
| 6 | RandomCrop 256→224 | 待核 | 有 | 待核 |
| 7 | 归因混淆（word_head × warmup） | 未拆开 ❌ | — | **最高** |

---

## 7. 已核实的常见误读（勿重复犯错）

| 误读 | 事实 |
|---|---|
| 「TFNet 论文是 arXiv:2409.11960」 | ✅ 是。但 README 的 "Relevant paper" 引用的是作者前作 **TSRNet (2207.00928)** 与 **TSC (2211.03387)** —— TFNet 的技术基础在这两篇 |
| 「消融 WER 是 18.7，说明时频双域很强」 | ❌ 那是 RWTH 数据集，且只值 0.8 pp |
| 「官方用更多数据所以 42.1」 | ❌ 官方也是 4973 条 |
| 「官方解决了低频词」 | ❌ 全文 0 处讨论 |
| 「ref03 是 Wójcicka 的 CSLR 综述」 | ❌ ref03 是 DeCoster 综述；Wójcicka 论文在 `新参考论文/03_Wojcicka_LREC2026.pdf`（**未提取文本**） |
| 「resnet34MAM 是随机的」 | ❌ 它内部加载 ImageNet 预训练（`model_urls['resnet34']`） |
