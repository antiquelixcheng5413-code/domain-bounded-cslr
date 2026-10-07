# 阶段总结：官方代码复现与基线确立（2026-10-06 ~ 10-07）

> 本文档汇总 P77–P92 的工作，可直接作为 FYP 报告的「实验章节」素材。
> 每条结论都标注了**证据等级**与**收据文件名**，可追溯。

---

## 1. 一句话结论

已在 CE-CSL 上跑通**官方 TFNet 代码原版**，并用官方 VAC 架构建立了
**第一个分布无偏的全词表基线：72.94%**（官方基准 45.1%，差 27.84pp）。

**差距构成已逐项核对**：架构、损失、优化器、词表、预训练权重**与官方完全一致**，
差距集中在**训练轮数（6 vs 55，差 9倍）**这一项上。

---

## 2. 最重要的方法论纠正（7 类错误）

这一节是本阶段最有价值的产出 —— 每一条都是「实测推翻自己的假设」。

| # | 错误 | 怎么被推翻 | 教训 |
|---|---|---|---|
| 1 | **照论文转述写实现** | 官方 TFNet 代码在 GitHub（作者 = 论文作者）。逐行读后发现 **6 处写错**：DFT轴向（沿特征维不是时间轴）、主干（ResNet34MAM 不是 Linear）、TemporalConv（pad=0 两次减帧）、输出层（NormLinear 不是 nn.Linear）、辅助损失（SeqKD 不是 VAE）、CTC 配置（`zero_infinity=True`） | **论文方法复刻前先搜官方代码** |
| 2 | **子集是有偏的** | `--max-train` 取列表前 N 条，而 CSV 里同translator 样本连续 ⇒ 前 2000 条只覆盖 **5/12 个 translator**，与全量分布 TVD = **0.595** | **任何取前 N 条的子集实验必须先报 translator 覆盖率** |
| 3 | **「单点 best」骗人 3 次** | P83 hflip / P85 词级头 / P89 LS，三次都靠 best 单点得出「有效」结论，逐 epoch 配对后**符号全部翻转** | **A/B 必须报逐 epoch 配对差+ 均值±std + 胜率 + 符号稳定性** |
| 4 | **AMP fp16 会NaN** | 13倍「加速」是空转：梯度范数 nan、权重变化小 6 倍。根因 CTC + SeqKD(KLDivLoss) 在 fp16 下溢出 | **要用 AMP 必须用 bf16，且每次验梯度范数** |
| 5 | **冻结主干不省显存** | 可训参数 35.25M → 24.07M（省 32%），但显存 9.04 → 9.04 GB **一点没省**（激活值是大头） | **显存要看激活，不只看参数** |
| 6 | **「分母相同 = 难度相同」** | 300/500/3517 词表分母都是 2843，但 `<unk>` 占比 26.06% / 19.45% / **0%** | **必须区分「口径」与「难度」** |
| 7 | **常识性数字最危险** | 「中文手语单 gloss 0.5–1.5 秒」「30fps」—— 论文 Table I 写 CE-CSL 的 FPS 是 **`varying`**，全文 `fps` 只出现 1 次 | **常识性数字看起来太合理，最容易被当成事实** |

---

## 3. 官方代码核对结果（6 处我写错的地方）

代码位置：`external/TFNet/`（克隆自 `github.com/woshisad159/TFNet`，作者 = CE-CSL 论文作者）

| 项| 我原先写的 | 官方真实 |
|---|---|---|
| 频域分支 | `rfft(x, dim=1)` 沿**时间轴** | `fft(x, dim=-1).abs()` 沿**特征维**，长度天然不变 |
| 主干 | `Linear(368→256)` | `ResNet34MAM`（ImageNet 2D 权重 `unsqueeze(2)` → 3D Conv3d + 逐层 MotorAttention） |
| 序列提取器 | 单层 `Conv1d(k=3)` | `[Conv1d(k=5,pad=0)+BN+ReLU+MaxPool(2)] × 2` ⇒ **48 帧 → 22 → 9** |
| 输出层 | `nn.Linear` | `NormLinear`（权重按输出通道 L2 归一化、无 bias） |
| 辅助损失 | 自创VAE（KL + MSE） | `SeqKD(T=8)` 序列知识蒸馏 |
| CTC | `zero_infinity=False, reduction='mean'` | `zero_infinity=True, reduction='none'` ← **直接解释了 P72 从 ep22 起永久 nan** |

**官方 TFNet 完整规格**：`hidden=1024, lr=1e-4, wd=1e-4, batch=2, 55 epoch`，
lr 在 35/45 降 80%，增强 = `RandomCrop(224) + RandomHorizontalFlip(0.5) + TemporalRescale(0.2)`，
归一化 `x/127.5 - 1`，`collate_fn` 两端各补 6 帧，解码 `ctcdecode` beam_width=10。

---

## 4. 关键实验结果

### 4.1 基线（全部官方口径WER = token 级语料级）

| 实验 | 配置 | 训练量 | translator | **WER%** | 收据 |
|---|---|---|---|---|---|
| P82 | VAC / 224res / 关镜像 | 600 | **2/12** | 89.06 | `p82-vac-noflip-*.json` |
| P83 | 同上 + 开镜像 | 600 | 2/12 | 90.71 | `p82-vac-hflip-*.json` |
| P85 | 同上 + 词级辅助头 | 600 | 2/12 | 90.92 | `p85-wordhead-*.json` |
| P87 | VAC / 160res | 2000 | 5/12 | 80.96 | `p87-res160-2k-*.json` |
| **P91** | **VAC / 160res / 全词表** | **4973** | **12/12** | **72.94** | `p91-full-4973-*.json` |
| P92 | VAC / **224res + bf16** | 4973 | 12/12 | 见收据 | `p92-full-224bf16-*.json` |
| 官方 VAC | 224res / 55ep / beam10 | 4973 | 12/12 | **45.1** | 论文 Table VII |

### 4.2 训练量是当前最有效的方向（三个点连成曲线）

| 训练量 | translator 覆盖 | WER% |
|---|---|---|
| 600 | 2/12 | 89.06 |
| 2000 | 5/12 | 80.96 |
| **4973** | **12/12** | **72.94** |

P91 逐 epoch：`80.23 → 80.08 → 76.18 → 75.19 → 74.03 → 72.94`
**最后两轮仍降 1.09pp，完全未饱和。**

### 4.3 已完成但判为「噪声内 / 无显著影响」的方向

| 方向 | 论文依据 | 结果 | 判定依据 |
|---|---|---|---|
| **RandomHorizontalFlip 开关** | 【有】官方默认 0.5 | −1.43 ± 1.19pp | 符号翻转，噪声内。**不足以判定中文手语镜像语义是否相反** |
| **词级辅助分类头（C 方向）** | 【间接】 | −1.55 ± 1.35pp；且主任务 CTC loss **+4.90%** | 符号翻转。词级头在**争夺主干容量**而非提供有益辅助 |
| **label smoothing（β=0.05/0.1）** | 【无】| 符号翻转，噪声内。但 distinct 20→23/27（**输出更多样**） | β=0.2 **有害**（+1.24pp），平滑过头 |
| **beam search（w=3/5/10/25/50）** | 【有】官方用 beam=10 | **全部劣于 greedy**（+1.73 ~ +2.64pp） | 根因：模型 max prob 0.8959 接近 one-hot，**blank 占 argmax 的 95.2%** ⇒ beam 一致让输出更短 ⇒ 增加删除错误 |

### 4.4 工程约束与解法（8GB 卡的硬限制）

| 配置 | 峰值显存 | 结论 |
|---|---|---|
| VAC T=182 B=2 @224res fp32 | 9.04 GB | ✗ 超卡 |
| **VAC T=182 B=2 @160res fp32** | **4.98 GB** | ✅ 我们的工作点 |
| VAC T=182 B=2 @224res **bf16** | 5.46 GB | ✅ P92 用它 |

**扫描出的显存斜率**（MB每帧·样本）：TFNet 55.7 / CorrNet 45.8 / **VAC 27.1**

**三个被实测推翻的「省显存」手段**：
- ❌ **AMP fp16**：13 倍加速是空转（梯度 nan）
- ✅ **AMP bf16**：梯度正常
- ❌ **冻结主干**：省参数不省显存（激活是大头）
- ✅ **降分辨率 224→160**：显存 ÷2.8，且**不砍帧数**（保留时间结构）

**⚠️ 降分辨率的隐性代价**（P92 正在验证）：
```
输入      avgpool 前空间尺寸
224(官方)   7×7
160(我们)   5×5    ⇒ 空间信息少 1.96倍
             receptive field 相对放大 1.40×
```

---

## 5. 与官方的差距构成（逐行核对代码后得出）

| 来源 | 状态 |
|---|---|
| 架构 / 损失函数 / 优化器 / lr 调度 / 词表 / ImageNet 预训练 | ✅ **完全一致**（直接用官方 `Module.py`） |
| **训练轮数6 vs 55** | 🔴 **主因，差 9 倍**，曲线未饱和可证 |
| 分辨率 160 vs 224 | ⚠️ 次因，机制已找到，量化见 P92 |
| 解码 greedy vs beam10 | ✅ **我们不吃亏**（beam 反而差 1.90pp） |

---

## 6. 词表口径铁律（用户 2026-10-07 定调）

> **300 词表的 WER 低不代表可读性高。坚持用全词表（3515/3517）。**

**依据**：

| 词表 N | 分母 | 参考里 `<unk>` 占比 | 整句可表达 |
|---|---|---|---|
| 300 | 2843 | **26.06%** | 13.8% |
| 500 | 2843 | 19.45% | 26.4% |
| 3516 | 2843 | **0.00%** | 100% |

**官方证据**：`DataProcessMoudle.py::Word2Id` 的 CE-CSL 分支
**无任何截断**（`set()` 去重 + 字典序），全仓库 grep
`max_words|max_tokens|min_freq|frequency` → **0 个截断参数**。
Table VII 全部成绩都在 3515 全词表下取得。

**送分机制已完整验证**（非推测）：
- 300 词表时 dev 有 **26.06%** 的位置参考是 `<unk>`
- `<unk>` 在词表索引 0，但 CTC 有 `+1` 偏移（`dataset.py:226`）⇒ 映射到**类 1**，不撞 blank(类 0)
- train target 里 `<unk>` 占 **28.31%** ⇒ 模型完全能学会输出它
- ⇒ 300 词表下这26.06% 是**真实的送分**

**⇒ 历史结果（P40/P42/P35）的定位**：P42 的 50.00% 是**小词表 + 送分红利**，
不能作为可读性证据，也**不能与官方 45.1% 对标**。

---

## 7. 文献工作

| 编号 | 论文 | 状态 | 用途 |
|---|---|---|---|
| ref24 | **CorrNet**（Hu et al., CVPR 2023） | ✅ 已下载 | 原文明确批评「process frames independently」⇒ **支持我们转向帧间建模**。官方代码 `hulianyuyy/CorrNet` |
| ref25 | **AdaSize**（Pattern Recognition 2024） | ⛔ **拿不到全文** | ScienceDirect 反爬 / arXiv 无预印本 / S2 CLOSED。已删除误下载的 HTML 伪 PDF |
| — | 官方 CE-CSL 论文（arXiv 2409.11960） | ✅ 本地 PDF | 逐字核查；`fps` 全文仅 1 次且写 `varying` |

⚠️ **AdaSize 证明的是「空间冗余」（112×112 vs 256×256），不是「时间抽帧」**
⇒ **不能作为「抽稀到 48 帧不掉点」的依据**（该主张至今无文献支撑）。

---

## 8. 待办（按优先级）

| 优先级 | 事项 | 依据 | 预计 |
|---|---|---|---|
| **1** | **加训练轮数**（P91 6→55 epoch） | 曲线未饱和 | 6vs 55 差 9倍 |
| 2 | 逐级升级模型：VAC → CorrNet → MAM-FSD → TFNet | Table VII 排名 | 每一级先测显存 |
| 3 | 重测词级辅助头（C 方向） | 待输出长度 > 3 token/句 后才有意义 | P91 已接近 |
| 4 | 自实现 CTC 长度惩罚（Graves 2006 公式，60 行 numpy） | 【有】CTC 原论文 | ctcdecode 装不上，pyctcdecode 不适配词级 |

---

## 9. 资产清单

| 路径 | 内容 |
|---|---|
| `external/TFNet/` | 官方代码 clone（仅 CE-CSL 相关） |
| `docs/planning/OFFICIAL_TFNET_SPEC.md` | 官方规格核对报告（12 节） |
| `docs/planning/ROADMAP_2026-10-06.md` | 研究路线图（8 方向 × 依据等级） |
| `docs/planning/LIT_PLAN_论文逐篇对照.md` | 文献逐篇对照 |
| `artifacts/official_rgb/` | RGB 帧 26GB（930,841 帧，**已 gitignore**） |
| `tools/blank_gov/p77_extract_rgb.py` | RGB 提取（原子写 + `.done` 断点续跑 + assert 拦 test） |
| `tools/blank_gov/p78_train_official_tfnet.py` | 官方模型训练脚本（支持 5 种 moduleChoice） |
| `tools/blank_gov/p83_hflip_ablation.py` | hflip 消融（**配对判据已修正**） |
| `tools/blank_gov/p87_accept_check.py` | 输出模式验收（判据写死在注释里） |
| `tools/blank_gov/p88_beam_ablation.py` | beam vs greedy |
| `tools/blank_gov/p89_label_smoothing.py` | label smoothing |
| `tools/blank_gov/official_wer.py` | 官方口径 WER 评估器 |
