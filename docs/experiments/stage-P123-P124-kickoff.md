# P123 / P124 启动与判据预案

日期：2026-10-11 · 状态：**已自动启动**（链条 `tools/blank_gov/p123_124_chain.py`）
前置：P121（`p121-disamb`）仍在跑，预计 10-11 16:40 左右出收据，链条自动接上。

---

## 1. 合理性评估结论

| 提案 | 结论 | 理由（一句话） |
|---|---|---|
| **补官方其余超参** | ✅ 合理，**已执行** | 依据【有】（官方 `Train.py` / `DataProcessMoudle.py` 源码逐行核对）；零成本项立即补，55 轮是唯一昂贵项，价值在**可比性** |
| **自监督预训练** | ⚠️ **降级为「先探针后决定」**，已执行探针层 | 依据最高只到【间接】（ref12 CCL-SLR 只在孤立词 ISLR 验证过），且存在【有】级**反向证据**（官方 CorrNet 47.2 > VAC 45.1）；直接投 9~12 小时不划算 |

---

## 2. 官方超参逐项对照（源码核对，非论文转述）

| 项 | 官方值 | 我们此前 | 现在 | 证据 |
|---|---|---|---|---|
| optimizer | Adam(lr=1e-4, wd=1e-4) | 同 | ✅ 一致 | `Train.py:121` |
| epochs | **55** | 20（P119 误跑 44） | ✅ 55 | `Train.py:157` |
| lr 调度 | MultiStepLR([35,45], γ=0.2) | 按总轮数缩放到 [13,16] | ✅ [35,45] | `Train.py:147-149` |
| batch / hidden | 2 / 1024 | 同 | ✅ | `params/config.ini` |
| CTC | blank=0, reduction='none', zero_infinity=True, `.mean()` | 同 | ✅ | `Train.py:110/201` |
| SeqKD | **×25**（T=8, use_blank=False） | 同（代码里已是 25.0） | ✅ | `Train.py:196` |
| collate 补帧 | **left_pad=6** + 末帧复制，dataLen=ceil(T/4)\*4+12 | 零填充 + dataLen=真实帧数 | ✅ 新增 `--official-pad` | `DataProcessMoudle.py:435-467` |
| 主干（VAC） | **2D ResNet18**，逐帧前向 | 同（一直没搞错） | ✅ | `Net.py:88` + `Net.py:331` |
| 分辨率 | 256 存盘 → RandomCrop(224) | 160 | ⚠️ **保留 160** | P92 实测 160 比 224 好 5.60 pp |
| AMP | GradScaler fp16 | fp32（P119）/bf16（P121） | ⚠️ **P123 用 fp32**（与 P119 严格可比） | fp16 在 CTC+KLD 下 NaN（P87 实测） |
| 解码 | ctcdecode beam_width=10 | greedy | ⚠️ **保留 greedy** | P120 实测 beam(10) 使 WER 61.86→125.58 |
| numWorkers/pinMemory | 8 / 1 | 自定义 loader，不适用 | — | 不影响 |

**改动落点**：`tools/blank_gov/p78_train_official_tfnet.py`
- 新增 `OFFICIAL_PAD` 全局 + `collate` 官方分支；新增 `--official-pad`、`--backbone-ckpt`
- 契约单测 `tools/blank_gov/test_official_pad.py`：**17/17 通过**（含旧口径回归保护）
  - ⚠️ 第一版单测判据写错了（把 `vid.shape[0]` 当时间维、把输入顺序当排序后顺序），
    修正后代码本身是对的 —— 再次印证 MEMORY 0.3「体检脚本必须先用已知答案验证判据」

---

## 3. P123：官方全套超参（55 轮）

```
--tag p123-official55  --module VAC  --img-size 160
--epochs 55 --total-epochs 55        # ⇒ milestones [35,45], γ=0.2
--word-head 0.3 --word-boundary ctc_align --warmup-ep 3
--official-pad  --mem-budget 5.2e9  --seed 42      # fp32（与 P119 同）
```

**与 P119（当前最佳 56.72%，ep33）的差别只有两处**：轮数/里程碑、官方补帧。

### ⚠️ 必须预先写明的混淆与预期

1. **混淆**：轮数与补帧是**捆绑**改的。若 P123 显著变好/变坏，无法直接归因。
   → 后续隔离方案（9 h）：`--epochs 20 --total-epochs 20 --official-pad`，
   与 P119 ep13–20 逐轮配对。
2. **预期收益不确定**：P119 在 ep33–44 已平台（56.72~57.60，波动 0.9 pp < 噪声 1.556 pp），
   且它当时 lr 已在 ep13/16 衰减到 4e-6。官方调度要到 ep35 才第一次衰减，轨迹不同 ——
   可能更好（35 轮满 lr 优化更充分），也可能更差（更晚衰减=更多过拟合，
   P119 训练 CTC 从 11.98 一路降到 10.43 而 dev 不动 ⇒ 已在过拟合区）。
   **粗估：1~3 pp，但完全可能落在噪声内。**
3. **真正的价值**：这是第一个「与官方同调度」的数字，才能把
   「我方 5x.x vs 官方 VAC 45.1」这句话说得有底气。

---

## 4. P124：自监督预训练 —— 为什么降级

### 证据等级表

| 主张 | 等级 | 出处 |
|---|---|---|
| VAC 主干是 2D ResNet18、逐帧前向 ⇒ torchvision resnet18 权重可直接替换 | 【有】 | `Net.py:88` / `Net.py:331` |
| 自监督预训练能提升**手语识别** | 【有】 | ref12 CCL-SLR 原文："Pre-training has been proven to be effective in boosting the performance of **Isolated** Sign Language Recognition (ISLR)" |
| SignBERT（Hu ICCV'21）同类方法 | 【有】 | ref24 CorrNet 参考文献第 841 行 |
| 自监督预训练能降 **CSLR / CE-CSL 的 WER** | 【**无**】 | 全库 grep 无支撑 |
| 反向证据：官方 CorrNet（核心即自监督一致性约束）在 CE-CSL 上 47.2，**比 VAC 45.1 差 2.1 pp** | 【有】 | 官方论文 Table |
| 域内无标注数据能改善低频类 | 【间接】 | 对比学习在低资源/少样本上的通用结论，非手语专用 |

### 设计（已实现）

- **P124a**：SimCLR（NT-Xent, T=0.2）在 CE-CSL **自己的** RGB 帧上预训练 2D resnet18
  （12 帧/视频 ≈ 60k 帧，batch 128，40 轮，≈ 5 h）
  - 不用任何 gloss 标签 ⇒ 无 label leakage
  - 归一化 /127.5−1、160 res 与下游一致；hflip 作为不变性（P119 已证 hflip 有效 −3.42 pp）
  - **刻意不用大比例随机裁剪**（scale 0.5~1.0）：手相对身体的位置是有意义信息，
    过强裁剪不变性可能抹掉判别信息（依据【无】，但方向保守）
- **P124b 探针**（≈ 20 min）：Cohen's d + 高频子集 macroAUC，与 P107B2 同口径

### 判据（**预先写死**，防作弊）

| 指标 | 随机基线 | 判活 | 判死 |
|---|---|---|---|
| 主：ΔCohen's d = d(SimCLR) − d(ImageNet) | 0 | Δd ≥ **+0.05** 且 bootstrap 95% CI 下界 > 0 | Δd ≤ +0.01 或 CI 上界 < +0.05 |
| 辅：高频子集 macroAUC | **0.5**（多分类 OVR） | — | — |

- bootstrap：按**视频聚类**重抽样 200 次（同视频片段不独立）
- ⚠️ 修正 P107B2 脚本的阈值错误：它写的是 `dead_auc=0.10 / alive_auc=0.25`，
  **低于随机基线 0.5**，任何像样特征都必然通过（MEMORY 0.6g）。本脚本已修正。
- 灰区 ⇒ 按判死处理（不投 9 h）

### 只有判活才做 P124c

`--epochs 20 --total-epochs 20 --backbone-ckpt p124a-simclr-final.pt`（其余= P119 配置）
⇒ 与 **P119 的 ep13–20 逐轮配对**，按项目噪声纪律出 `mean±std / 胜率 / SNR`
（噪声基准 std = 1.556 pp，SNR < 1 不可判）。

---

## 5. 时间线（GPU 串行）

| 阶段 | 预计 |
|---|---|
| P121（在跑，20 轮解混淆） | 10-11 ~16:40 |
| P123（55 轮官方调度） | ~16:40 → 10-12 ~17:00（≈ 24 h） |
| P124a（SimCLR 40 轮） | → ~22:00（≈ 5 h） |
| P124b（提取 2×2000 视频 + 探针） | → ~23:00（≈ 1 h） |
| P124c（**仅当判活**，20 轮微调） | → 10-13 ~08:00（≈ 9 h） |

失败处理：任一步 rc≠0，链条记日志并继续下一步（不静默中止）。
