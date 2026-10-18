# P148–P151：解码侧收尾 & 🔴 通道顺序错位（R/B 互换）实锤

日期：2026-10-18　类型：纯评估（P148/P149/P150/P151）+ 已启动重训

---

## 1. P148：beam + length penalty ⇒ **判死（单调有害）**

依据【有】ref15 Camgoz, Sign Language Transformers (CVPR 2020) p.6 原文：
"At inference time, we utilize beam search decoding with widths ranging from 0 to 10.
We also implement a length penalty [74] with alpha values ranging from 0 to 2."

动机：P146 实测 p_blank=0.875、生成率 77.2% ⇒ 猜测"欠生成"是主矛盾，
长度惩罚正好对症。

| alpha | 0 | 0.2 | 0.5 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| WER (bias0, beam10) | 55.67 | 56.02 | 56.44 | 57.25 | 59.18 | **61.93** |
| 生成率 | 78.1% | 80.3% | 84.1% | 90.4% | 95.7% | **102.0%** |

🔑 **反证**：把生成率硬拉到 102%（已超过参考长度）WER 反而崩到 61.93
⇒ **欠生成不是"输出太短"，是模型不知道该输出什么**
⇒ 坐实 sub 54.4% 才是本质，与 P133「del 全消除也只降到 ~32%」一致。
⇒ 长度惩罚判死；解码侧（greedy/beam/bias/长度惩罚）至此基本关闭。

---

## 2. P149：抽帧链路端到端核验（帧数 ✅ / 像素 ❌）

官方 `CE-CSLDataPreProcess.py`：`imageio.get_reader()` → `get_data(i)` →
`cvtColor(BGR2RGB)` → `resize(256)` → `imencode(.jpg)`。
我们的 `p77`：`cv2.VideoCapture` → `cvtColor(BGR2RGB)` → `resize(256)` → `imencode(.jpg)`。

| 检查 | 结果 |
|---|---|
| 帧数（我们落盘 vs 视频元数据，30 样本） | **差异 0，30/30 全等** ✅ |
| 首帧像素 MAE（训练读取 vs 视频直解） | **10.66** ⚠️ |
| 逐通道 MAE | **R 15.90 / G 1.04 / B 16.28** |

⇒ G 通道完全对得上、R 与 B 完全对不上 ⇒ **R/B 互换**。

---

## 3. P150：通道 A/B（同一 ckpt，只换输入顺序）

| 输入 | bias0 | bias0.75 |
|---|---|---|
| BGR（现状） | 55.84 | 54.71 |
| 真 RGB | **87.12** | 87.02 |

**Δ = +31.28 pp** ⇒ 通道顺序对该模型是**极其实质性**的
（resnet18 是 ImageNet 预训练，第一层卷积对通道语义敏感）。

---

## 4. P151：决定性验证 —— 官方到底是 RGB 还是 BGR

完整复现官方链路（照抄源码，不依赖任何假设），3 个样本结果一致：

| 环节 | 判定 | 证据（MAE） |
|---|---|---|
| `imageio.get_data()` 输出顺序 | **RGB** | vs 真RGB **0.00** / vs 互换 9.90 |
| **官方完整链路最终** | **真 RGB** | vs 真RGB **1.42** / vs 互换 10.16 |
| **我们完整链路最终** | **BGR（互换）** | vs 真RGB 10.30 / vs 互换 **1.41** |

帧数也一致（236 / 309 / 207，cv2 元数据与 imageio.count_frames() 完全相同）。

### 根因

官方抽帧用 `imageio`（**输出 RGB**），却同样做了一次 `cvtColor(BGR2RGB)`
（即"多"交换一次）；我们用 `cv2.VideoCapture`（**输出 BGR**）也做一次。
两边在"抽帧后内存中的数组"就差一次交换，后续链路相同 ⇒ **最终结果差一次交换**。

```
官方：RGB →(cvtColor 交换)→ BGR →(imencode 交换)→ 磁盘真RGB →(imread)→ →(cvtColor 交换)→ RGB ✅
我们：BGR →(cvtColor 交换)→ RGB →(imencode 交换)→ 磁盘R/B颠倒 →(imread)→ →(cvtColor 交换)→ BGR ❌
```

⇒ **我们一直在用 BGR 训练与评估，官方是 RGB。**
这是 10.74pp 缺口里**第一个被实锤的具体来源**。

---

## 5. 已启动：P150 组重训（tag `p150-rgbfix`）

唯一变量 `--fix-rgb-order`（读图后不再 `cvtColor` ⇒ 得到真 RGB），
其余参数与 P123 逐项一致（VAC / 160 / bf16 / batch2 / 55 轮 / lr1e-4 / wd1e-4 /
seed42 / official-pad / word-head 0.3 / warmup 3 / 里程碑 [35,45] γ=0.2）。

**判据（启动前写死）**
- 主指标：dev WER（official 口径，greedy，bias=0）
- 基线：P123 = 55.84
- 改善 ≥ **1.556 pp**（噪声基准）才算可判有效
- epoch 级配对只用 **ep32+**（P123 的 ep1–31 是坏解码截断的虚高值）

---

## 6. 如果成立，意味着什么

- 之前所有实验（P136–P142 超参、P144 换 seed）都是在**错位的通道**上做的
  ⇒ 那些"负结果"不因为通道而失效（同条件对照），但**绝对值 55.84 是被拖低的**
- 官方 45.1 与我们的 55.84 之间的 10.74pp，**至少有一部分**应归因于此
- ⇒ 修好后需要重跑关键对照（尤其是与官方的差距还剩多少）

## 7. 待办

- [ ] 等 55 轮跑完（约 24h）看 dev 曲线
- [ ] 若改善 ⇒ 在 test 上复核一次（需与用户确认，test 已用一次）
- [ ] 无论结果如何，通道顺序都应固化到 `p77` 抽帧侧（使磁盘 jpg 与官方一致），
      而不是只靠读图开关兜底
