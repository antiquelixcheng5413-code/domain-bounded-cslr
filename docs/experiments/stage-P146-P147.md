# P146 + P147：后验校准诊断 & 官方解码口径复核

日期：2026-10-18　类型：**纯评估，零训练**（dev 515，P123 best ckpt，epoch 39）

---

## 1. 为什么要做这两个

超参/正则化主线（P136–P142）全部为负后，剩下两个此前**只有现象、没有机理**的疑点：

1. 误差里 sub 占 54.4%、del 占 43.2%，生成率只有 77.2% —— 模型到底"学到"了什么？
2. 我们全程用 greedy 解码，而官方 dev 42.1 / test 41.9 是怎么解出来的？

---

## 2. P146：CTC 后验校准 / 峰锐度

脚本 `tools/blank_gov/p146_calibration_probe.py`（dev 515 + train 随机 300，seed 42）

**已知答案校验**：dev 全量 greedy bias=0 = **55.84** ✅（与训练时完全一致）
⇒ 代码路径正确；bias=0.75 → 54.75（−1.09），与 P134 一致，又一处复现。

### 帧级后验（bias=0 原始输出）

| split | p_blank | pnb_top1 | margin | entropy | **峰锐度** | frac>0.2 | 峰/样本 |
|---|---|---|---|---|---|---|---|
| dev | 0.8750 | 0.0649 | 0.0579 | 0.492 | **0.564** | 8.4% | 4.3 |
| train | 0.8613 | 0.0748 | 0.0666 | 0.521 | **0.598** | 9.7% | 4.4 |

- **峰锐度** = greedy 折叠后，每个输出 token 所对应「峰」内的最大非 blank 概率
- **p_blank = 0.875**：平均 87.5% 的概率质量压在 blank 上 ⇒ 模型**过度保守**

### 三条结论

1. 🔑 **过拟合不是主因**：train 与 dev 的峰锐度只差 0.034、frac>0.2 只差 1.3pp。
   模型在**训练集上**峰就不尖（0.598），不是"dev 上变钝"。
   ⇒ 与「train CTC 5.5 vs dev 55.84」的旧判读相反，抗过拟合主线（wd/lr/LS）判负是自洽的。
2. 主矛盾是**欠生成**：p_blank 0.875 解释了 del 43.2% / 生成率 77.2%，
   也解释了为什么 blank bias 有效（−1.09 dev / −0.69 test）。
3. ⚠️ **推翻旧解释**：此前用"后验不尖"解释 beam 崩溃，实测「实心峰/实际输出 = 0.60 < 1」
   ⇒ 站不住，需另找原因（见 P147）。

---

## 3. P147：官方解码口径是 **beam**，不是 greedy 🔴

### 源码核对（此前记错了）

| 位置 | 内容 |
|---|---|
| `external/TFNet/Train.py:152` | `decoder = decode.Decode(word2idx, wordSetNum + 1, beam)` |
| `external/TFNet/decode.py:24` | `ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=0, num_processes=10)` |
| `Train.py:282` / `Train.py:384` | 训练中与**验证时**都调 `decoder.decode(...)` |

⇒ **官方 42.1 / 41.9 是 beam_width=10 解出来的**。我们全程 greedy ⇒ 真实口径差异。
⚠️ 旧记忆写「官方用 greedy」，**已更正**。README 还要求 `ctcdecode==0.4`。

`ctcdecode==0.4` 在本环境装不上（源里只有 1.0.1/1.0.2，需编译）
⇒ P147 自己实现标准 CTC prefix beam search（Graves 2006），每帧 top-40
（对齐 ctcdecode 默认 `cutoff_top_n=40`），**无 LM / 无词典**（官方 decode.py 也没传）。

### 结果（dev 515，P123 best）

| 口径 | bias=0 | bias=0.75 |
|---|---|---|
| greedy（我们） | 55.84 | **54.71** |
| beam-10（官方） | 55.67 | 55.42 |

- **beam vs greedy（bias=0）：Δ = −0.17 pp** ⇒ ⚪ **在噪声内，不可判**
  （噪声基准 1.556pp；换 seed 口径 0.838pp）
- beam 下 bias 仍有收益（−0.25），但**不如 greedy + bias（−1.13）**
  ⇒ **当前最佳仍是 greedy + bias 0.75 ≈ 54.7%**

### 结论

🔴 **解码口径不是 10.74pp 缺口的来源**。对齐到官方 beam 后我们仍是 55.67。
同时更正一条旧结论：此前「beam(10) 让输出 +109%」是在早期 WER 61.86 的**差模型**上测的；
在 P123（55.84）上 beam 与 greedy 基本等价 —— 既不是灾难，也不是收益。

---

## 4. 净产出

- 关闭一个悬案：**解码口径已对齐，缺口不在这里**
- 新增一个可操作方向：**峰锐度 / blank 占比**（train 上也只有 0.598 ⇒ 欠生成是训练侧问题，
  不是泛化问题）⇒ 若继续改进，应针对**让模型敢输出**，而不是继续加正则
- 三个新指标可复用：`p_blank`、`peak_sharp`、`frac_gt02`

## 5. 未解决

- 10.74pp 缺口仍无解释。唯一能定论的对照仍是**官方 ckpt 权重**
  （https://pan.baidu.com/s/1KPbDL2nAvBsSsTDwc9og0A 提取码 0000；
  ⚠️ README 自相矛盾：下载项写 test 32.46%，Inference 表写 dev 42.1 / test 41.9）
- 若跑官方 ckpt，需同时记录它的 train-CTC 与这三项后验指标，才能判断"我们的模型到底差在哪一层"
