# 阶段报告：P178 前端推理口径修复

- **Experiment ID**: P176 / P177 / P178（同一条诊断链）
- **Date**: 2026-10-28
- **代码来源**: `external/TFNet`（直接 import 官方 `videoAugmentation` / `Net`，未重写）
- **词表**: 官方 Word2Id 3515 类（全收、无截断）
- **收据**: `artifacts/logs_p176_official.log`、`tools/blank_gov/p178_tokens_vac.json`

---

## Research question

前端（`app/backend/tfnet_service.py`）对 dev 视频几乎全部输出单个 `。`，而同一权重在官方评测口径下是 53.73%。
**这是模型本身崩了，还是前端预处理/推理路径有问题？**

---

## Verdict（判答）

**前端 bug。模型无问题。**

三条独立证据：

1. **官方路径逐位复现**：同一份 `p150-rgbfix-best.pt`，走 p78 官方管线（`artifacts/official_rgb` 预抽帧 + `RGBSeqDataset` + `_dev_eval`），dev WER = **53.73%**，连 `edits=1527` / `ref_tokens=2842` / `n_sentences=515` 全部一致。
2. **像素级对撞**：前端 `_load_frames` 与官方 `RGBSeqDataset` 对同一条视频（dev-00001，236 帧）产出的张量，修正后 **MAE=0.015**（在 `[-1,1]` 尺度，≈1.9/255），帧数完全一致。
3. **修完后前端复现官方数字**：前端全量 515 条 dev = **53.66%**，与官方 53.73% 差 **0.07pp**（2842 个 token 里差 1 个编辑）。

---

## 结果

| 指标 | 修复前 | 修复后 | 官方路径（基准） |
|---|---|---|---|
| dev corpus WER（全 515 条） | 84.25%（前 50 条测） | **53.66%** | 53.73% |
| 纯单句号 `。` 占比 | **100%**（50/50） | **1.4%**（7/515） | — |
| 空输出 | — | 10/515 | — |
| 不同输出数（515 条） | 极少 | **494** | — |
| 前 50 条 WER | 84.25% | 44.69% | — |

发布 TFNet 一侧同步验证（前 30 条）：WER **35.09%**，单句号 0/30，与 36.81%（dev 零样本）同量级。

---

## 根因（两个，都是前端 bug）

### bug 1：通道顺序遗漏 BGR→RGB（主因）

- 官方 `RGBSeqDataset`（`RGB_FIX=True`）读 `artifacts/official_rgb` 的 jpg，**不做** `cvtColor`；P77 写盘时已把通道换过一次，两次相抵后等价于 **真 RGB**。
- 前端 `cv2.VideoCapture` 直接给 **BGR**，却原样喂进模型。
- P177 实测（dev-00001，mp4 vs 官方帧，8 帧）：

| 前端喂法 | 与官方帧 MAE |
|---|---|
| mp4 → RGB | **1.42** ✅ |
| mp4 → BGR（旧行为） | 10.32 ❌ |

- 数据近乎灰度（`|R−B|` 中位 8），所以这个 bug 对**训练侧**只值 2.11pp，但在**前端单条推理**上叠加放大后直接把输出打到坍缩。

### bug 2：超大视频先把全部帧读进内存（导致反复"跑着跑着 WSL 崩"）

- `dev-00403.mp4` = **47.7 MB**（正常 ~3 MB，官方抽了 464 帧）。
- 旧实现 `while cap.read(): raw.append(frame)` 全部读完，**之后**才 `np.linspace` 下采样到 256 帧 ⇒ 峰值内存几 GB，实测把 WSL 整个带崩（连续 4 次崩在同一位置，报 `Wsl/Service/E_UNEXPECTED`）。
- 修复：先读 `CAP_PROP_FRAME_COUNT` 算目标帧号，再 `grab()` 跳读、只在目标帧 `retrieve()`，内存恒定在 256 帧；`FRAME_COUNT` 不可信时兜底只留前 256 帧。修复后该视频 **7.4s** 处理完，不再崩。

### 排除项：`/127.5 - 1` **不是** bug

一度怀疑前端那行 `/127.5 - 1` 是重复归一化（怀疑 `ToTensor` 已除 255）。核查后否定：

- `videoAugmentation.ToTensor` 只做 `transpose + .float()`，**不除 255**（P177 实测 ToTensor 后 max=251）。
- 官方 `RGBSeqDataset.__getitem__` **末行就是** `seq = seq.float() / 127.5 - 1.0`。
- 试删该行 ⇒ WER 从 84.25% **恶化到 95.60%**，已回滚。

> 教训：官方 `build_transforms` 里看不到归一化，归一化藏在 Dataset 里。只读 transform 就下结论会判反。

---

## 改动清单

`app/backend/tfnet_service.py`：

| 位置 | 改动 |
|---|---|
| 类属性 | 新增 `CHANNEL_FIX = True`（BGR→RGB）、`PIXEL_SCALE = "pm1"`（保留 `"raw"` 选项） |
| `_load_frames` | 抽帧改 `grab()` 跳读 + 目标帧 `retrieve()`，内存封顶 `MAX_FRAMES` |
| `_load_frames` | `cv2.cvtColor(frame, COLOR_BGR2RGB)`（受 `CHANNEL_FIX` 控制） |
| 注释 | 订正 `# [T,3,H,W] float [0,1]` → `[0,255]`（原注释错误，是本次误判的根源之一） |

---

## 残留问题（不影响主结论）

- 仍有 7/515 条输出纯 `。`、10/515 条空输出。模型判别力本就弱（peak sharpness 0.624 vs 官方 0.864），个别样本坍缩属预期。
- 0.07pp 残差主要来自 `dev-00403`：官方用 464 帧，我们按 `MAX_FRAMES=256` 均匀下采样到 256 帧，该条输出可能与官方不同。
- `MAX_FRAMES=256` 对 CE-CSL 正常样本（≤236 帧）不触发，不影响其余 514 条。

---

## 论文依据

- 通道顺序影响手语识别输入分布：本项目 P149/P123 已有实测（【有】本仓库内部实测，2.11pp）。
- 预处理必须与训练口径逐项对齐（否则分布偏移导致 CTC 坍缩）：【间接】CTC 对输入分布敏感是常识性结论，本报告以像素 MAE + WER 复现直接证明，无单篇论文引用。
- 超大视频流式解码：工程性修复，【无】论文依据。

---

## 复现命令

```bash
# 1) 官方路径基准（应得 53.73%）
./venv/bin/python -u tools/blank_gov/p176_official_path_eval.py

# 2) 像素级对撞（MAE 应为 1.42 = 转 RGB 后正确）
./venv/bin/python -u tools/blank_gov/p177_pixel_ab.py

# 3) 前端全量 dev（分片可续跑，$s 从 0 步进 65 到 515）
P178_SVC=vac P178_START=0 P178_END=65 ./venv/bin/python -u tools/blank_gov/p178_front_verify.py

# 4) 超大视频内存守卫
./venv/bin/python -u tools/blank_gov/p180_bigvideo_test.py 402
```
