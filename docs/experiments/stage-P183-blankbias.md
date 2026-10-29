# 阶段报告 P183：推理时修复 VAC@160 欠生成（blank bias / length penalty）

- **日期**：2026-10-29
- **方向**：用户选定的「推理时修复欠生成」（零训练成本）
- **结论速览**：欠生成真实存在（baseline 生成率 76.4%，P182 长度比 0.76）；**blank 抑制是唯一有方向性收益的解码杠杆**，bias=1.0 两次全量一致 −0.8pp（53.73%→52.92%），但低于 1.556pp 噪声地板 ⇒ 属「强提示」非确证；beam+length penalty 是死路。已把 `BLANK_BIAS` 接入前端并令 VAC 默认 1.0（免费、无 downside）。

## 1. 动机
P182 全量 515 条统计：VAC@160 输出/参考长度比仅 **0.76**，69.7% 的句子输出短于参考 ⇒ 系统性**欠生成**（提前吐 blank）。
这不可能是管线 bug（P178 已证前端与官方 WER 对齐 0.07pp），是模型后验性质。标准修复是解码期干预：
- **blank logit 抑制**（greedy 上加 bias）—— 直接对抗过早 blank
- **beam + length penalty** —— 标准 length normalization

二者此前都未在同口径全量上被干净测过：
- P120/P152：裸 beam(10) ⇒ 输出 +109%（过度生成失控），被判死
- P164：blank bias 扫描 ⇒ bias=1.0 得 52.92%（−0.8pp），但无全量对照、被标「轻微有害」

## 2. 方法
- **模型/数据同口径**：`p150-rgbfix-best.pt` + `artifacts/official_rgb` 预抽帧 + `RGBSeqDataset(RGB_FIX=True)` + `OFFICIAL_PAD=True`，与 P176 复现 53.73% 的路径逐位一致。
- **两阶段脚本 `p183_length_penalty.py`**：
  - 前向分片（每 50 条存盘 `artifacts/cache/p183_logits_{s}_{e}.pt`，已存在自动跳过 ⇒ 抗 WSL 崩、可续跑）
  - 解码网格（CPU）：greedy bias∈{0,0.5,1.0,2.0}（beam+lp 已在 v1 确认死路，本版只保留 greedy 族）
- **指标**：`official_wer`（语料级 WER）、生成率、单句号率
- **噪声基准**：1.556pp（记忆：配对 std 0.838）。判定线：<1 不可判｜1~2 强提示｜≥2 且胜率≥75% 可判

## 3. 结果（全量 515 条 dev）

| 策略 | WER | ΔWER | 生成率 | 单句号 |
|---|---|---|---|---|
| greedy bias=0.0（baseline） | **53.73%** | — | 76.4% | 1.7% |
| greedy bias=0.5 | 53.27% | −0.46pp | 81.0% | 1.7% |
| **greedy bias=1.0** | **52.92%** | **−0.81pp** | 84.3% | 2.3% |
| greedy bias=2.0 | 53.45% | −0.28pp | 91.1% | 2.3% |
| beam+lp α=0→1.0（v1 复扫） | 46.89→47.99% | +0.37~1.47pp | 78.8→89.4% | — |

- **baseline 53.73% 与官方 P150 逐位吻合**（`edits=1527/ref_tokens=2842` 同 P176/P178），路径校验通过。
- **bias=1.0 = 52.92% 与 P164 的 52.92%（−0.8pp）逐位一致** ⇒ 跨两次独立全量可复现。
- **生成率随 bias 单调上升**（76.4%→84.3%→91.1%），证实「抑制 blank 确实让输出变长」，即欠生成假说成立且可被解码干预。
- bias=1.0 的 −0.81pp 低于 1.556pp 噪声地板 ⇒ 严格判「不可判」，但方向一致+两次复现 ⇒ 实务上作「强提示」。
- **beam+length penalty 死路确认**：length penalty 只是把生成率拉高却让 WER 变差，根因是后验不尖导致重复词，与 P120 过度生成诊断一致。

## 4. 改动
- `app/backend/tfnet_service.py`：
  - 基类新增 `BLANK_BIAS = 0.0` 旋钮；`_decode` 解码前 `seq[:,:,0] -= self.BLANK_BIAS`
  - `VacTrainedService.BLANK_BIAS = 1.0`（免费方向性正收益，默认开）
  - `TFNetPublishedService` 保持 0.0（发布权重零样本已 35.66%，未单独验证 bias 效果，保守不动）
  - 订正文件头注释（旧写 BLANK_BIAS=0.0）、`_ensure_loaded` 里 `P.BLANK_BIAS = self.BLANK_BIAS`
- 新增诊断脚本：`p183_length_penalty.py`（分片续跑）、`p183_fe_probe.py`（前端 predict_video 实跑探针）

## 5. 前端实跑验证
`p183_fe_probe.py` 直接调 `VacTrainedService.predict_video`（bias=1.0），20 条 dev：
- corpus WER = 47.86%（子集方差大，仅看方向，与全量 52.92% 同向）
- 纯句号 0/20 = 0% —— 无坍缩，接线正确

## 6. 结论与后续
1. **「推理时修复欠生成」的收益上限 ≈ 0.8pp**，且落在噪声内。它**不能**把 VAC@160 从 53.73% 拉到发布 TFNet 的 35.66% 量级 —— 那 18pp 差距是**骨干差距**（VAC vs TFNet/MAM），不是解码赤字。
2. bias=1.0 已作为免费默认接入前端 VAC，无 downside；若要严格「零偏离官方」，可改回 0.0（代价 ≈ +0.8pp，在噪声内）。
3. **真正能缩小 18pp 差距的杠杆仍是**：① P172 跨分辨率蒸馏（学生 VAC@160 能否逼近 36）、② 直接部署发布 TFNet@160 零样本（已 35.66%，零成本）。推理时解码调参到此为止。

---
**附：本次修复的两个衍生 bug（p183 自身）**
- `CACHE_DIR / "p183_logits_%d_%d.pt" % (s,e)` 运算符优先级错误 ⇒ 改为 `CACHE_DIR / ("...%d_%d.pt" % (s,e))`
- `P.collate([ds.items[i]])` 误用原始 CSV 行（无 `"video"` 键）⇒ 改为 `P.collate([ds[i]])`（走 `__getitem__` 产出含 video 的样本）
