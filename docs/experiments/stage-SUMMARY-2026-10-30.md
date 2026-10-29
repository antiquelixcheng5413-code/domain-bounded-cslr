# 项目汇总报告 — 中文手语识别（CE-CSL / CSLR）

**日期**：2026-10-30 ｜ **状态**：✅ 已到实用天花板，工程清理收尾
**数据集**：CE-CSL（4973 训练 / 515 验证 / 测试，3517 类词表，每条 193 帧）
**硬件**：单机 8GB GPU（5060）；队友已退出，串行训练

---

## 0. 一句话结论

**本项目最佳可部署数字 = 零样本发布 TFNet@160（RGB）= test 35.66%（零训练成本）**；
真天花板 = 同权重满分辨率 **test 33.09% / dev 33.92%**。
我们 from-scratch 训练的最佳 VAC@160 = dev **53.73%**（含通道修复 +2.11pp、word_head 0.3 正则、blank bias −0.8pp）。
VAC→TFNet 的 ~20pp 差距是**骨干差距**，便宜杠杆已穷尽，无法靠 VAC 内调参弥补。

---

## 1. 关键结果时间线（WER，越低越好）

| 里程碑 | 配置 | dev / test WER | 备注 |
|---|---|---|---|
| P87（最早） | VAC@160 | 80.96% | 起点，输出坍缩 |
| P150 | VAC@160 + RGB + word_head0.3 | **53.73%**（dev） | from-scratch 最佳（含通道修复） |
| P150 + blank bias | 同上，解码 bias=1.0 | 52.92%（dev） | −0.81pp，噪声地板内强提示 |
| P156 零样本 | 发布 TFNet@160 RGB | dev 36.81% / **test 35.66%** | 零成本，零样本部署最佳 |
| P156 门禁 | 发布 TFNet@224 RGB | dev 33.92% / **test 33.09%** | 真天花板（门禁通过） |
| P171 微调 | 发布 TFNet@160 + 微调 | 39.90%@ep3 → 45%+ | 净负向，判死 |
| P172 蒸馏 | VAC@160 ← TFNet@224 教师 | 77.55%（仅 4 轮） | 判死 |
| P173 帧差 | VAC@160 + 5ch(帧差) | 78.47%（ep03） | 判死，帧差贡献≈0 |

---

## 2. 已排查方向总表

### 2.1 有效（保留）
| 项 | 收益 | 证据 |
|---|---|---|
| 通道修复（BGR→RGB） | **+2.11pp** | D1→P150 对照，强证据【有：cv2 BGR 源】 |
| word_head 0.3 正则 | 去头 +2.15pp（保留头更好） | P167 消融，24/24 胜【有】 |
| blank bias=1.0（VAC 解码） | −0.81pp（噪声内强提示） | P183 全量 515 两次复现【有】 |
| 160≫224 分辨率 | +21pp（160 远优于 224） | D3 ep32+ 配对 SNR≫2【有】 |
| 发布权重零样本直接部署 | test 35.66%（免费） | P156 / P156-test 验收【有】 |

### 2.2 判死（无增益 / 负向）
| 项 | 结果 | 判死依据 |
|---|---|---|
| dropout 0.3 探针 | 55.10% vs 53.73%（轻微有害） | P170 单变量对照 |
| 更长训练（P169） | best 57.28% vs 53.73（过拟合） | 55ep 已是谷底，硬拉长只过拟合 |
| beam + length penalty | +0.37~1.47pp 更差 | P183，后验不尖致重复词 |
| SimCLR 自监督 | 死路 | 审计结论 |
| SWA（过拟合尾巴） | ≈59%（负向） | P169 history 判死 |
| Gloss LM 浅融合 | 生成率崩到 2.9% | P166，数据稀疏 |
| lr/wd 超参搜索 | 1e-4 最优 | 网格已扫 |
| 跨分辨率蒸馏（P172） | KD 仅占梯度 ~3%，跨骨架+跨帧率不匹配 | 被免费零样本 35.66% 压过 |
| 微调发布权重（P171） | 过拟合摧毁预训练，不如零样本 | best 39.9% > 零样本 36.81% |
| 运动线索帧差（P173） | ep03 78.47 与 P150 持平，改善全来自 bias 修正 | 帧差通道贡献≈0 |
| CorrNet 骨干 | 论文硬编码最弱（47.2） | 换骨干补不了 8.6pp 训练赤字 |
| hflip / bf16 / 时序增强 / 小配置回代 | 不可判或无效 | 噪声内 |

---

## 3. 真天花板论证

- 发布权重文件 `TFNet-CE-CSL-CSLDaily-32.46.pth`（879.6MB），结构 = 完整 TFNet（resnet34MAM 骨干 + 时序卷积 + 双向 RNN + 5 个辅助头），`load_state_dict(strict=True)` 逐键精确匹配（P156 验证）。
- 此权重本就在 CE-CSL 上训练（文件名暗示可能联合 CSLDaily），零样本迁移到本项目 dev/test 直接有效：dev 33.92 / test 33.09，与 README 标注 32.46 三口径一致可信。
- **我们自己 from-scratch 训任何骨干都受 8.6pp 训练/数据赤字约束**（VAC 我们 53.73 vs 论文 45.1），即便训 TFNet 也大概率只能到 ~40%+，**追不上、更超不过这条零样本线**。
- ⚠️ 对外宣称「CE-CSL SOTA 级」前，须向作者核实权重是否联合训练（不影响"零样本迁移有效"结论）。

---

## 4. 工程产物索引

### 4.1 Checkpoints（保留）
- `artifacts/checkpoints/p150-rgbfix-best.pt` — VAC@160+RGB from-scratch 最佳（53.73）
- `artifacts/checkpoints/p173-flow-best.pt` — 帧差探针 best（ep02, 78.08，已判死）
- 发布权重：`/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/TFNet-CE-CSL-CSLDaily-32.46.pth`

### 4.2 阶段报告（docs/experiments/）
前端修复：`stage-P178-frontend-preprocess-fix.md`、`stage-frontend-readable-pipeline.md`
解码/欠生成：`stage-P163-p150-posterior.md`、`stage-P164-p150-bias-breakdown.md`、`stage-P183-blankbias.md`
零样本代价分解：`stage-P156-zero-shot-ablation.md`、`stage-P156-test.md`、`stage-P156-D1-official-ablation.md`
方向判死：`stage-P166-gloss-lm.md`、`stage-P169-longtrain.md`、`stage-P170-dropout.md`、`stage-P171-finetune-tfnet160.md`、`stage-P172-distill-crkd.md`、`stage-P173-flow-probe.md`
天花板收尾：`stage-wrapup-ceiling.md`

### 4.3 关键脚本（tools/blank_gov/）
- `p78_train_official_tfnet.py` — 官方 TFNet 训练/评估主脚本（含 VAC/CorrNet 支持、RGB 帧加载、WER 评估）
- 诊断：p175–p183 系列（前端评测、像素/张量对撞、分片全量、长度统计、解码扫描）

---

## 5. 复现关键口径（勿再犯）

- **通道**：cv2 读 mp4 得 BGR，必须 `cvtColor(BGR2RGB)`（官方 RGB_FIX 等价真 RGB）。
- **归一化**：`ToTensor` 不除 255，归一化在 `RGBSeqDataset.__getitem__` 末行 `/127.5-1` → `[-1,1]`。**删掉反而恶化到 95.6%**。
- **开关**：`CHANNEL_FIX=True` / `PIXEL_SCALE="pm1"` / `BLANK_BIAS`（VAC=1.0，TFNet=0.0）。
- **评估**：WER 只用 `official_wer.py`（语料级）；test 冻结只跑一次；报数须含 mean±std+胜率+SNR。
- **排期**：只用实测 ep001 速度，禁止跨配置外推。

---

## 6. 剩余待办（工程，不涨指标）

- [x] 删 p78 死参数 `--w-kd`（本次清理）
- [x] 整合汇总报告（本报告）
- [ ] 打包 train 数据（tar cf 不压缩，用户暂定不执行）
- [ ] 用户自查 D2/P155（先跑 dev 看能否复现 55.84）
- [ ] 可选：SWA/EMA 在健康尾巴抛光（预期 ~0~+1pp，可跳过）

---

## 7. 给"需要的人"的交接说明

1. 训练实际吃 `artifacts/official_rgb`（预处理 RGB 帧，29GB/113 万文件），由 `data/raw/CE-CSL/video`（9.2GB 原始 mp4）经 p78 管线再生。
2. 词表/切分见 `external/TFNet/data/CE-CSL/{train,dev,test}.csv`。
3. 直接拿发布权重做零样本推理即可得 test 35.66%；若要复现 VAC 训练，跑 `p78_train_official_tfnet.py --module VAC --img-size 160`。
4. 所有决策与数字见 `docs/experiments/stage-*.md`，本汇总为索引。
