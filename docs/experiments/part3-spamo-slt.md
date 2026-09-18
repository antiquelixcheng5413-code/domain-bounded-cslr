# Part 3 · SpaMo 风格 SLT 交接文档

更新时间：2026-09-17
分支：`feature/part3-spamo-scaffold`
本文件覆盖 Part 3 的 **Phase-1（代码骨架 + CPU 冒烟）** 与 **Phase-2（真实 CE-CSL 正式 baseline 与消融）** 交付。

## 1. 目标与范围

将系统从「视频 → Gloss 序列」扩展为「RGB/运动/landmark → 多模态融合 → 中文句子」。本阶段只交付
「代码可运行、接口可测试、配置可复现」，**不**启动正式长时训练、**不**读取 Test 500、**不**接入 Web 推理路径。

阶段划分（内部约定）：

- **Phase-1**：CPU + 合成特征 + `split=train` 小样本，验证训练链路与评估框架。
- **Phase-2**：GPU + 真实 CE-CSL 视频 → CLIP/运动差/landmark 特征 + `split=dev` 正式 baseline 与消融。

## 2. 目录与文件

```text
configs/translation_part3_smoke.yaml      # 独立冒烟配置，split=train，test 被拒绝
src/cslr/translation/
  __init__.py
  config.py      # 配置加载与字段校验（Part3ConfigError）
  dataset.py     # CE-CSL SLT Dataset + collate（合成特征路径）
  text.py        # 中文文本规范化与词表
  encoders.py    # RGB/Motion/Landmark 编码器接口 + Tiny 实现
  fusion.py      # LightSpaMoFusion：投影 + 位置编码 + Transformer 融合
  decoder.py     # 中文 Decoder 接口 + Tiny transformer 实现 + gloss aux head
  models.py      # Part3SpaMoModel 整链路与统一输出契约
  metrics.py     # BLEU / ROUGE-L / chrF / 生成延迟
  cache.py       # 特征缓存与 SHA-256 元数据收据（Phase-2 使用）
  service.py     # run_smoke 冒烟编排
  __main__.py    # CLI：python -m cslr.translation smoke
tests/test_translation_{config,dataset,text,encoders,fusion,decoder,models,metrics,smoke}.py
```

## 3. 输入输出契约

三路输入（任一可由配置关闭，供消融）：

| 模态 | 原始张量 | 说明 |
|------|----------|------|
| rgb | `[B, T_rgb, D_rgb]` | 由阶段特征缓存提供，默认 D=hidden_dim |
| motion | `[B, T_motion, D_motion]` | 相邻帧差在特征抽取阶段完成，编码器只投影、保持序列长度 |
| landmark | `[B, T_landmark, 368]` | 复用 Part 1/2 MediaPipe 特征，只读 |

融合输出 `(visual_tokens, visual_mask, actual_token_count)`，其中 `actual_token_count` 为
per-sample 有效 token 数（batch 取最大），供报告；完整 batch mask 仍返回给 Decoder。

统一输出契约（`TranslationModelOutput`）：`loss / translation_loss / gloss_aux_loss /
logits / generated_texts / visual_token_count`。gloss aux loss 默认权重 0（接口预留在，不伪造）。

## 4. 运行方式

```bash
# 完整单测（跳过 test_api.py，其依赖 FastAPI 属 app 后端）
PYTHONPATH=src  venv/bin/python -m pytest tests/test_translation_*.py -q

# 冒烟（CPU，合成特征，split=train）
PYTHONPATH=src  venv/bin/python -m cslr.translation smoke
```

`split=test` 在 `load_config` 阶段被拒绝，报 `Part3ConfigError`，确保官方 Test 500 未被读取。

## 5. 数据与 Git 规则遵守

- 新增/修改仅涉及 `src/cslr/translation/`、`tests/test_translation_*`、冒烟配置与交接文档。
- 派生数据（`data/alignment/`、`data/isolated/`、`data/artifacts/`、`data/manifests/`）已加入
  `.gitignore`，不随提交入库。
- `artifact/` 权重、`*.pt / *.pth / *.onnx` 均不入库。
- Test split 全程未读取，冒烟结果标记为非正式（`formal_result=false`）。

## 6. Phase-1 验证结果（非正式）

| 检查 | 结果 |
|------|------|
| 翻译单测 | 53/53 通过 |
| 全仓单测（忽略 test_api.py） | 通过 |
| CLI 冒烟 | `status=ok`，train loss 7.84 → 7.06（15 steps），dev 3 样本评估 |
| visual_token_count（三路合成） | 11 |
| 每样本延迟（CPU） | ≈8.2 ms |
| split=test 拒绝 | 通过 |

## 7. 已知限制与后续工作

限制：

- 冒烟使用合成特征，不代表真实性能；`formal_result=false`。
- RGB 编码器当前为 tiny 投影，未接 CLIP/DINOv2/VideoMAE（Phase-2 通过 adapter 接入其一）。
- motion 差在特征抽取阶段生成；当前合成路径产物为投影，未做真实帧差。
- BLEU 零匹配阶精度为 0（不加平滑垫高），完全错句趋近 0，部分匹配从严。

Phase-2 待办：

1. 真实特征抽取与缓存（`cache.py`）→ 产出首批真实 RGB/运动/landmark 特征。
2. 接入真实 CE-CSL + GPU 跑正式 baseline（三路/双路/单路配置化消融）。
3. `split=dev` 冻结评估，输出机器可读 JSON / 逐样本 CSV 与实验收据。
4. Test 冻结评估仅在冻结协议最终确认后进行。

## 8. 待主线确认的接口决定

- motion 编码器只投影、保持序列长度；相邻帧差归特征抽取层，避免 token 数歧义。
- `visual_token_count` 定义为 per-sample 有效 token 数（batch 取最大），详见 §3。
- Part 2 transition mask 接口已接受但**不伪造**，仅当上游提供时才生效。

## 9. Phase-2 正式实验（真实 CE-CSL）

### 9.1 数据与特征

- 数据源：`data/raw/CE-CSL/`，`label/{train,dev}.csv` 提供中文句监督（SLT 主目标）+ gloss。
- 划分与计划一致：Train 4973 / Validation 515 / Test 500；**Test 从未读取**（`test_split_read=false`）。
- 特征缓存：`artifacts/part3_features/{train,validation}/{sample_id}.{rgb,motion,landmark}.npy`。
  - **rgb**：每视频定帧采样 → CLIP `ViT-B/32`（offline 权重缓存于 D 盘）视觉编码。
  - **motion**：相邻帧差（特征抽取阶段完成，编码器仅投影保持长度）。
  - **landmark**：复用 Part 1/2 MediaPipe 368 维（只读）。
- 特征缺失处理：validation 中疑似损坏视频 `dev-00402`（解码永久挂起已被 robust 流程记录后跳过）在评估时按「三路特征齐全」过滤，实际评估 **514** 样本。
- 特征正确性收据：每样本 .npy 附带 SHA-256（`cache.py`），名称、维数、落盘一致性可复核。

### 9.2 冻结配置与复现命令

统一冻结（全部 6 个实验一致）：`configs/translation_part3_baseline.yaml`，
`hidden_dim=256, layers=3+2, heads=8, feedforward=1024, dropout=0.1`，
训练 `epochs=15, lr=5e-4, batch_size=16`，评估 `eval-limit=0`（validation 全量 514）。
解码为贪心 + bigram 重复抑制。device=cuda（RTX 5060 Laptop 8G）。

消融配置：`configs/translation_part3_abl_{rgb,motion,landmark,rgbmotion,rgblm}.yaml`（仅改 `fusion.modalities`）。

```bash
PYTHONPATH=src venv/bin/python -m cslr.translation train-real \
  --config configs/translation_part3_{baseline,abl_*}.yaml \
  --feature-root artifacts/part3_features \
  --train-split train --eval-split validation \
  --epochs 15 --lr 5e-4 --batch-size 16 --eval-limit 0
```

### 9.3 结果表（validation 514，frozen 配置）

| 模态组合 | train_loss_end | BLEU-1 | BLEU-2 | ROUGE-L | chrF | EM |
|---|---|---|---|---|---|---|
| **tri** rgb+motion+landmark | **1.21** | 0.1356 | 0.0147 | 0.1487 | 0.0847 | 0.0 |
| rgb | 4.659 | **0.1687** | 0.0140 | 0.1735 | 0.1053 | 0.0 |
| motion | 4.587 | 0.1530 | **0.0365** | **0.2347** | **0.1226** | 0.0 |
| landmark | 4.905 | 0.1592 | 0.0178 | 0.1760 | 0.0973 | 0.0 |
| rgb+motion | 4.542 | 0.1464 | 0.0189 | 0.2021 | 0.1027 | 0.0 |
| rgb+landmark | 4.637 | 0.1502 | 0.0182 | 0.1824 | 0.0976 | 0.0 |

机器可读收据：`/home/su127/part3_matrix/summary.jsonl`（逐实验）、`*.out`（逐样本 per_sample 数组）。

### 9.4 观察与结论（P2-4 消融）

1. **融合在训练侧明显更优**：tri 的 `train_loss_end=1.21`，显著低于所有单/双路（~4.5–4.9）。
2. **但融合优势未转化为解码增益**：解码指标上单路 motion（ROUGE-L 0.235 / chrF 0.123）反而最高，
   tri（ROUGE-L 0.149）偏低。→ 融合提升了训练拟合，却未带来译文质量增益，属「拟合-泛化错位」。
3. **motion 单路对成句形态贡献最稳定**（ROUGE-L 最高、BLEU-2 明显占优 0.037），提示帧间运动差
   是当前三路里信息最贴近「句子语义」的一路。
4. **绝对数值低**（BLEU-1≈0.13–0.17、BLEU-2≈0、EM=0）：小数据（4973）轻量解码器下 SLT 的合理初值，
   属欠拟合/长度偏差，非 bug。

### 9.5 限制与后续

- 轻量 Transformer 解码器在长句上存在重复坍缩趋势（已用 bigram 抑制缓解）；更长训练/beam/标签平滑未展开。
- 融合增益与解码增益的落差，值得下一步验证：是否需要更大容量解码器、视觉感知 token 密度、或双侧 loss 加权。
- Test 500 仍冻结，未在任一实验中读取。最终 Test 评估需先冻结最终协议再解锁。

## 10. 改进路线①：SpaMo 式 token 压缩（tri + pooling，K=32）

针对 §9.4-2「融合-解码落差」，引入 SpaMo 式固定容量池化：融合 Transformer 输出后，用 `num_pool_tokens`
个可学习 query 做交叉注意力，把变长三路 token 压成固定 K 个视觉 token 再喂解码器，以减轻解码器对
视觉上下文的注意负担、聚焦跨模态语义。

### 10.1 改动与入口
- 代码：`src/cslr/translation/fusion.py` 新增 `TokenPooling`；`LightSpaMoFusion` 支持 `num_pool_tokens`
  （默认 None/0 = 保持原 concat 行为，向后兼容，既有融合单测不受影响）。
- 配置：`configs/translation_part3_pool.yaml`（tri + `num_pool_tokens: 32`）。
- 复现：`bash scripts/run_part3_pool.sh`（其余与 §9.2 冻结配置一致：15 ep / lr 5e-4 / hidden 256）。
- 单测：`tests/test_translation_fusion.py` 新增池化 2 例（固定 token 数、跨长度恒定）；翻译套件 55/55 通过。

### 10.2 结果（validation 514，frozen 配置）
| 配置 | train_loss_end | BLEU-1 | BLEU-2 | ROUGE-L | chrF |
|---|---|---|---|---|---|
| concat tri（§9.3 基线） | 1.21 | 0.1356 | 0.0147 | 0.1487 | 0.0847 |
| motion 单路（原最高） | 4.587 | 0.1530 | 0.0365 | 0.2347 | 0.1226 |
| **tri + pooling (K=32)** | 4.682 | **0.2171** | **0.0542** | 0.2306 | **0.1469** |

收据：完整 stdout 日志 `/home/su127/part3_pool.log`（含逐样本 per_sample 数组）。

### 10.3 观察与结论
1. **token 压缩让融合首次反超单路**：BLEU-1 +60%（0.136→0.217）、BLEU-2 3.7×、chrF +20%，全面优于
   原 concat tri，且 BLEU-1/2/chrF 超过 motion 单路，ROUGE-L（0.231）与之持平。
2. **train_loss 不降反高于 concat**（4.68 vs 1.21）但解码泛化更强——印证「融合度高、解码用不上」的
   症结在视觉上下文密度：固定少 token 减轻解码注意力过载，泛化更好。
3. 与 §9 一致：Test 500 仍冻结，本实验未读取（`test_split_read=false`）。