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
2. **解码指标不可信（核验见 §15）**：§15 判别显示，本表几乎所有配置都陷入句子级重复坍缩——motion
   全 514 条塌成「不要的江。」、rgb 全塌成「那是我我是你的的了。」（472/514）、landmark 全塌成
   「他是是我是有的。」、rgb+motion 全塌成「他的的了了。」，仅 tri / rgb+landmark 保留 2–4 种输出。
   因此表内 BLEU/ROUGE/chrF 多为重复病句与参考语料的字词重叠假象，不能作为成句质量证据。
3. **「motion 对成句贡献最稳定」为误判**：motion 的 ROUGE-L 0.235 来自其唯一病句「不要的江。」与参考的
   高频字重叠，并非更贴近句子语义。
4. **「绝对数值低属欠拟合初值」为误判**：除 EM 全 0 外，每栏都只有 1–4 种输出，这是解码器在 4973
   小样本上的重复坍缩病态，不是广度不足的欠拟合。此状态下跨配置 BLEU 高低对比一律不构成有效证据。

### 9.5 限制与后续

- 轻量 Transformer 解码器在本数据下整体陷入**句子级重复坍缩**（见 §15）；bigram 抑制不足以解决，
  更长训练/beam/标签平滑亦未扭转。
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
1. **「token 压缩首次反超单路」为误判（核验见 §15）**：§15 判别确认 pool K=32 的 514 条预测 100%
   坍缩为同一病句「不不要的的人的了。」。表内 BLEU-1 0.217 / BLEU-2 0.054 / ROUGE-L 0.2306 全部来自该句
   与参考的高频字/bigram 重叠，非真实译文质量，「反超单路」不成立。
2. **高 BLEU 不来自更好成句**：pool 训练 loss（4.68）不高，但生成退化为单句；相对地，concat tri（§9）
   至少产出过 2 句通顺句（§15）。不能将本表 BLEU 解读为泛化提升。
3. 与 §9 一致：Test 500 仍冻结，本实验未读取（`test_split_read=false`）。

## 11. 改进路线②：跨模态时间对齐（align_frames）

针对特征观测：`rgb=(T,512)` 与 `motion=(T-1,512)` 本就是同一帧时间线（motion = 相邻帧差，长度差 1），
而 `landmark=(48,368)` 是无时间轴的固定 48 个全局语义 token。因此「跨模态时间对齐」的正确实现是把
rgb/motion 在同一帧索引上配对成一个 frame token，landmark 保留为全局上下文，再进融合 + 池化。

### 11.1 改动与入口
- 代码：`src/cslr/translation/fusion.py` 新增 `align_frames`（默认 False，关闭时走原 concat，向后兼容）；
  开启且在次含 rgb+motion 时，用 `frame_proj`（Linear(2D→D)）把逐帧 `[rgb_i, motion_i]` 并为一个 token，
  丢弃未配对的 rgb 末帧；landmark 等余下模态照旧接续。
- 配置：`configs/translation_part3_align.yaml`（tri + `align_frames: true` + `num_pool_tokens: 32`）。
- 复现：`bash scripts/run_part3_align.sh`（其余与冻结配置一致）。
- 单测：fusion 新增对齐 2 例 + concat 回归 1 例；fusion 套件 9/9 通过。

### 11.2 结果（validation 514，frozen 配置）
| 配置 | train_loss_end | BLEU-1 | BLEU-2 | ROUGE-L | chrF |
|---|---|---|---|---|---|
| concat tri（§9.3 基线） | 1.21 | 0.1356 | 0.0147 | 0.1487 | 0.0847 |
| tri + pooling（§10） | 4.682 | **0.2171** | **0.0542** | 0.2306 | **0.1469** |
| **tri + align + pooling（§11）** | 4.195 | 0.1586 | 0.0286 | 0.1928 | 0.1045 |

收据：完整 stdout 日志 `/home/su127/part3_align.log`（含逐样本 per_sample 数组）。

### 11.3 观察与结论
1. **相对判读需谨慎（核验见 §15）**：本配置同样全 514 条坍缩为单一病句「今天是我们的。」，表内 BLEU
   与 concat 的差异同为坍缩句重叠假象，「帧对齐优于 concat」「不如纯 pooling」的差分结论在不结合 §15
   判别时不可信。
2. 帧对齐作为机制探索方向无碍，但两侧都未真正成句，无法据此判定其相对优劣。
3. Test 500 仍冻结，本实验未读取（`test_split_read=false`）。

## 12. 改进路线③：标签平滑与 beam search（无效）

在 §10 的 pool 配置上叠加训练/解码侧技巧，验证能否再提升译文质量。

### 12.1 改动与入口
- 代码：`src/cslr/translation/models.py` 新增 `label_smoothing`（接到
  `CrossEntropyLoss(label_smoothing=…)`）；`decoder.py` 新增 `generate_beam`
  （按样本、逐前缀 batch 解码；`beam_size=1` 退化为贪心），`Part3SpaMoModel.forward`
  新增 `beam_size` 路由。两者默认关闭，向后兼容。
- 由 config 驱动：`model.label_smoothing` / `model.beam_size`。
- 复现：`bash scripts/run_part3_smooth.sh`（A：平滑 0.1）；`scripts/run_part3_smooth_beam.sh`
  （B：平滑 0.1 + beam 4）。
- 单测：翻译套件 57/57 通过。

### 12.2 结果（validation 514，§10 pool 基线上叠加）
| 配置 | train_loss_end | BLEU-1 | BLEU-2 | ROUGE-L | chrF |
|---|---|---|---|---|---|
| tri + pooling（§10） | 4.682 | **0.2171** | **0.0542** | **0.2306** | **0.1469** |
| + label_smoothing=0.1 | 4.551 | 0.1636 | 0.0210 | 0.1926 | 0.1065 |
| + label_smoothing + beam=4 | 1.540 | 0.1336 | 0.0000 | 0.1456 | 0.0775 |

收据：stdout 日志 `/home/su127/part3_smooth.log`、`/home/su127/part3_smooth_beam.log`。

### 12.3 观察与结论（负结果）
1. **标签平滑不助**：BLEU-1 −25%、chrF −27%。小数据 + 轻量解码器下软化收益为零（同为坍缩背景下）。
2. **beam search 更差**：叠加后 BLEU-1 0.134、BLEU-2 归零。beam 挑选高累计概率的短/简单成句，
   在此粒度上不增质量（或需 length-penalty/长度归一，本次未做）。
3. → 本节对比同受 §15 坍缩假象影响；「pool-alone（§10）为最优」按 §15 判定不成立（pool 亦坍缩，绝对
   质量全为 0）。提升方向应优先解决解码器重复坍缩（§15 尾）。
4. Test 500 仍冻结，本实验未读取（`test_split_read=false`）。

## 13. 改进路线④：预训练 mT5 解码器 adapter（负结果 · 退化坍缩）

针对 §12 结论「应转向解码器容量或数据」，尝试把轻量 transformer 解码器替换为预训练 **mT5-small**
解码器作中文语言先验，验证能否借助其强语言模型提升译文质量。

### 13.1 改动与入口
- 代码：新增 `src/cslr/translation/mt5_decoder.py`（`MT5ChineseDecoder`）：融合输出经 `visual_proj`
  （LayerNorm→Linear，D→d_model=512）作交叉注意力 memory；目标侧用 mT5 词表子词编码（decoder_start=<pad>）；
  LM head 为共享 embedding 转置（tied）。训练更新 visual_proj + 全部 mT5 解码层（未冻结）。
- 配置/组装：`config.py` 新增 `model.decoder_backbone="tiny"|"mt5"`、`mt5_path`、`mt5_max_target_tokens`；
  `models.py` 据 backbone 选择 `MT5ChineseDecoder` / 原 tiny 解码器，均走同一 `decode_from_visual` 契约。
- 权重：mT5-small 经 `hf-mirror.com` 下载缓存于 D 盘（直连 huggingface.co 不通），
  落地 `/mnt/d/part3_models/mt5-small-saved`（safetensors 688MB + tokenizer）。
- 配置：`configs/translation_part3_mt5.yaml`（tri + `num_pool_tokens: 32` + `decoder_backbone: mt5`，其余与 §10 冻结一致）。
- 复现：`bash scripts/run_part3_mt5.sh`。
- 单测：`tests/test_translation_mt5.py` 用合成 fake T5 覆盖 teacher-forcing/编码/生成/参数更新，4/4 通过。

### 13.2 结果（validation 514，tri+pool 基线上换解码器）
| 配置 | train_loss_end | BLEU-1 | BLEU-2 | ROUGE-L | chrF | EM |
|---|---|---|---|---|---|---|
| tri + pooling（§10，tiny 解码器） | 4.682 | **0.2171** | **0.0542** | **0.2306** | **0.1469** | 0.0 |
| **tri + pool + mT5 解码器** | 10.587 | 0.0221 | 0.0000 | 0.1770 | 0.0679 | 0.0 |

收据：后处理标准化日志 `/home/su127/mt5_train.log`（含逐样本 per_sample 数组），记为 `tri-pool-mt5`。
评估改为 `chunk=8` + `torch.cuda.empty_cache()`（`service.py`），规避生成阶段显存峰值 OOM。

### 13.3 观察与结论（负结果 · 强退化）
1. **train_loss 收敛到约 10.6**，但**生成完全退化**：514 条验证预测 100% 坍缩为同一字符串「我。」，
   EM=0、BLEU-2=0 —— mT5 解码器自学成一个近似「中文 LM」，其强劲语言先验在贪心 argmax 下压过视觉
   交叉注意力，所有样本输出最频繁短句后随即 EOS。
2. **对照 §9.4-2「融合-解码落差」显著增强**：预训练解码器比 tiny 解码器更依赖语言先验、更易忽略
   视觉上下文，在 4973 小样本上负迁移更严重（BLEU-1 0.217→0.022，−90%）。
3. → 简单「装个大预训练解码器」在此数据规模下不成立，需「冻结解码器仅训投影/交叉注意力」或加入视觉
   条件强化（如 pooling 显式监督）。作为有意义的负结果记录。（对照引用的「pool 为全局最优」按 §15 不成立。）
4. Test 500 仍冻结，本实验未读取（`test_split_read=false`）。

### 13.4 缓解实验：冻结语言路径（A）与视觉对齐监督（B，均为负结果）

针对 §13 的坍缩，尝试两种机制不同的缓解，验证是否能让 mT5 不再退化为纯语言模型：

- **A · 冻结语言路径（`mt5_freeze: cross_only`）**：冻结 mT5 的共享 embedding / self-attn / FF / LN，
  仅训 `visual_proj` + 解码器交叉注意力，强制视觉条件化。
- **B · 视觉↔文本对齐 aux（`mt5_visual_aux_weight: 1.0`）**：新增 `visual_text_align`
  cosine 损失，把池化视觉表示拉向目标 token 平均嵌入，给视觉路径一条绕过 LM 主导的直接监督。

| 配置 | train_loss_end | BLEU-1 | BLEU-2 | ROUGE-L | chrF | 预测形态 |
|---|---|---|---|---|---|---|
| tri + pool（§10，tiny 解码器） | 4.682 | **0.2171** | **0.0542** | **0.2306** | **0.1469** | 全 514 条 =「不不要的的人的了。」（§15） |
| mt5 全程可训（§13） | 10.587 | 0.0221 | 0.0 | 0.1770 | 0.0679 | 全 514 条 =「我。」 |
| mt5 + A 冻结语言路径 | 12.544 | 0.0156 | 0.0 | 0.0271 | 0.0245 | 全 514 条 =「。。。」 |
| mt5 + B 视觉对齐 aux | 9.271 | 0.0017 | 0.0 | 0.1956 | 0.0701 | 全 514 条 =「。」 |

收据：`/home/su127/mt5_cross_train.log`（A）、`/home/su127/mt5_vaux_train.log`（B，均含 per_sample）。

结论（三种策略全部坍缩，方向性负结果）：
1. **坍缩对可训练参数策略鲁棒**：全程可训→「我。」；冻结语言路径→「。。。」；加视觉对齐 aux→「。」。
   本质同一：mT5 解码器在小数据上自学成近似中文 LM，贪心 argmax 下视觉交叉注意力（32 池化 token）
   信号始终压不过其 250k 词表的语言先验。
2. **B 的 aux 确实让视觉路径在学（ROUGE-L 0.196，三种里最高）**，但仅学会了输出置信度最高的
   句末标点「。」，未产生任何内容词 →「拟合-解码落差」在 mT5 上被放大到极值。
3. → mT5 预训练解码器方向在本数据规模下为死路（有意义的负结果）。「pool（tiny 解码器）为全局最优」
   按 §15 判定不成立——它同样坍缩为单句病句。若未来数据量大幅增大、改用「t5 作先验 + logit 混合
   （product-of-experts）而非微调」、或视觉 token 密度显著提高，才值得复验。
4. Test 500 仍冻结，本实验未读取（`test_split_read=false`）。

## 14. 改进路线⑤：视觉 token 密度扫描（K=64/128，负结果）

针对 §13 与 §12 共同指向的「视觉信号弱 / 解码坍缩」假设，在 §10 的 tiny 解码器 + pool 基线上把
`num_pool_tokens` 从 32 提到 64/128，验证"视觉上下文太少"是否是坍缩根因。其余冻结设置与 §10 完全一致。

- 配置：`configs/translation_part3_pool64.yaml` / `translation_part3_pool128.yaml`（仅 `num_pool_tokens`）。
- 复现：`bash scripts/run_part3_pool64.sh` / `run_part3_pool128.sh`。

| K（pool tokens） | train_loss_end | BLEU-1 | BLEU-2 | ROUGE-L | chrF | 预测形态 |
|---|---|---|---|---|---|---|
| 8 | 4.759 | 0.1517 | 0.0 | 0.1895 | 0.0939 | 全 514 条 =「那里有的的了。」|
| 16 | 4.833 | 0.1390 | 0.0 | 0.2016 | 0.0915 | 全 514 条 =「今天的的了。」|
| **32（§10）** | 4.682 | **0.2171** | **0.0542** | **0.2306** | **0.1469** | 全 514 条 =「不不要的的人的了。」（§15 修正） |
| 64 | 1.205 | 0.0455 | 0.0208 | 0.0505 | 0.0318 | 全 514 条 =「你需要一些水果吗？」|
| 128 | 4.307 | 0.1456 | 0.0171 | 0.2016 | 0.1034 | 全 514 条 =「非常是我的。」|

收据：`/home/su127/pool8.log`、`/home/su127/pool16.log`、`/home/su127/pool64.log`、`/home/su127/pool128.log`。

结论（负结果 · 双向完整扫描，含 §15 修正）：
1. **「K=32 是甜点且正常成句」结论作废（§15）**：判别显示 K=32 同样 514/514 坍缩为「不不要的的人的了。」
   （此前 BLEU-2=0.054 非零，是病句与参考的 bigram 重叠假象）；8/16/64/128 亦各坍缩为单句。本表 5 档
   全部句子级重复坍缩，**不存在正常成句档**。
2. 与 mT5 坍缩互为镜像：mT5 是 **词/token 级**坍缩（全「我。」），tiny 解码器是 **句子级**坍缩（含 K=32，
   各输出固定病句）。共同点：小数据（4973）下解码器整体沉入「重复输出」，无规避配置。
3. → 提升方向应转向数据量或更强的视觉→文本监督，而非调视觉 token 密度。**「pool-alone（K=32）全局最优
   定稿」作废（§15）：当前没有可用配置。**

## 15. 评估可信度核验（补记 · 全局重复坍缩）

对 §9–§14 全部评估日志的逐样本 `per_sample` 做预测多样性判别（统计 distinct 译文数及其占比），发现此前
「K=32 正常成句」「motion 解码最优」等结论为**误判**，特此更正，并把「全局重复坍缩」固化为重要负结果。

判别方法：对每份日志解析 `per_sample`，统计 distinct 预测数与 top 句占比。结果（validation 514）：

| 日志（配置） | distinct | top 占比 | 实测预测形态 |
|---|---|---|---|
| abl_motion.out（§9） | 1 | 100% | 全 514 =「不要的江。」|
| abl_rgbmotion.out（§9） | 1 | 100% | 全 514 =「他的的了了。」|
| abl_landmark.out（§9） | 1 | 100% | 全 514 =「他是是我是有的。」|
| part3_pool.log（§10，K=32） | 1 | 100% | 全 514 =「不不要的的人的了。」|
| mt5 三变体 / align / smooth(beam) / pool8/16/64/128 | 1 | 100% | 各输出固定病句 |
| part3_matrix/tri.out（concat 三路，§9） | 2 | 51% | 「这是我第一次来美容院。」(261) /「这是我们吃蔬菜的。」(253) |
| abl_rgblm.out（§9） | 4 | 47% | top「这是我我是的的。」等病句 |
| abl_rgb.out（§9） | 3 | 92% | 「那是我我是你的的了。」(472) |

判定与更正：
1. **§9–§14 几乎所有配置都是整段句子级重复坍缩**，仅 tri / rgb+landmark / rgb 保留 2–4 种输出且仍高度集中。
   没有任何配置产出多样、语义对齐的译文；全部 EM=0。
2. 因此 §9.4、§10.3、§11.3、§12.3、§13.3、§14 中所有跨配置 BLEU/ROUGE/chrF 比较（「K=32 反超 motion」、
   「pool 首次反超单路」「32 是唯一甜点」等）**均建立在重复病句与参考的重叠假象上，不构成有效证据**，
   已在上文各节逐条更正。
3. **相对最「像样」的是最强的 concat 基线（§9 tri，不加 SpaMo）**：它产出过 2 句通顺中文（美容院/吃蔬菜），
   其余配置（含 SpaMo pool）连通顺度都不如它；但这两句与视频语义无关（记忆锚点），EM 仍为 0，绝对不可用。
   该观察与「加 SpaMo 提升效果」的原叙事相反。
4. **根本病因**：解码器在 4973 小样本上的**重复坍缩**（tiny 解码器=句子级、mT5=词级），叠加 BLEU 指标对
   病句的虚高掩盖了坍缩。融合/压缩方法从未在「可产出多样译文」的状态下被公平度量，**不能据此判定 SpaMo
   方法本身有效或无效**。后续一切改进应先解决重复坍缩（数据扩充 / 更长训练 / 更强的长度与重复约束 /
   更强的视觉→文本监督），并辅以 distinct 判别，避免再用坍缩输出的 BLEU 做横比。

Test 500 在本核验及全部历史实验中被全程冻结，从未读取。

## 16. 改进路线⑥：大模型方向 —— Gloss-token → 国产 LLM（教授建议，突破性改善）

**背景**：§15 确认自研轻量解码器（tiny/mT5）在 4973 小样本上**全面重复坍缩**，自研解码器这条路径被
小数据锁死。教授建议改为「token 直接给大模型 / 隐含层信息给大模型」，借预训练中文大模型的语言先验
绕过坍缩。本节省 `Gloss → LLM = 中文句子` 的 oracle 验证（路线 A1，gold gloss 作输入）。

### 16.1 方法
- 输入：CE-CSL 官方标注的 **Gold Gloss 序列**（如「10/年/鱼/禁止1/区/时间/长/不/。」），`/` 切分后喂模型。
- 模型：本地微调后留存的中文 LLM（modeling 语言先验），prompt 要求「把手语 gloss 序列翻译成通顺中文句」。
- 指标：与全项目一致的 BLEU-1/2、ROUGE-L、chrF、EM + **distinct 多样性判别**（§15 教训的固化）。
- 本次为 oracle：不设视觉→gloss 编码器，只验证「大模型能否把 gloss 译成多样通顺中文」。官方 test 未读。
- 收据 JSON：`/home/su127/route_a_dev_full.json`（Qwen2.5-1.5B, dev 全 515）、
  `/home/su127/route_a_qwen05b100.json`（Qwen2.5-0.5B, dev 100）。

### 16.2 结果（validation/dev，gold gloss oracle）

| 模型 | dev 规模 | distinct | BLEU-1 | BLEU-2 | ROUGE-L | chrF | EM | 坍缩 |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-1.5B-Instruct** | 515 | **514** | **0.6706** | **0.4989** | **0.6725** | **0.5792** | 0.0835 | 无 |
| Qwen2.5-0.5B-Instruct | 100 | 99 | 0.6388 | 0.4658 | 0.6485 | 0.5561 | 0.04 | 无 |
| 自研 tiny 解码器（§10「最优」，历史对照） | 514 | 1 | 0.217(假象) | 0.054 | 0.231 | 0.147 | 0 | **全坍缩** |

抽样（Qwen2.5-1.5B，ref→pred，语句均通顺、语义对应）：
- `10年的禁鱼区不是很长时间。` → `十年禁渔区时间长了。`
- `2023年的高考有一千多万考生。` → `2023年高考报名人数超过千万。`
- `下一次强降雨出现在重庆。` → `下一次大雨在重庆。`

### 16.3 观察与结论（突破性）
1. **坍缩彻底消失**：Qwen-1.5B 在 dev 全 515 条上 distinct=514（几乎每句不同），0.5B 也达到 99；
   对比 §15 的「所有自研解码器 distinct=1」。教授「大模型解决重复坍缩」的判断**成立**。
2. **指标全面大幅上升（在真译文上，非假象）**：BLEU-1 0.217→**0.671**、BLEU-2 0.054→**0.499**、
   ROUGE-L 0.231→**0.673**、chrF 0.147→**0.579**、EM 0→**8.4%**。这是历史上首次 EM>0、首次 distinct>2。
3. **规模敏感性温和**：0.5B vs 1.5B 差异约 5% BLEU-1，说明该增益主要来自「多语言预训练先验」本身而非容量；
   更大的模型可能再小幅提升，但边际递减。
4. **诚实边界**：这是 **gold-gloss oracle**，证明「token 进大模型」可行；但视觉→gloss 的识别那一段
   （红线）仍未做。完整端到端需再接一个视觉编码器产出 gloss token（路线 A2 / B）。
5. 因此**自研 tiny/mT5 解码器路径（§9–§14）正式关闭**，后续主攻大模型方向（§16 oracle → 端到端桥接）。

### 16.4 复现命令
```
MODEL=/mnt/d/part3_models/ms_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master
venv/bin/python scripts/route_a_gloss_llm.py --label data/raw/CE-CSL/label/dev.csv \
  --model $MODEL --device cuda --limit 515 --out /home/su127/route_a_dev_full.json
```
（模型经 modelscope 下载至 `/mnt/d/part3_models/ms_cache/`；InternLM 因旧 modeling 与新版 transformers
 不兼容未纳入，跨厂商对比拟在 B 路线用闭源 API 进行。）

## 17. 改进路线⑥续：视觉×大模型端到端（B1 尝试，负结果 · 语义未对齐）

**背景**：§16 证明「gold gloss → 大模型」行稳且不坍缩，但那是 gold-gloss oracle。B1 尝试打通真正的
端到端——把三路视觉特征投影成大模型能看的嵌入前缀，让大模型自己做「视觉→中文」。

### 17.1 方法（scripts/route_b_visual_qwen.py）
- 视觉侧：RGB/motion/landmark 三路缓存特征各经 `Linear→GELU→Linear` + 时序池化到固定 token，
  再 LayerNorm + 投影到 Qwen 的 hidden_size，作输入嵌入前缀。
- 语言侧：冻结 Qwen，前缀后接固定指令「请根据手语动作，翻译成通顺的中文句子：」再接目标文本，
  teacher-forcing 交叉熵，只训练投影层。
- 生成侧：手动自回归解码（见 17.3 坑 1），第一步用「视觉+指令」前缀嵌入，之后逐字续。

### 17.2 结果（全部语义未对齐，负结果）
| 训练规模 | BLEU-1 | distinct | 现象 |
|---|---|---|---|
| 300 样本 × 30 ep | 0.139 | 20/20 | 通顺中文但语义无关；混入 BOM/泰文/乱英文 |
| 全量 4973 × 8 ep | **0.084** | 48/50 | 反而更差，爆「Human/Assistant/Question:」chat 模板 |

抽样（全量训后，ref → pred，均为"无关但通顺"）：
- 「红色在过年期间是很吉祥的颜色。」 → 「我们应当树立正确的世界观、人生观和价值观。」
- 「经济全球化是必然的趋势。」 → 「是的，我也是。」
- 「给宝宝喂一些牛奶。」 → 「我们不能把工作和家庭关系分开。」

### 17.3 观察与结论
1. **先退后进两步**：全量训后 loss 只降到 2.13（300 样本能到 0.05），eval 即使是训练过的样本也答不对，
   说明投影层学的不是「视觉→语义」，而是「续写常见中文」。
2. **根因**：冻结大模型 + 只训轻量投影层，视觉信号太弱，压不住多语言/对话先验——这跟 tiny/mT5 坍缩
   是同一病理的镜像（那里坍缩成病句，这里坍缩成"无关通顺句"）。
3. **两个工程坑（已固化）**：
   - transformers `generate()` 对 `inputs_embeds`-only 有 bug（把前缀长度当 0，输出空串），需手动自回归解码。
   - 只传视觉前缀不给指令时，大模型直接 EOS（空输出）；加固定指令前缀后能出句。
4. **换文本 LLM 骨架（Qwen→GLM→InternLM）不解决此病根**（病因是数据小与条件弱，非模型选择）。
5. **因此 B1「冻结投影」路线关闭**。后续主攻两阶段「视觉→gloss→LLM」（复用 §16 已通的 A 通道），
   并以「原生多模态 VL 零样本」作并行验证。详见仓库外团队计划
   `中文手语SLT大模型方向-团队执行计划.txt`（阶段 P0–P3）。

## 18. 改进路线⑦：冻结 Qwen2.5-VL 视觉塔 48 帧编码器（全量 · 帧数不是杠杆）

**背景**：P4 交接 §9.1 建议「换表征、留识别头」——用冻结的 Qwen2.5-VL 视觉塔替代 landmark
作为 CTC 输入，且 16 帧缓存信息量不足，明确指出的下一步是「同编码器在 48–96 帧采样」。
本仓库在 2026-09-28 把该方向推进到**全量**：模型（`/mnt/d/cslr-tools/models/Qwen2.5-VL-3B-Instruct`，
7.1 GB）下载到 D 盘，按 48 帧对**全部 4973 train + 515 dev** 提取 `[T,2048]` 特征（0 失败），
再用同一 CTC 头训练并 dev 评测。

### 18.1 方法与复现

```bash
# 1) 提取（train + dev，T=48；dtype float32，feature_size 2048）
PYTHONPATH=src ./venv/bin/python3 -m cslr.recognition.qwen_vl_features \
  --data-root data/raw/CE-CSL --output data/processed/ce-csl-qwenvl48 \
  --split train --frames 48 --model /mnt/d/cslr-tools/models/Qwen2.5-VL-3B-Instruct --device cuda
PYTHONPATH=src ./venv/bin/python3 -m cslr.recognition.qwen_vl_features \
  --data-root data/raw/CE-CSL --output data/processed/ce-csl-qwenvl48 \
  --split dev   --frames 48 --model /mnt/d/cslr-tools/models/Qwen2.5-VL-3B-Instruct --device cuda
# 2) 训练 + 评测（一键脚本：scripts/run_vl_frames.py）
python3 scripts/run_vl_frames.py --frames 48 --vocab-cap 300 --skip-extract
```

收据：`artifacts/metrics/part4-vl48-cap300-train.json`（训练）、
`artifacts/metrics/part4-eval-vl48-cap300.json`（dev 评测），均 `test_split_read: false`。

**性能修复（本轮顺带完成）**：原 `sample_frames` 每帧 `set(CAP_PROP_POS_FRAMES)` 随机 seek
（实测 70 ms/帧，MP4 关键帧 seek 极慢），改为顺序 `grab()+retrieve()` 后 3.6 s → 0.32 s/条，
端到端单条从 5.7 s 降到 2.3 s，全量 5488 条提取约 3.5 h。单元测试 `test_clip_features.py` 6/6 通过。

### 18.2 结果（dev 515，全量训练 4973，与历史横向对比）

| 表征 | 帧数 | 训练样本 | WER | CER | seq EM | distinct | kinds | 平均假设长度 | blank_ratio |
|---|---|---|---|---|---|---|---|---|---|
| MediaPipe landmarks | 96 | 4803 | 0.8234 | **0.6710** | 0.000 | 0.0039 | 2 | 1.03 | — |
| frozen Qwen2.5-VL（run 1） | 16 | 1293 | **0.8014** | 0.7791 | 0.000 | **0.0719** | **36** | **1.95** | — |
| frozen Qwen2.5-VL（run 2） | 16 | 1396 | 0.8129 | 0.7918 | **0.0020** | 0.0379 | 19 | 1.75 | — |
| **frozen Qwen2.5-VL（本轮）** | **48** | **4973** | 0.8529 | 0.7107 | 0.000 | 0.0214 | 9 | 1.67 | **0.965** |

### 18.3 观察与结论（负结果 · 帧数不是杠杆）

1. **「16→48 帧」未兑现 §9.1 预期，反而更接近 landmark 的坍缩特征**：distinct 从 36（16 帧、
   1396 样本）掉到 11、输出仅 9 种 token、425/515 条输出 `<unk>`、blank_ratio 0.965。CER 0.7107
   比 16 帧（0.779）略好，但来自「输出更少 token」（平均长度 1.67 vs 1.95），是**更深的坍缩**，
   不是信号变好。
2. **训练内部信号同样反常**：train_loss 在第 36 epoch 后**转负**（-0.13 → -2.3），val_loss 从
   8.7 一路升至 15，第 47 epoch early stop（best_epoch=7）。负 CTC 训练损失 +「训练越好、验证越差」
   的过拟合特征，说明模型在**记忆训练分布**，没有学到可泛化的 gloss 判别。
3. **与 §5/§8.2 结论汇合**：把通用视觉塔从 16 帧加到 48 帧、数据从 1293 补全到 4973，模型依然
   坍缩，只是坍缩程度波动。**帧数不是杠杆；瓶颈仍在表征本身**——冻结通用视觉塔（无论是 landmark
   坐标还是 VL patch embedding 的逐帧均值池化）对「正在打哪个 gloss」区分度不足。
4. **值得注意的信号**：全量训练（4973）首次让 vocab_utilization 爬升到 20–29%（训练中段），但
   最终评测仍回落到 2.99%——表明模型学到了「多说什么 token」的分布，却没有学到「对哪段视频说
   哪个 token」的对应关系。这与 B1（§17）的「续写常见中文」是同一病理。
5. **方向性结论**：继续在「冻结编码器 + 逐帧池化 + CTC」这条线上加帧数、调词表或调 LR 属于低
   边际收益。真正未验证的杠杆是 **CLIP RGB/运动/landmark 三路融合 + 更强的时序建模**（SpaMo
   主线）以及**视觉信号压过语言先验的判别式预训练**，需在 P2 之后的团队计划中决策。

## 19. 两阶段端到端对照基线（Route C：CTC 预测 gloss → LLM → 中文）

**目的**：A 通道（§16）用的是 gold gloss（oracle 上界 BLEU-1 0.671）。Route C 把 CTC 识别器
的**预测** gloss 接进同一个 LLM 通道，量化「识别误差经 LLM 放大/缓冲」的真实端到端基准。
链路：`视频特征 → CTC 识别器 → 预测 gloss → Qwen2.5-1.5B-Instruct → 中文句子`。

**方法**（`scripts/route_c_ctc_gloss_llm.py`，2026-09-28）：
- checkpoint：`artifacts/checkpoints/ctc-vl48-cap300.pt`（§18 的 VL48 全量模型）
- 特征：`data/processed/ce-csl-qwenvl48/`（48 帧 VL 塔，dev 515 全覆盖）
- LLM：`/mnt/d/part3_models/ms_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master`
  （与 A 通道同一模型、同一 prompt，可直接对比）
- 评测：dev 全量 515 条，route_a 的 `run_qwen`/`summarize` 原样复用
- 收据：`artifacts/metrics/part4-route-c-vl48-dev515.json`（`test_split_read: false`）

**结果（dev 515）**：

| 路线 | gloss 来源 | BLEU-1 | BLEU-2 | ROUGE-L | chrF | distinct | EM |
|---|---|---|---|---|---|---|---|
| A（§16 oracle） | gold gloss | **0.671** | 0.499 | 0.673 | — | **514** | 8.4% |
| **C（本实验）** | CTC 预测 | 0.0048 | 0.0 | 0.039 | 0.015 | 8 | 0.0 |

**观察与结论**：
1. **端到端被 CTC 坍缩完全锁死**：CTC 预测几乎全是 `<unk>` 或单个 `要`，LLM 输出只有 8 种句子
   （`你好！` 占 33.2%、若干单字），BLEU-1 0.0048 ≈ 0，EM 0。识别误差没有给 LLM 任何可用的
   语义输入。
2. **LLM 忠实传递、既不放大幻觉也不补救**：gloss=`要` → 输出`要`；gloss=空 → 模板`你好！`。
   与 §17 B1（LLM 自主续写常见中文）相反——prompt 里的 gloss 足够强时 LLM 不会自己编，
   说明**两阶段架构本身无病，病在 P1 识别质量**。这与 §18.3 的结论一致：问题被精确定位到
   「视觉特征 → gloss」这一段。
3. **对后续的工程意义**：route_c 是可复现的两阶段评测入口。任何 P1 改进（换表征/预训练）都
   可以直接用它看端到端是否从 0.0048 爬升，而不是只看中间 WER。
4. **下一步（§19 之后）**：判别力诊断（§20）将用 balanced 指标（per-gloss ROC-AUC）复核
   「特征到底有没有 gloss 信息」——如果 AUC 高，说明信息在特征里但 CTC 没榨出来，判别式
   预训练值得做；如果 AUC 低，则必须换特征来源。

## 20. 判别力诊断（balanced per-gloss ROC-AUC · 实词有强信息、虚词本质不可判）

**动机**：§18 之前的探针用「多数类准确率」报 landmark 0.864，被类别不均衡污染，无法复核。
本轮用 **per-gloss ROC-AUC**（平衡指标）重新回答「缓存的四种特征族里到底有没有 gloss 信息、
有多少」，为判别式预训练是否值得做提供依据。

**方法**（`scripts/discrim_probe.py`，2026-09-28）：
- 特征：landmark（48×368）/ CLIP rgb（193×512）/ CLIP motion（192×512）/ VL48（48×2048）
- 聚合：逐帧**均值池化**（最保守的线性探针），每样本一个向量
- 分类：`OneVsRestClassifier(LogisticRegression)`，train 4973 拟合、dev 评测
- 标签：train 频次 ≥5 的 gloss 取 top-200（多标签 one-hot）
- 指标：per-gloss ROC-AUC（dev 有正例的 173 个 gloss）、macro 均值、多数类基线 0.5
- 收据：`artifacts/metrics/part4-discrim-probe.json`（`test_split_read: false`）
- 注意：队友 part3 缓存 dev 仅 514/515（缺 dev-00403），probe 跳过并记录覆盖率

**结果（macro AUC，基线 0.5）**：

| 特征族 | macro-AUC | 最强 5 gloss（AUC） | 最弱 5 gloss（AUC） |
|---|---|---|---|
| landmark | **0.6821** | 房子 .988 / 休息 .974 / 怎么样 .972 / 希望 .967 / 睡觉 .959 | 了 .021 / 相信 .049 / 又 .117 / 决定 .174 / 行李 .197 |
| VL48 | 0.6459 | 行李 1.0 / 房子 .998 / 告诉（我） .967 / 休息 .963 / 又 .955 | 了 .047 / 好不好 .130 / 别人 .173 / 决定 .239 / 走 .248 |
| rgb | 0.6265 | 告诉（我） .998 / 行李 .988 / 这 .942 / 汽车1 .934 / 这里 .929 | 知道 .129 / 决定 .177 / 学校 .238 / 相信 .244 / 走 .252 |
| motion | 0.5354 | 卖 .951 / 票 .948 / 东西2 .935 / 上 .909 / 了 .897 | 房子 .060 / 走 .080 / 前 .125 / 不行 .127 / 天气 .140 |

**观察与结论（修正 §18.3 的表征结论）**：
1. **「特征完全无 gloss 信息」不成立**：四族 AUC 全部 >0.5，静态均值池化 + 线性层即可挤出
   信息。此前「~80% WER 天花板 = 表征无判别力」的说法应修正为**「弱判别力」**。
2. **实词 vs 虚词的极端分化（核心发现）**：名词/动词/形容词（房子、休息、行李、告诉（我））
   在**所有**特征族里都达到 0.93–1.0 AUC；语法功能词（了、又、好不好、决定、走）几乎不可判
   （0.02–0.25）。此分化跨表征稳定 → **虚词信息不在单模态视频里**，依赖上下文/语言先验，
   视觉→gloss 在虚词上存在信息论下限。
3. **motion 特征最弱（0.5354）**，仅「卖/票/东西2」等动作性 gloss 有信号 → motion 可作为
   冗余通道，不宜作主特征。
4. **对方向的直接启示**：
   - **判别式预训练上限明确**：只能强化实词路（AUC 0.93→更高更稳），无法创造虚词信息；
     预期要管理，别指望它把 WER 从 0.85 拉到可用。
   - **两阶段架构（视觉→实词 gloss + LLM 语法重组）有了新依据**：A 路线（§16）的成功部分
     归因于 LLM 恰好补上视觉缺失的虚词；端到端（§19）的坍缩则来自 CTC 连实词都没学稳。
   - **下一步最有性价比的实验**：在判别式预训练之外，可先做「**只评测实词 gloss 的 CTC
     质量**」（把 WER 拆成实词/虚词两部分），看实词识别是否已显著好于总 WER——若实词 WER
     已明显低，说明模型学到了该学的，虚词天花板是数据性质而非训练问题。

## 21. 实词/虚词分拆 WER（CTC 连信息最强的实词都没学到 · 瓶颈在训练而非虚词天花板）

**动机**：§20 的探针证明「特征里有强实词信息（AUC 0.93–1.0）、虚词信息论下限」。那么 CTC
的 0.85 WER 到底烂在实词还是虚词？如果实词已好、虚词拖累，说明天花板在数据（两阶段架构
合理）；如果实词也崩，说明瓶颈在训练/表征榨取（判别式预训练是正路）。本节用分拆回答。

**方法**（`scripts/gloss_split_wer.py`，2026-09-28）：
- 模型：VL48 checkpoint `ctc-vl48-cap300.pt`（§18），dev 515 全量 greedy 解码
- 桶：
  - **content**：非功能词全集（1885 参考词 / 515 样本全覆盖）
  - **function**：词典虚词集（助词/连词/介词/副词/语气词：了/的/又/也/都/就/在/把/被/
    和/与/或/吗/呢/吧/不/没/很/更/最…；217 参考词 / 193 样本）
  - **probe_strong_content**：§20 landmark 族 top-5 AUC（房子/休息/怎么样/希望/睡觉，
    特征判别力 0.959–0.988，dev 出现 10 次）
  - **probe_weak_content**：§20 landmark 族 bottom-5（相信/又/决定/走…，dev 4 次）
- 聚合：仅统计参考子序列非空的样本；WER=Σ编辑距离/Σ参考词长；token recall=匹配数/参考数
- 收据：`artifacts/metrics/part4-gloss-split-wer.json`（`test_split_read: false`）

**结果**：

| 桶 | WER | token recall | 参考词数 | 覆盖样本 |
|---|---|---|---|---|
| all | 0.9567 | 0.0438 | 2102 | 515 |
| content（实词） | 0.9533 | 0.0472 | 1885 | 515 |
| function（虚词） | 0.9862 | 0.0138 | 217 | 193 |
| probe_strong_content | 1.0000 | **0.0000** | 10 | 10 |
| probe_weak_content | 1.0000 | 0.0000 | 4 | 4 |

**结论（修正 §20.4 的第 4 点推断）**：
1. **实词识别同样崩盘**：content WER 0.9533 ≈ all 0.9567，实词 recall 仅 4.7%——模型
   并没有「先学好实词」；虚词 217 词就算全对，总 WER 也只降到 ~0.86，**主瓶颈不是虚词
   天花板，而是视觉→实词 gloss 的时序识别本身没学会**。
2. **信息与提取的剪刀差（核心）**：probe_strong_content 5 个词在特征里有 0.93–0.99 AUC
   （均值池化+线性即可判别），但 CTC 在 dev 10 次出现中 **0 命中**。特征信息充分、模型
   完全没用上 → 这是训练/架构问题，存在巨大的可榨取空间。
3. **虚词 recall 0.0138 接近零**：虚词既无特征信息（§20）又无模型利用，两阶段架构中
   LLM 补虚词（§16 A 路线 BLEU 0.671）仍是必要组件，但它**不是**当前 WER 的主因。
4. **方向的最终排序**：
   - **判别式预训练/更好时序表征现在是第一优先级**：特征里有强实词信号（0.93+ AUC）
     而 CTC 零命中，预训练（或更强的对齐架构）是把这部分信号转化为 WER 收益的最大杠杆。
   - 虚词问题退居次位：等实词识别上来了，再通过 LLM/语言模型补虚词。
   - 三路融合（SpaMo 主线）的意义重新确认：它本质是把 rgb/landmark/VL 的互补判别信号
     对齐进一个表征——前提是时序编码器能把信号榨出来，否则仍会重演「信息在、提取不出」。

## 22. 对比预训练（InfoNCE 视频↔gloss 对齐）被诊断性否定：信息上限在特征材质，不在对齐

**动机**：§21 结论「特征里有强实词信息（AUC 0.93–1.0）、CTC 却没榨出来」指向判别式预训练。
方案 A 把它具体化为：视频→gloss 词 InfoNCE 对齐，把前端权重热启动到 CTC。目标是让编码器
学到更易被时序头利用的表征。结果否定，且诊断完整。

**做法**（`src/cslr/recognition/contrastive_pretrain.py`，2026-09-28）：
- `ContrastiveEncoder`：LayerNorm → projection(256) → 长度感知均值池化 → L2 归一。其
  `normalize.*` / `projection.*` 子图与 `CTCRecognizer` 前端**结构完全一致**，可热启动。
- `GlossEmbedding`：词表 300 词的可学习嵌入（index 0 = `<unk>`）。
- Loss：multi-label InfoNCE，每样本对其含有的每个实词 token 打正，词表其余 token 为负
  （`temperature=0.07`，in-batch 无额外负采样）。
- 数据：train 4973 全量，VL48 特征 + 训练集 z-score 标准化（test 冻结，未读）。
- 预训练：40 epoch，cosine decay，AdamW，loss 5.9 → 2.84 稳定收敛，4973 样本共 145 s。
  收据 `artifacts/checkpoints/contrastive-vl48*.receipt.json`。
- 热启动：`CTCRecognizer` 载入 `copy_frontend_weights`（严格跳过形状不匹配），LSTM/classifier
  随机初始化，`--init-frontend` 接入 CTC 训练。

**结果 1 · 热启动 CTC 无改善**（`ctc-vl48-warm.pt` vs 基线 `ctc-vl48-cap300.pt`）：

| 模型 | best WER | blank_ratio | distinct(unique) | best_epoch |
|---|---|---|---|---|
| 基线 VL48（无预训练） | 0.8526 | 0.9652 | 24 | 7 |
| 热启动 VL48（预训练前端） | 0.8543 | 0.9635 | 24 | 11 |

WER / blank / distinct 在噪声内持平，坍缩依旧（hypothesis 平均长 1.74，dev 423/515 条
预测为 `<unk>`）。

**结果 2 · 预训练 encoder 的表征判别力探针**（`scripts/probe_contrastive_encoder.py`）：
用与 §20 完全相同的 per-gloss ROC-AUC 机制，测 `ContrastiveEncoder` 输出的 clip 表征
（train 4973 fit，dev 515 eval）：

| 表征 | macro-AUC |
|---|---|
| VL48 原始特征（池化，§20） | 0.6459 |
| **对比预训练 encoder（压缩到 256）** | **0.6435** |
| landmark（§20，最强族） | 0.6821 |

**结论（把瓶颈定位到特征材质，修正 §21 的方向预期）**：
1. **对比预训练没有提升判别力**：encoder AUC 0.6435 ≈ 原始 VL48 0.646（探针噪声内，甚至略低）。
   InfoNCE 只是把 2048 维的特征压缩投影到 256 维，**改变的只是维度，不是信息量**——clip 表征
   的线性判别力被原始池化特征的信息上限锁死。
2. **§21「特征是强的、只是没榨出来」需要再修正**：强实词（房子/休息/行李/告诉（我））在
   **单 gloss** 上确有 0.93–1.0 AUC，但**所有 gloss 的宏平均**只有 ~0.64。即大部分实词并不被
   特征明显区分，只有少数高频实词信息充分。模型无法靠这些少数词把序列整体识别出来。
3. **真正的瓶颈是 VL48 冻结编码器的逐帧特征材质信息不足**（均值池化线性上限 0.646，低于
   landmark 的 0.682）。下游无论用 CTC、池化+对齐还是预训练，都无法超越输入特征的信息上限。
4. **判别式预训练的价值被否定**（在本特征材质下）：它不能凭空造信息。若坚持两阶段，改变输入
   材质（更大的 VL 塔、更密帧、复合 rgb+landmark+VL 的融合特征）才是前提；在 VL48 上再堆
   对齐/预训练是无意义的。
5. **下一步的有效顺序**：先解决「输入特征信息上限」（提升 clip 判别力宏平均），再谈时序建模。
   §23 记录据此设计的三路融合或更密 VL 表征方案。

## 23. 融合判别力探针：SpaMo 路线被证伪（无互补信号可融合）

**动机**：§22 结论后，「融合 rgb+landmark+VL 互补信息」一度被列为 SpaMo 的候选杠杆。但
SpaMo 在 part3 已多次失败（token 压缩 / mT5 decoder / concat 全塌缩）。本探针要裁断的是：
那失败是**训练塌缩（可修）**还是**输入特征根本没有可融合的互补信号（无药可救）**。
答案直接决定融合分支是否还值得重试。

**做法**（`scripts/probe_fusion.py`，2026-09-28）：与 §20 同一协议——train 4973 fit OneVsRest
逻辑回归、dev 514 交集样本上逐 gloss 算 ROC-AUC。把四族**均值池化后的向量直接拼接**成 3440 维
融合表征，与各自单项并排对比（基线=最佳单项）。所有族训前 z-score。

**结果**：

| 表征 | macro-AUC |
|---|---|
| landmark（最佳单项） | **0.6915** |
| rgb | 0.6135 |
| motion | 0.5419 |
| vl48 | 0.6313 |
| **融合 landmark+rgb+motion+vl48（3440 维）** | **0.6399** |

**融合增益 = −0.052**：融合宏平均不仅没超过最佳单项（0.6915），反而比它低 5.2 个点。

**结论（裁断）**：
1. **四族特征的信息高度重叠、而非互补**。拼接后一个线性分类器得到的判别力反而低于仅用
   landmark，说明各族的可判别信号基本是同一份几何信息的重复投影，拼起来只是稀释了
   landmark 中最有效的部分。
2. **SpaMo 类多模态融合路线没有得到任何支持**，可以直接放弃、不重试。part3 的融合失败
   **是输入特征信息不足的必然结果，不是可修复的训练塌缩**——即便换成再好的时序/解码架构，
   也没有可融合的互补信号可供它利用。
3. **§22 主线被进一步锁定**：所有下游（应用不同特征、不同融合、不同对齐预训练）都无法突破
   输入特征的信息上限（宏平均 ~0.69）。唯一仍站得住的杠杆是**更换输入特征材质本身**——
   更强的视觉编码、更密的时序采样、或能编码更高层语义的表征（而不是逐帧低层几何/嵌入再加
   池化）。
4. 连同 §22 一并构成最终证据链：**问题在表征层输入的材质，已排除数据质量、模型架构、训练、
   融合、预训练五个维度**。

## 24. 免费裁决补充：池化策略（§24.1）与 LLM 容噪窗口（§24.2）

§18–§23 后，"换更强编码器"是唯一剩余杠杆。但租 GPU / 走 API 花钱，故先做两个**零成本**实验，
分别裁断两个能改写结论的免费假设：判别天花板是不是均值池化抹平的（§24.1）、以及识别误差能否
被 LLM 语义重组兜住（§24.2）。

### 24.1 池化策略探针：天花板在编码层，不在池化（`scripts/probe_pool_strategy.py`）

**动机**：此前所有探针（§20/§22/§23）都对特征做 **mean-pool 后判判别**。若判别力是被均值池化
抹掉的，则"更强时序聚合"是零成本杠杆，不需要换编码器。

**做法**：landmark 帧特征（48×368）在 60 个高频 gloss 上，用与 §20 相同协议，比较聚合策略：
mean（现状）、max、mean+std、等距采样 4/8 帧拼接（保留稀疏时序）。

**结果**：

| 策略 | 维度 | macro-AUC |
|---|---|---|
| mean（基线） | 368 | 0.6803 |
| max | 368 | 0.6708 |
| **mean+std** | 736 | **0.7132** |
| sample4（时序拼接） | 1472 | 0.6494 |
| sample8（时序拼接） | 2944 | 0.6867 |

**结论**：
1. mean-pool 确实抹掉了一点点信号（mean+std 补到 0.7132），但仅 +3 个点，属于统计噪声量级，
   远不到"识别可用"（≥0.75）的门槛。
2. **保留稀疏时序对线性判别无益甚至有害**（sample4 跌到 0.649）：滞后帧的时序特征本身不含
   额外 gloss 信息——这与 CTC 用 BiLSTM 同样建模失败互相印证，坐实时序地址不缺、特征材质缺。
3. **免费层面的判别力挖掘至此穷尽**，天花板确凿在编码层本身（§22/§23 结论不变）。

### 24.2 LLM 容噪窗口：能救乱序/漏词，救不了错词（`scripts/probe_llm_robustness.py`）

**动机**：针对"识别序列不对能否靠 LLM 重组救语义"的疑问，量化 LLM 到底能吸收多少识别噪声。

**做法**：把 dev 金 gloss 序列按四档污染（替换错词 30%/50%、替换35%+删15% 的识别型噪声、
打乱顺序），经与 §16 完全相同的 route_a 通道（Qwen2.5-1.5B，dev 子集 40 条）出中文，比 BLEU。

**结果**（BLEU-1）：

| 噪声档 | BLEU-1 | distinct | 解读 |
|---|---|---|---|
| clean（金 gloss） | 0.660 | 40/40 | 一致性校验 ✓（≈§16 0.67） |
| **shuffle（乱序）** | **0.610** | 40/40 | 几乎不掉，词序完全不是问题 |
| 替换 30% 错词 | 0.448 | 40/40 | 掉两成 |
| 替换 35% + 删 15% | 0.374 | 40/40 | 落到可用边缘 |
| 替换 50% 错词 | 0.328 | 40/40 | 逼近坍缩区 |

**结论**：
1. **LLM 的重组能力很强，但有明确边界**——分类成两类误差：
   - **可兜底**：乱序（shuffle BLEU 0.61）、漏词、句式残缺。LLM 语言先验能重排补全。
   - **不可兜底**：**换错词**。Noise 越高 BLEU 近似线性崩（0.66→0.45→0.37→0.33）。因为错误词
     把语义带偏，LLM 无从"猜回"原词，只能顺着错词编通顺但错义的句子（distinct 始终 40/40
     说明它依然流畅，只是内容错了）。
2. **识别器误差的本质是"错词"而非"乱序"**——这正中 LLM 救不了的类别。CTC 输出 425/515 条
   `<unk>`/替换词（§21 token recall 0.044），正是"换错词"型误差，所以端到端（§19 BLEU 0.005）
   被锁死，LLM 无法兜底。
3. **给识别器定了量化目标而非终极要求**：无需词级 100% 正确——只要把**词级正确率从 ~0.04 提到
   ~0.7**，LLM 就能把端到端拉到 BLEU≈0.4 的可用区（30% 错词档即 0.45）。这既松绑了识别压力，
   又反证当前 0.69 判别力（不足以保证 0.7 词正确）远不够，**仍需更强特征材质**（§22/§23）。

## 25. Kaggle 方案：换 Qwen2.5-VL-7B 塔重提特征（待执行）

§22–§24 把唯一未验证杠杆锁定在"更换输入特征材质"。更大的视觉塔（7B vs 3B）超过本机 8G 免费
算力，故借 **Kaggle T4 (16GB)** 跑，零运营成本。

**交付物**（本机已入库，待用户在浏览器执行）：
- `scripts/pack_kaggle_data.py`：把 train/dev 视频 + 精简 manifest 打成上传包，**test 天然排除**。
- `kaggle/extract_and_probe_vl7b.ipynb`：7B 视觉塔前向提特征（复用 §9.1 相同的 `sample_frames`
  + 逐帧 patch token 均值池化 → `[T,D]` float32）+ **小样本判别探针**二合一。
- `kaggle/README.md`：上传→建 notebook→贴→回传的全流程与决策表。

**关键设计：先小样本裁决，再全量**。notebook 默认 `LIMIT=40`，只提 40 train + 40 dev，用与 §20
同协议的线性探针出 `MACRO_AUC`：

| 小样本 MACRO_AUC | 含义 | 动作 |
|---|---|---|
| > 0.75 | 7B 确实带新信息（突破 0.69 上限） | 全量提 + 回传 host + 本机重训 CTC |
| 0.69–0.75 | 微弱提升 | 视预算决定，倾向不做 |
| ≈ 0.69 | 同族放大无用（换汤不换药） | 停止，转非逐帧池化的高阶时序编码 |

**要点**：
- 7B 视觉塔 `D` 可能 ≠ 3B 的 2048，`feature_size` 自动探测，CTC 头维自适应，无需改代码。
- 全程不读 test 500；收据 `test_split_read: false`。
- 若 7B 也卡在 ~0.69，则剩余方向仅为**非逐帧/非池化的高阶时序视觉编码**（如视频级联合建模），
  或**数据扩充（§P3）**——二者都不在本机免费算力内。
