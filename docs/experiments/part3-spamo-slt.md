# Part 3 · SpaMo 风格 SLT 交接文档

更新时间：2026-09-17
分支：`feature/part3-spamo-scaffold`
本文件覆盖 Part 3 的 **Phase-1（代码骨架 + CPU 冒烟）** 交付，正式实验见 Phase-2。

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