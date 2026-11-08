# 阶段报告：前端可读化管线（TFNet 零样本 → Qwen2.5-1.5B → 中文）

日期：2026-10-28
分支：experiment/blank-governance-p0-p38
提交：fc9be7c

## 做了什么
把「视频 → 识别(gloss) → 大模型 → 通顺中文」的非 oracle 端到端链接入演示前端：
- 复用 scripts/route_a_gloss_llm.py 的 prompt 与 gloss 预处理（去首尾标点、空格连接）。
- 新增 app/backend/translate_service.py：Qwen2.5-1.5B-Instruct 单例惰性加载，
  权重路径 /mnt/d/part3_models/ms_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master。
- routes.py: /api/v1/predict 增加 translate 表单字段；命中后调用翻译器，返回加 translation 字段。
- 前端：新增「翻译为中文（Qwen2.5）」勾选框（默认开），主展示改为可读中文，保留原始 Gloss 明细。

## 验证结果（GPU, cuda）
TFNet + Qwen 共存占显存 ~6.3/8.2GB，留约 1.9GB 余量，当前负载 OK。
dev 随机样本实测（tfnet + translate=on）：
- 迷茫/.        → 迷茫
- 称赞/.        → 赞美（近义，合理）
- 油/.          → 油
- 早前单测 中国/粮食/进口/国家/不是/. → 中国不是粮食进口国家。（整句）
vac + translate 优雅降级：gloss 为 。 时 translation 为空，不报错。

## 结论
- 这是「可读性」问题的实际解法：LLM 把 gloss 变通顺中文，绕开 VAC 坍缩。
- 注意：前端可读结果来自「发布 TFNet 零样本识别 + 作者权重 + Qwen 翻译」，并非我们训练的模型。
- 我们训练的 VAC(P150) 在翻译链下仍常坍缩成 。，翻译为空——再次坐实 VAC 不可用。

## 待办
- 显存余量偏紧（~1.9GB）；若大视频 OOM，可把 TFNet 转 fp16 或 Qwen 移 CPU。
- 未推送远程（等用户说「推送」）。
