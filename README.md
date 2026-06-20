# CE-CSL Dataset-Bounded Chinese Sign Language Recognition

基于 CE-CSL 数据集的受限词组/语义单元识别与中文语义重构原型。

> 本项目是受控研究原型，不是诊断、公共服务或业务决策系统。当前“受限域”指识别范围受 CE-CSL 数据集中的 `Gloss`、词组和句子标签约束，不再固定某个业务场景作为主实验范围。

## 当前状态

- 已完成项目骨架、FastAPI/Web 页面、Docker/CI、MediaPipe Holistic 特征提取、LSTM/BiLSTM/TCN/compact Transformer 模型定义、ONNX 推理接口。
- CE-CSL 已确定为最终主实验数据集；源归档保留在 `E:\Download\CE-CSL.zip`。
- `data/manifests/ce-csl.csv` 已改为使用 CE-CSL `Gloss` 字段作为训练标签来源。
- `data/manifests/ce-csl-gloss-vocab.csv` 是从 train split 生成的可复现 token-frequency 词表，默认 `min_frequency=2`。
- 已接收并验收 CE-CSL 全量 landmark 特征：5,988 个 `48 x 368 float32` 文件，SHA-256
  tree digest 为 `7c240ffa3d5493c51f0c019ad8b1d068bdc40645507d6b6ee68803a5ea634386`。
- 尚未训练可报告的真实模型。
- 无模型时接口返回 `model_unavailable`，不会伪造预测；`CSLR_DEMO_MODE=true` 只用于检查界面流程，不能写入实验结果。

## 关键文档

- [团队交接指南](docs/onboarding/team-handoff.md)
- [本机运行手册](docs/onboarding/local-runbook.md)
- [CE-CSL 最终数据集审计](docs/datasets/ce-csl-audit.md)
- [候选数据集评分](docs/datasets/candidate-scorecard.md)
- [Docker 服务审计](docs/setup/docker-service-audit.md)
- [存储迁移核查](docs/setup/storage-migration-audit.md)
- [第六周局部汇报清单](docs/reports/week-06.md)

## Windows 与 Docker

日常开发仍在 Windows、PowerShell 和 VS Code 中完成。Docker Desktop 使用 WSL 2 backend 提供统一运行环境，不要求单独准备 Linux 电脑。

所有新下载的软件、安装包、大型数据和工具缓存应放在 `D:` 盘；当前 Docker 程序和数据已迁移到 `D:\DockerDesktop`、`D:\DockerDesktopData`、`D:\DockerDesktopConfig`。CE-CSL 源归档当前保留在 `E:\Download\CE-CSL.zip`，项目工作副本为 `E:\college\FYP\data\ce-csl`。

Docker 登录不是必需项。只有拉取私有镜像、推送镜像或遇到 Docker Hub 匿名限流时才需要登录。

## 启动 Web 系统

真实模式要求存在：

```text
artifacts/exports/lstm.onnx
artifacts/exports/lstm.labels.json
```

启动：

```powershell
docker compose up --build app
```

打开：

```text
http://localhost:8088
```

健康检查：

```text
http://localhost:8088/api/v1/health
```

只检查界面流程时：

```powershell
$env:CSLR_DEMO_MODE="true"
docker compose up --build app
```

停止服务：

```powershell
docker compose down
```

## 目录职责

| 路径 | 职责 | 提交 Git |
|---|---|---|
| `app/` | FastAPI 后端和浏览器上传/摄像头页面 | 是 |
| `configs/` | 数据、特征、模型、训练配置 | 是 |
| `src/cslr/data/` | dataset adapter、manifest、Gloss token 工具 | 是 |
| `src/cslr/features/` | MediaPipe 提取、归一化、重采样、质量检查 | 是 |
| `src/cslr/models/` | LSTM、BiLSTM、TCN、Transformer | 是 |
| `src/cslr/training/` | 单标签历史 baseline 和 Gloss token 多标签训练 | 是 |
| `src/cslr/inference/` | ONNX CPU 推理、低质量/低置信拒识 | 是 |
| `src/cslr/semantic/` | 可选模板映射，当前不作为主实验默认输出 | 是 |
| `data/manifests/` | 匿名样本索引和可复现词表 | 是 |
| `data/ce-csl/` | CE-CSL 解压工作副本 | 否 |
| `data/processed/` | landmark 特征 | 否 |
| `data/clean_datas/` | 已接收并验收的本地 landmark 特征包 | 否 |
| `artifacts/checkpoints/` | PyTorch 权重 | 否 |
| `artifacts/exports/` | ONNX 权重 | 否，使用 Release |
| `docs/` | 文献、数据决策、实验和阶段报告 | 是 |

## 数据流程

CE-CSL 结构：

```text
label/{train,dev,test}.csv
video/{train,dev,test}/{Translator}/{Number}.mp4
```

标签 CSV 包含：

```csv
Number,Translator,Chinese Sentences,Gloss,Note
```

本项目主实验使用：

- `Gloss`：主要监督来源，拆分为 token 后做多标签识别。
- `Chinese Sentences`：语义解释参考，不作为完整句子闭集分类主目标。
- 官方 `train/dev/test`：保留官方划分，不用团队自录替代。

生成 manifest：

```powershell
docker compose run --rm dev build-manifest configs/datasets/ce_csl.yaml `
  --data-root data/ce-csl `
  --output data/manifests/ce-csl.csv
```

验证 manifest：

```powershell
docker compose run --rm dev validate-manifest data/manifests/ce-csl.csv
```

生成 Gloss token 词表：

```powershell
docker compose run --rm dev build-gloss-vocab data/manifests/ce-csl.csv `
  --output data/manifests/ce-csl-gloss-vocab.csv `
  --min-frequency 2
```

单视频提取：

```powershell
docker compose run --rm dev extract data/ce-csl/video/train/A/train-00001.mp4 `
  --output data/processed/ce-csl/train-00001.npy
```

全量提取：

```powershell
docker compose run --rm dev extract-manifest data/manifests/ce-csl.csv `
  --data-root data/ce-csl `
  --output data/processed/ce-csl `
  --report artifacts/logs/ce-csl-extraction.csv `
  --continue-on-error
```

不要给全量提取加 `--overwrite`，这样中断后重跑会跳过已完成样本。

## 训练与导出

默认训练配置 `configs/training.yaml` 使用 `task: gloss_token_multilabel`，损失函数为 BCEWithLogitsLoss，指标为 micro-F1、macro-F1、subset accuracy 和 per-token F1。

训练 LSTM baseline：

```powershell
docker compose run --rm dev train `
  --manifest data/manifests/ce-csl.csv `
  --features data/clean_datas/ce_csl `
  --model-config configs/models/lstm.yaml `
  --training-config configs/training.yaml `
  --output artifacts/checkpoints/lstm.pt
```

导出 ONNX：

```powershell
docker compose run --rm dev export `
  artifacts/checkpoints/lstm.pt `
  artifacts/exports/lstm.onnx
```

导出会同时生成 `lstm.labels.json`，其中包含 `task`、`labels`、`label_counts` 和 `prediction_threshold`。

## 测试

Docker：

```powershell
docker compose --profile test run --rm test
```

本地已有 Python 3.11+ 环境时：

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python scripts/check_repository_safety.py
```

## Git 协作规则

主分支为 `main`。每项工作使用短期分支：

```powershell
git switch -c feature/short-name
git add path/to/files
git commit -m "feat(scope): short description"
git push -u origin feature/short-name
```

禁止提交原始视频、完整 landmark、权重、日志、身份资料、`.env` 和密钥。ONNX 模型通过 GitHub Release 发布，不放入普通 Git 历史。

## 六周节点

1. 第 1 周：WSL/Docker、Git、数据集评分、adapter 契约。
2. 第 2 周：锁定 CE-CSL、manifest/split、单视频提取。
3. 第 3 周：全量 landmark 提取和质量报告。
4. 第 4 周：LSTM Gloss token baseline 和基础评估。
5. 第 5 周：ONNX 导出、Web 接入、低置信拒识。
6. 第 6 周：Web 端到端演示、F1、错误分析、延迟和复现说明。

报告中不把 CE-CSL 结果解释成业务意图准确率，也不把完整中文句子闭集分类作为主结果。
