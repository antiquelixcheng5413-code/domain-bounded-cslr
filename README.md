# Domain-Bounded Chinese Sign Language Recognition

面向医院前台场景的受限域中国手语识别与中文语义重构原型。

> 这是受控研究原型，不是医疗设备，也不用于诊断、分诊或紧急决策。公开数据集是
> 主要训练与评估来源；团队录制仅用于界面和流程验证。

## 当前状态

- 已完成项目骨架、统一数据契约、MediaPipe Holistic 特征提取、四类时序模型、
  ONNX 推理接口、规则语义模块、FastAPI/Web 页面、Docker 和 CI。
- NationalCSL-DP 已成为暂定主数据集；最终锁定仍需检查一个完整图片帧归档。
- 已校验官方标签表和一个 Participant 08 原视频归档，并成功提取一个官方样本：
  133 个源帧、91.73% 有效帧、输出 `48 x 368` 特征。
- 尚未训练或发布模型。默认运行时会返回 `model_unavailable`，不会伪造预测。
- `CSLR_DEMO_MODE=true` 只用于检查网页流程，结果不得写入实验报告。

数据集评分和审计记录：

- [候选数据集评分](docs/datasets/candidate-scorecard.md)
- [NationalCSL-DP 审计](docs/datasets/nationalcsl-dp-audit.md)

## Windows 10 环境

项目日常仍在 Windows、PowerShell 和 VS Code 中操作。Docker Desktop 使用 WSL 2
作为后台运行环境，不要求另备 Linux 电脑。

### 1. 确认虚拟化

打开“任务管理器 → 性能 → CPU”，确认“虚拟化”为“已启用”。若未启用，需要进入
BIOS/UEFI 打开 Intel VT-x 或 AMD-V。

### 2. 安装 WSL 2

以管理员身份打开 PowerShell：

```powershell
wsl --install
```

命令完成后重启电脑，再运行：

```powershell
wsl --status
```

### 3. 安装 Docker Desktop

安装 Docker Desktop for Windows，保留 “Use WSL 2 based engine” 设置。启动后验证：

```powershell
docker version
docker run --rm hello-world
```

官方说明：

- <https://learn.microsoft.com/windows/wsl/install>
- <https://docs.docker.com/desktop/setup/install/windows-install/>

## 启动 Web 系统

真实模式要求 `artifacts/exports/lstm.onnx` 和同目录的
`lstm.labels.json`：

```powershell
docker compose up --build app
```

打开 <http://localhost:8088>，健康检查位于
<http://localhost:8088/api/v1/health>。若端口被占用，可先设置
`$env:CSLR_PORT="其他端口"`。

只检查界面流程时：

```powershell
$env:CSLR_DEMO_MODE="true"
docker compose up --build app
```

页面会持续标记这是 demo 结果。停止服务：

```powershell
docker compose down
```

## 目录职责

| 路径 | 职责 | 是否提交 Git |
|---|---|---|
| `app/` | FastAPI 后端及浏览器摄像头页面 | 是 |
| `configs/` | 数据、特征、模型、训练和医院意图配置 | 是 |
| `src/cslr/data/` | 数据集 adapter 和 manifest 校验 | 是 |
| `src/cslr/features/` | MediaPipe 提取、归一化、重采样和质量检查 | 是 |
| `src/cslr/models/` | LSTM、BiLSTM、TCN、Transformer | 是 |
| `src/cslr/training/` | 特征数据集和训练流程 | 是 |
| `src/cslr/inference/` | ONNX CPU 推理与拒识 | 是 |
| `src/cslr/semantic/` | 受限意图到中文模板 | 是 |
| `data/manifests/` | 匿名样本索引 | 是 |
| `data/splits/` | 固定随机及 signer-independent 划分 | 是 |
| `data/raw/` | 公开数据原始视频 | 否 |
| `data/processed/` | landmark 特征 | 否 |
| `artifacts/checkpoints/` | PyTorch 权重 | 否 |
| `artifacts/exports/` | ONNX 权重 | 否，使用 Release |
| `artifacts/metrics/` | 用于论文的紧凑指标和图表 | 是 |
| `docs/` | 文献、数据决策、实验和阶段报告 | 是 |

## 数据流程

NationalCSL-DP 是暂定主数据集，但完整图片归档结构尚待核验，因此 adapter 仍不能
写死未经观察的目录。数据集配置位于
`configs/datasets/nationalcsl_dp.yaml`。每个 adapter 最终统一生成：

```csv
sample_id,video,label,signer,session,split
```

验证 manifest：

```powershell
docker compose run --rm dev validate-manifest data/manifests/example.csv
```

提取单个视频的 48 帧特征：

```powershell
docker compose run --rm dev extract data/raw/DATASET/example.mp4 `
  --output data/processed/example.npy
```

当前每帧输出 368 维：

- 双手 126 维。
- 上半身 32 维。
- 选定面部点 24 维。
- 四个模态存在掩码。
- 182 维一阶运动差分。

有效帧比例低于 80% 时，推理拒绝输出意图。

## 训练

先为 manifest 中每个 `sample_id` 生成
`data/processed/<sample_id>.npy`，再运行：

```powershell
docker compose run --rm dev train `
  --manifest data/manifests/selected-dataset.csv `
  --features data/processed `
  --model-config configs/models/lstm.yaml `
  --output artifacts/checkpoints/lstm.pt
```

训练输出必须记录数据集版本、split、seed 和 Git commit。当前 runner 提供受控基线；
正式实验还需在数据集锁定后补充类别权重和 signer-independent 汇总。

导出 ONNX：

```powershell
docker compose run --rm dev export `
  artifacts/checkpoints/lstm.pt `
  artifacts/exports/lstm.onnx
```

导出命令会同时生成 `lstm.labels.json`，推理服务要求这两个文件同时存在。

## 测试

Docker：

```powershell
docker compose --profile test run --rm test
```

本地已有 Python 3.11 环境时：

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python scripts/check_repository_safety.py
```

## Git 协作

主分支为 `main`。每项工作使用短期分支：

```powershell
git switch -c feature/dataset-adapter
git add .
git commit -m "feat: add selected dataset adapter"
git push -u origin feature/dataset-adapter
```

通过 Pull Request 合并。分支建议：

- `feature/...`：功能。
- `experiment/...`：实验。
- `fix/...`：修复。
- `docs/...`：文档。

禁止提交原始视频、landmark 全量数据、参与者身份、同意书、密钥、日志和权重。
ONNX 模型随版本标签上传到私有 GitHub Release。

## 六周节点

1. 第1周：WSL/Docker、Git、数据集评分、adapter 契约。
2. 第2周：锁定数据集、manifest/split、单视频提取。
3. 第3周：批量提取及质量报告。
4. 第4周：LSTM 训练和基础评估。
5. 第5周：signer-independent 初测、ONNX 和语义模板。
6. 第6周：Web 端到端演示、F1、混淆矩阵、延迟和复现说明。

详细检查表见 [docs/reports/week-06.md](docs/reports/week-06.md)。
