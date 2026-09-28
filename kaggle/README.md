# Kaggle 提更强特征（Qwen2.5-VL-7B）· 使用说明

背景：§22/§23 已把 `0.69 macro-AUC` 定义为本机免费算力下的特征判别上限。要破它
必须换更大的视觉塔，从原始像素重提特征——本机 8G 装不下，故借 Kaggle T4 (16GB)。
本目录交付零操作你账号的产物：打包脚本 + 可上传 notebook。

## 文件

| 文件 | 作用 |
|---|---|
| `scripts/pack_kaggle_data.py` | 把 train/dev 视频 + 精简 manifest 打成上传包（**不含 test**）|
| `kaggle/extract_and_probe_vl7b.ipynb` | 在 Kaggle 上提取 7B 视觉塔特征 + 小样本判别探针二合一 |

## 步骤

### 1. 本机打包（我来跑，或你手动）

```bash
cd /home/su127/FYP/domain-bounded-cslr
python scripts/pack_kaggle_data.py \
    --data-root data/raw/CE-CSL --out /tmp/kaggle_pkg --zip
```

产出 `/tmp/kaggle_pkg/`（含 `manifest.csv` 与 `video/`）与 `package.zip`（约 10GB）。

### 2. 上传到 Kaggle

- https://www.kaggle.com/datasets/new → 填标题 `ce-csl-videos` →
  上传 `package.zip`（或整个 `kaggle_pkg` 内容）→ 设为 Private。

### 3. 建 notebook

- https://www.kaggle.com/ → New Notebook。
- 右上 **Settings → Internet** 打开（需联网拉 Qwen2.5-VL-7B）。
- **Add Input** → 选刚上传的 `ce-csl-videos` Dataset（挂到 `/kaggle/input`）。
- GPU：Settings → Accelerator → **GPU T4 x2**。

### 4. 贴入 `extract_and_probe_vl7b.ipynb`

默认 `LIMIT=40`，先做**小样本探针**：

- 跑完看输出 `MACRO_AUC = ?`。
- 若 **> 0.75**（突破 0.69 天花板）→ 绿色通行：把 `LIMIT` 改 `None`，
  重跑提取 + 探针 cell + 末尾 zip cell，下载 `ce-csl-qwenvl7b.zip` 回来。
- 若 **约 0.69 或更低** → 红色终止：**不要再花全量算力**，7B 换汤不换药，
  结论是"同族放大无收益"，回来告诉我们。

### 5. 回传特征

把下载的 `ce-csl-qwenvl7b.zip` 解压到 host：

```bash
mkdir -p artifacts/part3_features_vl7b/{train,dev}
unzip ~/Downloads/ce-csl-qwenvl7b.zip -d /tmp/vl7b
# 按 sample_id 的前缀 split 归位，或用 notebook 里已按 train/dev 分好的文件
```

之后用与 landmark 完全相同的 `cslr.recognition train/evaluate` 跑，对比 WER。

## 纪律提醒

- 全程**不读取 test 500**（打包脚本只收 train/dev；notebook 只读这两 split）。
- 7B 的视觉塔特征维度 `D` 可能与 3B 不同（`feature_size` 会自动探测）——
  CTC 头维随它自适应，无需改代码。
- 收据都带 `test_split_read: false`，符合仓库规范。

## 预期决策表

| 小样本 MACRO_AUC | 含义 | 动作 |
|---|---|---|
| > 0.75 | 7B 确实带新信息 | 全量提 + 回传 + 本机重训 |
| 0.69–0.75 | 微弱提升 | 视预算决定是否全量，倾向不做 |
| ≈ 0.69 | 同族放大无用 | 停止，转考虑非逐帧池化的高阶时序编码 |