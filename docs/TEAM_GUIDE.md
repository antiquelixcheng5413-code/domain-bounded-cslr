# 队友操作指南（D1 / D3 / D4）

> 目标：三台机器并行，把「我们和官方差 22pp」拆成
> **通道 / 分辨率 / 容量** 三块，各自值多少 pp。
> 我们（5060 8G）跑 P150 测通道；**队友 A 跑 D1+D3 测分辨率；队友 B 跑 D4 测容量**。
> 3060 是 Ampere（CC 8.6）⇒ **支持 bf16**，我们的 `--amp-bf16` 可直接用。

---

## 0. 你要做什么（一句话版）

| 你是 | 先做 | 再做 |
|---|---|---|
| **队友 A** | **D1**（16 分钟，轻量包即可）→ 结果发群里 | **D3**：`bash tools/blank_gov/run_d3_vac224.sh smoke` 通过后正式跑 |
| **队友 B** | **D4**：**必须先冒烟** `run_d4_tfnet224.sh smoke` | 冒烟过了再正式跑 55 轮 |

D1 先做是因为：它会告诉我们"160 分辨率到底损失了多少"，**如果损失很小，D3 那 30 小时就能直接省掉**。

---

## 1. 拿代码

```bash
git clone <团队仓库地址> cslr        # 团队仓库（不是个人仓库）
cd cslr
git checkout experiment/blank-governance-p0-p38   # 我们的工作分支
```

⚠️ 分支名是 `experiment/blank-governance-p0-p38`，主分支上**没有**这些脚本。

---

## 2. 装环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision          # 要匹配你的 CUDA（3060 用 CUDA 11.8 或 12.x 都行）
pip install opencv-python numpy tqdm
```

然后自检（**先跑这个，它会告诉你缺什么**）：

```bash
python3 tools/blank_gov/check_env.py
```

自检会打印：GPU 型号/显存、bf16 是否支持、dev/train 帧数够不够、csv 在不在、磁盘够不够。
**全部 ✅ 再往下走。**

---

## 3. 放数据

从我们这拿两个包（移动硬盘直拷最快，网盘传 22GB 很慢）：

| 包 | 大小 | 解压到 |
|---|---|---|
| `cecsl-dev-pkg.tar` | ~2.7 GB | `artifacts/official_rgb/dev/` |
| `cecsl-train-pkg.tar` | ~23 GB | `artifacts/official_rgb/train/`（只有 D3/D4 需要） |
| `TFNet-CE-CSL-CSLDaily-32.46.pth` | 880 MB | 任意位置，D1 要用（记下路径） |

```bash
cd artifacts/official_rgb
tar xf /path/to/cecsl-dev-pkg.tar      # 解出 dev/
tar xf /path/to/cecsl-train-pkg.tar    # 解出 train/（D3/D4 才需要）
```

目录结构应该是：
```
artifacts/official_rgb/dev/<translator>/<sample_id>/00000.jpg ...
artifacts/official_rgb/train/<translator>/<sample_id>/00000.jpg ...
```

期望帧数：dev **95891**、train **930841**（`check_env.py` 会核对）。

---

## 4. D1：官方权重消融（队友 A，16 分钟）

**作用**：在官方 TFNet 权重上，直接量出分辨率和通道各值多少 pp。

```bash
./venv/bin/python -u tools/blank_gov/p156_official_ablation.py 0 fast
```

⚠️ 脚本里 `CK` 路径默认写的是我们机器的位置，你需要改成你的权重路径：
```python
CK = Path("/你的/路径/TFNet-CE-CSL-CSLDaily-32.46.pth")
```

**期望输出**（4 个组合的 dev WER）：
```
224+RGB  33.92   ← 基线，应能复现这个数（复现不出说明环境/数据有问题）
160+RGB  ??      ⇒ 与 33.92 的差 = 分辨率代价
224+BGR  ??      ⇒ 与 33.92 的差 = 通道代价
160+BGR  ??      ⇒ 两者叠加 = 我们的取舍总代价
```

**⚠️ 拿到结果先发群里再决定要不要开 D3。**

---

## 5. D3 / D4：训练

**一律先冒烟**（2 轮、40 条，几分钟），确认能跑通再上 55 轮：

```bash
# 队友 A
bash tools/blank_gov/run_d3_vac224.sh smoke
bash tools/blank_gov/run_d3_vac224.sh          # 冒烟过了才跑这个（约 30h）

# 队友 B
bash tools/blank_gov/run_d4_tfnet224.sh smoke   # 必须做
bash tools/blank_gov/run_d4_tfnet224.sh         # 约 50h
```

日志在 `artifacts/logs_D3_vac224.log` / `artifacts/logs_D4_tfnet224.log`。

**如果 OOM**（日志里有 `out of memory`）：
```bash
BATCH=1 bash tools/blank_gov/run_d3_vac224.sh    # 降 batch 重跑
```

---

## 6. 判据（很重要，别比错）

- **基线**：P123，dev **55.84%**（我们的 VAC+160+BGR 模型）
- **主指标**：`official_wer.py` 的 `WER_official`（语料级）
- ⚠️ **只能配对 ep32 以后**。P123 的 ep1–31 是坏解码截断的**虚高值**
  （实测 ep31→ep32 有 **−15.70pp 的跳变**：74.74 → 59.04）。
  拿早期轮次跟我们比会得出完全错误的结论。
- 判定阈值：改善 ≥ **1.556pp** 才算"可判有效"（小于这个在噪声里）。

---

## 7. 约定

1. **test 冻结**：任何人都不能在 test 上选/调超参。
2. **Git**：开自己的分支 `team/<名字>-<任务>`；**只有项目负责人说"推送"才推**，平时本地 commit 就行。
3. **训练脚本一律带 `-u`**（无缓冲）—— 否则进程被杀时日志一片空白，我们踩过两次。
4. 出结果（尤其是 D1）**先发群里**，可能让我们省掉几十小时。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: torch` | 用了系统 python3。必须用 `./venv/bin/python` |
| 训练跑着跑着没日志、GPU 归零 | 没加 `-u`（脚本里已加）；或 WSL 会话回收 ⇒ 重跑 |
| `Cannot allocate memory`（打包/压缩时） | 系统内存小，别用 gzip 压大文件；jpg 本来就是压缩的 |
| 显存爆 | 降 `BATCH=1`；D4 已经是 batch 1，再爆就改 `--img-size 160` |
| dev 帧数对不上 | 解包路径错了，应为 `artifacts/official_rgb/dev/<translator>/<sid>/` |
| 结果跟我们差很多 | 先确认 `--fix-rgb-order` 有没有带上（脚本里已带） |
