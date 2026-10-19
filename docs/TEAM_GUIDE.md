# 队友操作指南（D1 / D3 / D4）

> 目标：三台机器并行，把「我们和官方差 22pp」拆成
> **通道 / 分辨率 / 容量** 三块，各自值多少 pp。
> 我们（5060 8G）跑 P150 测通道；**队友 A 跑 D1+D3 测分辨率；队友 B 跑 D4 测容量**。
> 3060 是 Ampere（CC 8.6）⇒ **支持 bf16**，我们的 `--amp-bf16` 可直接用。

---

## 0. 你要做什么（一句话版）

| 你是 | 先做 | 再做 |
|---|---|---|
| **队友 A** | **D1**（16 分钟）→ 结果发群里 | **D3**：`run_d3_vac224.sh smoke` 通过后正式跑 |
| **队友 B** | **D4**：**必须先冒烟** `run_d4_tfnet224.sh smoke` | 冒烟过了再正式跑 55 轮 |

**D1 先做，它同时是数据一致性检验**（见第 3 节）—— 跑不出 33.92 就别往下走。

---

## 1. 拿代码

```bash
git clone <团队仓库地址> cslr
cd cslr
git checkout experiment/blank-governance-p0-p38   # 我们的工作分支
```

⚠️ 主分支上**没有**这些脚本，必须切到这个分支。

---

## 2. 装环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision          # 匹配你的 CUDA（3060 用 CUDA 11.8 / 12.x 均可）
pip install opencv-python numpy tqdm
python3 tools/blank_gov/check_env.py   # 自检：GPU/显存/bf16/帧数/csv/磁盘
```

---

## 3. 数据（你们已有数据集，重点是**验证一致性**）

### 3.1 🔴 先用 D1 做数据一致性门禁（必做）

不管你们的帧是怎么来的，**先跑 D1 的第一个组合**：

```bash
./venv/bin/python -u tools/blank_gov/p156_official_ablation.py 0 fast
```

看输出里的 `224_RGB`（官方权重的原配配置）：

| 结果 | 含义 |
|---|---|
| **≈ 33.92** ✅ | 你们的帧与我们/官方等价 ⇒ **可以继续往下跑** |
| 差很多 ❌ | 数据不一致（抽帧方式/分辨率/帧数不同）⇒ **先停下来查，别浪费 30 小时** |

这一步只要十几分钟，却能避免后面几十小时白跑。

### 3.2 如果你们有的是**原始视频**（还没抽帧）

用仓库里**未修改的** `p77_extract_rgb.py` 抽，然后训练时**一定带上 `--fix-rgb-order`**：

```bash
./venv/bin/python tools/blank_gov/p77_extract_rgb.py --split train --workers 6
./venv/bin/python tools/blank_gov/p77_extract_rgb.py --split dev   --workers 6
```

### 3.3 🔴 通道规则（混用就又错了，务必看）

我们实测发现：抽帧链路里存在 R/B 通道错位（官方是 RGB，我们抽出来是 BGR 语义）。
修正方式是**一对绑定的开关**，两种合法组合：

| 抽帧脚本 p77 | 训练/评估参数 | 结果 |
|---|---|---|
| **原始版**（有 `cvtColor(BGR2RGB)`，即当前仓库里的） | **带** `--fix-rgb-order` | ✅ 真 RGB |
| **修正版**（已去掉 `cvtColor`，任务 A1） | **不带** `--fix-rgb-order` | ✅ 真 RGB |
| 原始版 + 不带 | ❌ BGR（错） |
| 修正版 + 带 | ❌ BGR（错，又反过来） |

**默认仓库里的 p77 是原始版**，所以 `run_d3/d4_*.sh` 里都带了 `--fix-rgb-order` ——
**除非你们改了 p77，否则不要去掉这个参数。**

> 如果有人领了任务 A1（改 p77），改完必须通知所有人，并同步去掉脚本里的
> `--fix-rgb-order`，否则两边结果会差 30pp 量级。

### 3.4 期望的目录与帧数

```
artifacts/official_rgb/dev/<translator>/<sample_id>/00000.jpg ...
artifacts/official_rgb/train/<translator>/<sample_id>/00000.jpg ...
```
dev **95891** 帧、train **930841** 帧（`check_env.py` 会核对）。

---

## 4. D1：官方权重消融（队友 A，约 16 分钟）

见 3.1。脚本里 `CK` 路径要改成你们的权重位置：
```python
CK = Path("/你的/路径/TFNet-CE-CSL-CSLDaily-32.46.pth")
```

**产出**（dev WER）：
```
224+RGB  33.92   ← 基线，必须复现
160+RGB  ??      ⇒ 与 33.92 之差 = 分辨率代价
224+BGR  ??      ⇒ 与 33.92 之差 = 通道代价
160+BGR  ??      ⇒ 两者叠加 = 我们的取舍总代价
```
⚠️ 这是"推理时换输入"的**上界**；从头训练会适应，真实损失更小。
**拿到结果先发群里** —— 若分辨率代价很小，D3 那 30 小时可以直接省掉。

---

## 5. D3 / D4：训练

**一律先冒烟**（2 轮、40 条）：

```bash
bash tools/blank_gov/run_d3_vac224.sh smoke
bash tools/blank_gov/run_d3_vac224.sh          # 冒烟过了才跑（约 30h）

bash tools/blank_gov/run_d4_tfnet224.sh smoke   # 必须做
bash tools/blank_gov/run_d4_tfnet224.sh         # 约 50h
```

OOM 时降 batch：
```bash
BATCH=1 bash tools/blank_gov/run_d3_vac224.sh
```

日志：`artifacts/logs_D3_vac224.log` / `artifacts/logs_D4_tfnet224.log`。

---

## 6. 判据（很重要，别比错）

- **基线**：P123，dev **55.84%**（我们的 VAC+160+BGR 模型）
- **主指标**：`official_wer.py` 的 `WER_official`（语料级）
- 🔴 **只能配对 ep32 以后**。P123 的 ep1–31 是坏解码截断的**虚高值**
  （实测 ep31→ep32 有 **−15.70pp 跳变**：74.74 → 59.04）。
  拿早期轮次比会得出完全错误的结论 —— 我们今天就差点被它骗了一次。
- 判定阈值：改善 ≥ **1.556pp** 才算"可判有效"，小于这个都在噪声里。

---

## 7. 约定

1. **test 冻结**：任何人都不能在 test 上选/调超参。
2. **Git**：开自己的分支 `team/<名字>-<任务>`；**只有项目负责人说"推送"才推**。
3. **训练脚本一律带 `-u`**（无缓冲）—— 否则进程被杀时日志空白（我们踩过两次）。
4. 出结果（尤其 D1）**先发群里**。
5. 改 `p77` 或 `p78` 前**先在群里说一声**（我们这边有实验在依赖它们）。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: torch` | 用了系统 python3，必须用 `./venv/bin/python` |
| 跑着跑着没日志、GPU 归零 | 没加 `-u`（脚本已加）；或会话回收 ⇒ 重跑 |
| D1 的 224+RGB 不是 33.92 | 数据不一致 ⇒ 查抽帧方式与帧数，别继续 |
| 结果和我们差 30pp 量级 | 通道开关用反了 ⇒ 回看 3.3 |
| 显存爆 | `BATCH=1`；D4 已 batch 1，再爆就 `--img-size 160` |
| 打包/压缩时 `Cannot allocate memory` | 系统内存小，别 gzip 大文件（jpg 已压缩） |
