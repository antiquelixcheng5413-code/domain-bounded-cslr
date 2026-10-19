# 队友操作指南（D3 / D4）

> **v6 · 2026-10-19 13:25 更新**（变更记录见文末第 10 节）
>
> 目标：两台机器并行，把「我们和官方差 22pp」拆成
> **通道 / 分辨率 / 容量** 三块，各自值多少 pp。
>
> 🔧 **v4 新增**：`RESUME=1` 续跑开关（见 5.1）——跑挂了**别直接重跑脚本**，否则从第 1 轮重来。
> 🟢 **D1 我们已经跑完了**（结果见第 4 节）——**你们不用再跑 D1**，直接做 D3 / D4。
> 3060 是 Ampere（CC 8.6）⇒ **支持 bf16**，我们的 `--amp-bf16` 可直接用。

---

## 0. 你只要做什么（一句话版）

| 你是 | 任务 | 命令 |
|---|---|---|
| **队友 A** | **D3**：VAC + 224 | `run_d3_vac224.sh smoke` → 通过后再正式 55 轮 |
| **队友 B** | **D4**：TFNet + 224 | `run_d4_tfnet224.sh smoke`（**必须做**）→ 再正式 55 轮 |

**不要碰 D1**，我们跑完了，而且它需要一份 880MB 的官方权重（你们没有，也不必传）。

### 0.1 先弄清楚：这套实验到底在测什么

**问题**：我们训出来的模型 dev WER **55.84%**，官方发布的权重是 **33.92%**，差 22pp。
我们把可能的来源拆成三块——**通道 / 分辨率 / 容量**，你们负责后两块。

四个配置构成一个「**相邻只差一个变量**」的阶梯，所以每一个减法都合法：

| 配置 | 架构 | 分辨率 | 通道 | 谁在跑 | 用途 |
|---|---|---|---|---|---|
| P123（旧） | VAC | 160 | BGR | 我们，已跑完 | 旧基线 55.84 |
| **P150** | VAC | 160 | **RGB** | 我们，进行中 | **新基线**（所有减法的被减数）|
| **D3** ← 队友 A | VAC | **224** | RGB | **你** | D3 − P150 = **分辨率值多少 pp** |
| **D4** ← 队友 B | **TFNet** | 224 | RGB | **你** | D4 − D3 = **容量值多少 pp**；D4 − 33.92 = **我们能否复现官方** |
| 官方权重 | TFNet | 224 | RGB | 已发布 | 33.92 |

⇒ 你们跑的不是「再训一个模型」，而是给这张表补上两个空格；
**单独看你自己那个数字没有意义**，必须别人那格出来才能相减。

**四个名词速览**（够你读懂日志就行）：

| 名词 | 是什么 |
|---|---|
| **WER** | 词错误率 = 100% × (插入+删除+替换) / 参考词数，越小越好。用 `tools/blank_gov/official_wer.py` 算（gloss 分词口径），**不要自己按空格切** |
| **VAC** | resnet18 逐帧提特征 → 时序卷积 → BiLSTM → CTC。35M 参数，我们的默认架构 |
| **TFNet** | 换成 resnet34 + 3D 运动注意力（resnet34MAM），80M 参数，官方的架构。**是 3D 主干，首次跑容易踩坑 ⇒ 必须先冒烟** |
| **CTC + greedy** | 输出是「每帧一个字符概率」，greedy 取每帧最大后折叠去重。我们测过 beam search 与 greedy 等价，所以用 greedy |

---

## 1. 拿代码

```bash
git clone https://github.com/antiquelixcheng5413-code/domain-bounded-cslr.git cslr
cd cslr
git checkout experiment/blank-governance-p0-p38   # ← 我们的工作分支，必做
```

⚠️ **两个坑**：
1. **主分支上没有这些脚本**，`clone` 后不 `checkout` 会拿到空的 `tools/blank_gov/`。
2. 这个仓库有两个 remote，**它们的 `main` 还不一样**（个人仓库 `origin` 的 main 是 `ffbc11d`、
   团队仓库 `shared` 的 main 是 `3f1646e`）⇒ 上面给的是**团队仓库**地址，别 clone 错，
   也别在主分支上找脚本。

```bash
git branch --show-current                  # 确认输出：experiment/blank-governance-p0-p38
ls tools/blank_gov/run_d3_vac224.sh        # 确认文件在
```

---

## 2. 装环境

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision          # 匹配你的 CUDA（3060 用 CUDA 11.8 / 12.x 均可）
pip install opencv-python numpy tqdm
python3 tools/blank_gov/check_env.py   # 自检：GPU/显存/bf16/帧数/csv/磁盘
```

### 2.1 怎么读 `check_env.py` 的输出

它分 6 段，符号含义：**✅ 通过 / ⚠️ 提醒 / ❌ 缺东西**。

| 段 | 必须看到什么 | 不合格怎么办 |
|---|---|---|
| [1] Python 与 torch | `Compute Capability 8.x ⇒ bf16 支持` | 若显示 CC < 8.0（不支持 bf16）⇒ 去掉 `--amp-bf16`，会慢但能跑 |
| [2] 依赖 | cv2 / numpy / tqdm 三个全 ✅ | 缺哪个 `pip install` 哪个 |
| [3] 数据 | `dev/ 帧数 95891`、`train/ 帧数 930841` | 帧数对不上先别往下跑，回 3.1 |
| [4] CSV | 找到 train/dev/test 三个 csv | 没找到 ⇒ 数据没放对位置 |
| [5] 官方权重 | ⚠️ **这是正常的，不用管** | 只有 D1 需要它；D3/D4 不需要 |
| [6] 磁盘 | 剩余 > 30 GB | 训练会写 ~1GB 的 ckpt（best + last）|

⚠️ 它页脚会印一行 D1 命令——**那是留档的，你们不要执行**。

---

## 3. 数据一致性（没有官方权重，改用帧数门禁）

### 3.1 门禁命令

```bash
python3 tools/blank_gov/check_env.py
```

**必须通过的硬指标**：dev **95891** 帧、train **930841** 帧，目录约定：

```
artifacts/official_rgb/dev/<translator>/<sample_id>/00000.jpg ...
artifacts/official_rgb/train/<translator>/<sample_id>/00000.jpg ...
```

帧数对不上 ⇒ **抽帧方式或帧数跟我们不一样，先停下来查**，别直接往下跑 30 小时。
（原 D1 门禁需要 880MB 官方权重，你们跑不了；帧数 + 通道规则合起来已足够排除绝大多数不一致。）

### 3.2 🔴 通道规则（混用就彻底错了，务必细看）

我们实测发现：抽帧链路里存在 R/B 通道错位（官方是 RGB，抽出来是 BGR 语义）。
修正方式是**一对绑定的开关**，只有两种合法组合：

| 抽帧脚本 p77 | 训练/评估参数 | 结果 |
|---|---|---|
| **原始版**（有 `cvtColor(BGR2RGB)`，即当前仓库里的） | **带** `--fix-rgb-order` | ✅ 真 RGB |
| **修正版**（已去掉 `cvtColor`） | **不带** `--fix-rgb-order` | ✅ 真 RGB |
| 原始版 + 不带 | ❌ BGR（错） |
| 修正版 + 带 | ❌ BGR（错，又反过来） |

**当前仓库里的 p77 是原始版**，所以 `run_d3/d4_*.sh` 里都带了 `--fix-rgb-order`
——**除非你们改了 p77，否则不要去掉这个参数。**

> 如果有人改了 p77，必须通知所有人，并同步去掉脚本里的 `--fix-rgb-order`，
> 否则两边结果会差 **30pp 量级**（我们的 D1 实测：通道代价 +63.79pp）。

### 3.3 如果你们有的是**原始视频**（还没抽帧）

用仓库里**未修改的** `p77_extract_rgb.py`，然后训练时**一定带上 `--fix-rgb-order`**：

```bash
./venv/bin/python tools/blank_gov/p77_extract_rgb.py --split train --workers 6
./venv/bin/python tools/blank_gov/p77_extract_rgb.py --split dev   --workers 6
```

---

## 4. D1（我们已跑完，仅供你理解背景，**你们不用跑**）

我们用自己的官方权重在 dev 516 上跑了四个组合（greedy、语料级 WER）：

| 组合 | WER | vs 224+RGB |
|---|---|---|
| **224+RGB**（官方原配） | 33.92 | — |
| 160+RGB | 88.63 | **+54.71** |
| 224+BGR | 97.71 | **+63.79** |
| 160+BGR | 98.35 | **+64.43** |

⇒ 分辨率代价 +54.71pp、通道代价 +63.79pp。
⚠️ 这是「**推理时换输入**」的**上界**，从头训练会适应，不等于真实训练损失；
而且**不可线性相加**（生成率掉到 3.2% 已贴 100% 天花板）。

---

## 5. D3 / D4：训练

**一律先冒烟**（2 轮、40 条）：

```bash
bash tools/blank_gov/run_d3_vac224.sh smoke
bash tools/blank_gov/run_d3_vac224.sh            # 冒烟过了才跑正式（约 30h）

bash tools/blank_gov/run_d4_tfnet224.sh smoke    # 必须做
bash tools/blank_gov/run_d4_tfnet224.sh          # 约 50h
```

OOM 时降 batch：
```bash
BATCH=1 bash tools/blank_gov/run_d3_vac224.sh
```

日志会自动追加到 `artifacts/logs_D3_vac224.log` / `artifacts/logs_D4_tfnet224.log`
（脚本内部用 `tee -a` 追加，多次续跑不会覆盖历史）。

### 🔴 5.1 中断了怎么办：一定要用 RESUME=1 续跑

`p78` 的 `--resume` **默认是空的，不会按 tag 自动找 ckpt**。
⇒ 跑挂了直接重跑脚本 = **从第 1 轮重来**，前面几十小时全丢。

```bash
# 中断后（不管是 OOM、被杀、还是会话回收），用这条原地续跑：
RESUME=1 bash tools/blank_gov/run_d3_vac224.sh
RESUME=1 bash tools/blank_gov/run_d4_tfnet224.sh
```

脚本会自动做三件事：
1. 找 `artifacts/checkpoints/<tag>-last.pt`；
2. **校验文件大小**（<100MB 视为残file，直接拒绝并提示，因为 CUDA sticky error
   会写出 18KB 的坏权重）；
3. 把 `--resume` 拼进去，并从 ckpt 里恢复 epoch / optimizer / scheduler
   ⇒ **lr 里程碑不会被续训打乱**。

⚠️ 会话回收是常态不是意外：我们这边同一个训练就被无 traceback 地杀过两次。
⇒ 建议用 `tmux` / `screen` 挂着跑，或者保持终端开着；但**无论如何先记住 RESUME=1**。

### 5.2 冒烟到底看什么（通过判据）

冒烟只训 40 条跑 2 轮，跑完同时满足这四条才算过：

1. 脚本最后打印 **退出码 0**
2. 日志里出现 **2 行 `ep001` / `ep002`** 的 WER 行
3. 全程没有 `CUDA out of memory`
4. 生成了 `artifacts/checkpoints/<tag>-last.pt` 且**大小 > 100 MB**

⚠️ **冒烟出来的 WER 会很难看（比如 90%+），这是正常的，不是失败。**
冒烟只验证「能不能跑通」，**绝不用来判断效果好坏**——
我们踩过：60 条冒烟看着 +2.47pp，全量跑出来却是 −0.35pp，方向直接反了。

### 5.3 怎么判断它还在正常跑

- 每轮约 **26~45 分钟**（D3 慢些、D4 更慢），**epoch 行是每轮结束才打印**
  ⇒ 刚启动后半小时没有 epoch 行是正常的，别以为卡了。
- 健康的标志是 **GPU 利用率在 0%~90% 之间周期性跳动**（数据加载时会掉到 0），
  **单独一次 0% 不代表卡死**。
- 真的卡死 = **连续 10 分钟以上 GPU 利用率恒为 0，且日志不再增长**，这时才去看 5.1。
- 正常的一行长这样：
  ```
  ep027 ctc 14.2261  **WER_official 59.11%**  差SOTA -17.01pp
  ```
  （`ctc` = 训练集 CTC 损失，应逐步下降；`WER_official` = dev 词错误率）

### 5.4 产物分别在哪

| 产物 | 路径 | 什么时候才有 |
|---|---|---|
| 训练日志 | `artifacts/logs_D3_vac224.log`（或 `logs_D4_tfnet224.log`）| 一直在写 |
| 收据 json | `artifacts/metrics/blank-gov/p157-vac224-rgb-official-tfnet.json`<br>（D4 是 `p158-tfnet224-rgb-official-tfnet.json`）| 🔴 **只在训练正常跑完时写** |
| 最佳权重 | `artifacts/checkpoints/p157-vac224-rgb-best.pt`（约 545MB）| 每轮都在更新 |

⚠️ 收据名字结尾是 `-official-tfnet.json`，**不是**直接 `<tag>.json`。

🔴 **日志才是最重要的那个**：收据只在跑完时才生成，
中途被中断（这几乎一定会发生）时，**逐轮 WER 曲线只存在于日志里**。
真被中断了也别删东西，把当时的日志 + `<tag>-last.pt` 交回来就行。

从日志里摘逐轮曲线的办法：

```bash
grep -E "ep[0-9]{3}" artifacts/logs_D3_vac224.log > /tmp/d3_curve.txt
```

---

## 6. 🔴 判据（这一节最容易出错，请一定看完）

### 6.1 不许拿 55.84 当基线——口径对不上

我们报的最佳 dev **55.84%** 是 **P123 = VAC + 160 + BGR**。
而 D3/D4 是 **RGB 口径**（`--fix-rgb-order`）。

⚠️ D1 实测：**同一个模型换个通道顺序就塌 31~64pp**
（我们的模型 BGR→RGB +31.28pp；官方模型 RGB→BGR +63.79pp）。
⇒ **拿着 D3 的数字去比 55.84，得不出任何有效结论**，好也可能是假好，差也可能是假差。

### 6.2 正确的三条对比律

| 要回答的问题 | **正确比法** | 谁在跑 |
|---|---|---|
| 通道值多少 pp | **P150**（VAC+160+**RGB**） vs **P123**（VAC+160+**BGR**） | 我们（进行中） |
| 分辨率值多少 pp | **D3**（VAC+**224**+RGB） vs **P150**（VAC+**160**+RGB） | ← **你的任务** |
| 容量值多少 pp | **D4**（TFNet+224+RGB） vs **官方权重 33.92** | ← 你的任务 |

⇒ **D3 单独一个数字本身不能判胜负**，必须等 P150 出来两个数才相减。
排期上 P150 比 D3 先出，所以你跑完时基线已经在群里了。

### 6.3 两条通用纪律

- 🔴 **只能配对 ep32 以后**。P123 的 ep1–31 是坏解码截断的**虚高值**
  （实测 ep31→ep32 有 **−15.70pp 跳变**：74.74 → 59.04）。拿早期轮次比会得出完全错误的结论。
- **判定阈值**：改善 ≥ **1.556pp** 才算「可判有效」，小于这个都在噪声里。

---

## 7. 约定

1. **test 冻结**：任何人都不能在 test 上选/调超参。
2. **Git**：开自己的分支 `team/<名字>-<任务>`；**只有项目负责人说"推送"才推**。
3. **训练脚本一律带 `-u`**（无缓冲）—— 否则进程被杀时日志空白（我们踩过两次）。
4. 出结果（尤其 D1）**先发群里**。
5. 改 `p77` 或 `p78` 前**先在群里说一声**（我们这边有实验在依赖它们）。
6. **别在训练跑着的时候打包/压缩大文件** —— 会抢 IO 导致训练变慢甚至被杀；
   打包大文件一律用 `tar cf` 不压缩（`czf` 会 OOM，踩过）。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ls tools/blank_gov/` 里没有 `run_d3_*.sh` | 没切分支 ⇒ `git checkout experiment/blank-governance-p0-p38` |
| `ModuleNotFoundError: torch` | 用了系统 python3，必须用 `./venv/bin/python` |
| 跑着跑着没日志、GPU 归零 | 大概率会话被回收（我们踩过两次，无 traceback）。**别直接重跑** ⇒ 用 `RESUME=1 bash run_d3_vac224.sh` 续（见 5.1），否则从第 1 轮重来 |
| 结果和我们差 30pp 量级 | 通道开关用反了 ⇒ 回看 3.2 |
| 显存爆 | `BATCH=1`；D4 已 batch 1，再爆就 `--img-size 160` |
| 打包/压缩时 `Cannot allocate memory` | 系统内存小，别 gzip（jpg 已压缩，compress 只省 7%） |
| 想拿 D1 当数据门禁 | 跑不了（需 880MB 官方权重）⇒ 用 3.1 的帧数门禁代替 |
| 冒烟的 WER 特别难看 | 正常，只训了 40 条 2 轮 ⇒ 冒烟只看「能不能跑通」（见 5.2） |
| 半天没有 epoch 行 | 正常，epoch 行每轮结束才打印（26~45 min/轮）⇒ 见 5.3 |
| `check_env` 的 [5] 报 ⚠️ | 正常，官方权重只有 D1 需要；D3/D4 不需要 |

### 8.1 卡住了？先自己过一遍这四项

按顺序自查，八成问题都能自己定位：

1. `git branch --show-current` ⇒ 是不是 `experiment/blank-governance-p0-p38`？
2. `python3 tools/blank_gov/check_env.py` ⇒ [3] 帧数对不对？（见 2.1）
3. `tail -20 artifacts/logs_D3_vac224.log` ⇒ 有没有 epoch 行？GPU 还在动吗？（见 5.3）
4. `ls -la artifacts/checkpoints/` ⇒ `<tag>-last.pt` 在不在、有没有 >100MB？

还不行就把下面四样一起发群里，我们能直接定位：

```bash
git branch --show-current
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
tail -30 artifacts/logs_D3_vac224.log
python3 tools/blank_gov/check_env.py
```

---

## 9. 最后要交什么（清单）

按重要性排序，**前两项是必需的**。

### ① 训练日志 —— 必需

```
artifacts/logs_D3_vac224.log     # 或 logs_D4_tfnet224.log
```

我们要拿它做**逐轮配对**（只能用 ep32 以后，见 6.3），
所以请把**完整日志**交回来，不要只挑最后几行。

### ② 收据 json —— 跑完了就交，没跑完就没有

```bash
ls artifacts/metrics/blank-gov/*-official-tfnet.json
#   D3: p157-vac224-rgb-official-tfnet.json
#   D4: p158-tfnet224-rgb-official-tfnet.json
```

里面有 `history`（逐轮 `epoch / train_ctc / dev_wer_official`）、
`best_wer_official`、`best_epoch`、以及**全量 config**（img_size / module / lr 等）
⇒ 有它我们就能核对你跑的到底是不是我们说的那个配置。
⚠️ 注意名字结尾是 `-official-tfnet.json`。

### ③ 最佳权重 —— 先问我们要不要

```
artifacts/checkpoints/p157-vac224-rgb-best.pt   # 约 545MB
```

传不动就先不传。只有我们后面要做误差构成 / 后验锐度分析时才会单独问你要。

### ④ 一段文字结论 —— 一句话就够

```
tag / 实际跑了几轮 / best WER 多少 / 落在第几轮 / 中途断过几次、怎么续的
```

### 万一要中途停

别直接删。把当下的**日志 + `<tag>-last.pt`** 交回来即可，
我们既能拿它做部分分析，也能让你用 `RESUME=1` 接着跑（见 5.1）。

---

## 10. 版本变更

### v6（2026-10-19 13:25）
- 🔴 **修正 5.4 里写错的收据路径**：真实名字是
  `artifacts/metrics/blank-gov/<tag>-official-tfnet.json`（结尾有 `-official-tfnet`），
  不是 `<tag>.json`。这是我之前凭印象写的，核对 p78 源码 1088 行后才改对。
- 🔴 新增警示：**收据只在训练正常跑完时写**，中途被中断时
  **逐轮 WER 曲线只存在于日志里** ⇒ 日志才是最重要的产物，必须交完整日志。
- 新增 **第 9 节「最后要交什么」清单**：日志 / 收据 / 权重 / 一句话结论，
  并说明中途要停时该怎么交。

### v5（2026-10-19 13:15）
- 新增 **0.1 实验设计详解**：四配置阶梯表（相邻只差一个变量）+ 四个名词速览，
  让你知道自己的数字该怎么和别人的相减
- 新增 **2.1 如何读 check_env 输出**：6 段逐条判读（并说明 [5] 报 ⚠️ 是正常的）
- 新增 **5.2 冒烟通过判据**（四条）、**5.3 怎么判断还在正常跑**（含健康日志样例）、
  **5.4 产物在哪**
- 新增 **8.1 卡住了先自查四项** + 求助时该发哪四样
- 修 `check_env.py`：页脚不再教你们跑 D1（原来那行还带着会跳过门禁基线的 `fast`）

### v4（2026-10-19 13:00）
- 🔴 **新增 `RESUME=1` 续跑开关**（见 5.1 节；`run_d3_vac224.sh`、`run_d4_tfnet224.sh`）：
  `p78` 的 `--resume` 默认为空、**不会按 tag 自动找 ckpt**。
  30~50h 的任务被打断后若直接重跑脚本 ⇒ **从第 1 轮重来**，前面几十小时全丢。
  加了开关后脚本会自动找 `<tag>-last.pt`、校验大小（<100MB 拒收残file）、
  并从 ckpt 恢复 epoch / optimizer / scheduler（lr 里程碑不会被打乱）。
- 日志真正落盘：原先只定义了 `LOG` 变量却没重定向，等于不写文件；
  改为 `tee -a` 追加，多次续跑不覆盖历史。
- ⚠️ **已 clone 过旧版的同学先 `git pull`**，并确认拿到的是新版再开跑：
  ```bash
  git pull
  grep -c RESUME tools/blank_gov/run_d3_vac224.sh   # 输出 >0 才是新版
  ```

### v3（2026-10-19 12:20）
- 适配「D1 已由我们跑完」：你们不用再跑 D1（需要 880MB 官方权重，跑不了也不需要）
- 数据一致性门禁改用 `check_env.py` 帧数核对（dev 95891 / train 930841）
- 修正判据口径：55.84 是 BGR 口径，D3/D4 是 RGB 口径，跨口径直接比无效 ⇒ 见 6.2 三条对比律

> 📌 同一目录下另有 `team_split_3way.md`、`team-parallel-tasks.md`
> ——那是**早期草稿，已作废**，里面的操作步骤（让你们跑 D1、`p156 ... fast`、`tar czf`）
> 都不要照做。
