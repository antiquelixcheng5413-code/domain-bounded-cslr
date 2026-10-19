# 队友操作指南（D3 / D4）

> **v4 · 2026-10-19 13:00 更新**（变更记录见文末第 9 节）
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
跑完把**日志文件**和 `artifacts/metrics/blank-gov/` 下的收据 json 一起发回来。

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
| 通道值多少 pp | **P150**（VAC+160+**RGB**） vs **P123**（VAC+160+**BGR**） | 我们（进行中，约今晚出） |
| 分辨率值多少 pp | **D3**（VAC+**224**+RGB） vs **P150**（VAC+**160**+RGB） | ← **你的任务** |
| 容量值多少 pp | **D4**（TFNet+224+RGB） vs **官方权重 33.92** | ← 你的任务 |

⇒ **D3 单独一个数字本身不能判胜负**，必须等 P150 出来两个数才相减。
排期上 P150 比 D3 先出（P150 剩 ~13h，D3 约 30h），所以你跑完时基线已经在群里了。

### 6.3 两条通用纪律

- 🔴 **只能配对 ep32 以后**。P123 的 ep1–31 是坏解码截断的**虚高值**
  （实测 ep31→ep32 有 **−15.70pp 跳变**：74.74 → 59.04）。拿早期轮次比会得出完全错误的结论。
- **判定阈值**：改善 ≥ **1.556pp** 才算"可判有效"，小于这个都在噪声里。

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

---

## 9. 版本变更

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
