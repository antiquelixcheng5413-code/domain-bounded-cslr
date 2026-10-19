# 可并行任务清单（队友领用）

> **两个前提**
> 1. 队友用**自己的电脑/显卡** ⇒ 不占我们的主卡（我们正跑 P150，约 24h）。
>    ⚠️ 若队友要**在我们的机器上**跑，绝对不行 —— 会触发 CUDA sticky error。
> 2. 只在我们机器上跑的任务仍标「需主卡」，见 B 类。

---

## 0. 数据与包（先解决这个）

| 内容 | 大小 | 用途 |
|---|---|---|
| dev 帧 | **2.7 GB**（95891 帧） | 所有评估类任务 |
| 官方权重 `TFNet-CE-CSL-CSLDaily-32.46.pth` | **880 MB** | 消融/诊断 |
| train 帧 | **23 GB**（930841 帧） | 训练类任务 |
| test 帧 | 2.9 GB | 最终验收（冻结，别动） |

**轻量包 ≈ 3.6GB**（dev + 官方权重）⇒ 够跑下面 D1/D2，传输快。
**完整包 ≈ 26GB**（加 train）⇒ 才跑 D3/D4。

打包命令（在我们的 WSL 里执行）：
```bash
cd ~/FYP/domain-bounded-cslr/artifacts/official_rgb
tar czf /tmp/cecsl-dev.tar.gz dev                       # 轻量包核心
# 完整包再加 train（耗时长）
tar czf /tmp/cecsl-train.tar.gz train
```

---

## D 类：队友用自己的 GPU（推荐领这些）

### D1. 官方权重消融（≈16min 算力）🔴 最优先
**脚本**：`tools/blank_gov/p156_official_ablation.py`
**需要**：dev 帧 + 官方权重（轻量包即可）
**做什么**：在官方 TFNet 上改推理侧输入，量出
- 224→160 掉多少 pp（分辨率代价）
- RGB→BGR 掉多少 pp（通道代价）
- 两者叠加掉多少（= 我们的取舍）

**为什么先跑它**：只要 16 分钟就能知道这两项的量级，
从而**决定我们队列里 B（224 训练，24h）到底值不值得跑**。
⚠️ 是"推理时换输入"的上界，从头训练会适应 ⇒ 真实损失更小。

**验收**：dev 上 4 个组合的 WER 表（224+RGB 应复现 33.92）

---

### D2. 排查 P155 异常（30 秒算力，但要想）
**现象**：我们 P123 在 train 子集 WER 87.35% / 生成率 14.6%，
比 dev（55.84）还差 ⇒ **不合理**。
已排除：子集采样错位、greedy_decode 的 GPU/CPU 差异、数据读取（帧数/dl/L 都正常）。
**需要**：dev（验证脚本正确性）+ train 子集 150 条（~0.7GB）
**价值**：若确认是真「欠拟合」，整条改进路线要重排（加训练/容量空间会很大）。

---

### D3. VAC + 224 + RGB 训练（≈24h 算力，需 ≥8GB 显存）
**命令**（与我们队列里 B 完全一致，可直接并行）：
```bash
./venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag p157-vac224-rgb --module VAC --img-size 224 --batch 2 \
  --mem-budget 5.2e9 --amp-bf16 --epochs 55 --total-epochs 55 \
  --lr 1e-4 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad \
  --word-head 0.3 --word-boundary ctc_align --warmup-ep 3 --workers 4 \
  --fix-rgb-order
```
**意义**：测分辨率 160→224 值多少（P127 在 BGR 下测的"224 更差"可能是错的）

### D4. TFNet + 224 + RGB 训练（≈41h 算力，需 ≥6GB 显存）
同上但 `--module TFNet --batch 1 --mem-budget 6.5e9 --tag p158-tfnet224-rgb`
**意义**：测容量 35M→80M 值多少

> D3/D4 若队友跑，我们这边队列里对应的步骤就可以删掉，总工期从 ~3.7 天缩到 ~1.5 天。

---

## A 类：零 GPU，任何机器都能做（按优先级）

### A1. 把通道修正固化到抽帧侧（p77）🔴 最重要
现在靠 `p78 --fix-rgb-order`（读图时不再 cvtColor）兜底，属于补丁不是根治。
改 `p77_extract_rgb.py`：抽帧时**去掉** `cvtColor(BGR2RGB)`，
直接对 cv2 的 BGR 数组 `resize(256)` + `imencode` ⇒ 磁盘 jpg 与官方一致。
**验收**：重抽 dev 后，`imread`+`cvtColor` 得到的图 vs 视频直解的真 RGB，MAE 从 10.30 → <2
⚠️ **只重抽 dev 验证，切勿重抽 train**（正在跑的实验依赖它）

### A2. 核对「逐句等权」口径（P152 遗留）
官方自报 33.19，我们 `official_wer.py` 算 187，但语料级一致（33.92 vs 33.19）。
读 `DataProcessMoudle.WerScore` 与我们的 `WER_per_sentence` 逐项比对。
⚠️ 不要动 `WER_official`（主指标，已与官方对齐）

### A3. 清理死参数 `--w-kd`
help 写"官方未给"，实际在 `p78:895/897/903` 写死 25.0（与官方一致）。
改完**必须跑一次 `--help`**（argparse help 里有字面 `%` 会直接崩）

### A4. 核实「语言模型解码」在 CSLR 的论文依据
SLRT（ref15）p.6 的 α 是 **length penalty** 不是 LM ⇒ LM 方向目前**无**直接依据。
查本地论文库有无 CSLR 用外部 LM 解码（搜 language model / n-gram / shallow fusion）。
**验收**：有 → 篇名+页码+原句；无 → 明确写【无】（我会据此重估该方向）

---

## C 类：为后续方向做准备（CPU 可做）
- **C1**：用 train.csv 的 4973 条 gloss 训 3/4-gram LM（纯 CPU）+ 实现
  `score = logp_ctc + α·logP_LM + β·len` 的 beam 解码，CPU 跑通 20 条冒烟
- **C2**：写官方 ckpt → 我们格式的转换（`moduleNet_state_dict` ↔ `model_state`）

---

## 环境要求（队友机器）
- Linux/WSL + Python 3.11+ ；`pip install torch torchvision`（需与 CUDA 匹配）
- OpenCV（`cv2`）、numpy、tqdm
- 评估类任务还需 `imageio`（装不上也不影响，我们已绕过）
- 仓库代码：`git clone` 后把帧放到 `artifacts/official_rgb/{dev,train}`

## 协作约定
1. **Git**：开独立分支 `team/<名字>-<任务>`，别直接动
   `experiment/blank-governance-p0-p38`。改 `p78` 主脚本前先说一声。
2. **推送**：按用户约定，**只有用户说"推送"才推**，各自本地 commit 即可。
3. 训练脚本一律加 `-u`（无缓冲），否则进程被杀时日志一片空白（踩过两次）。
4. **test 帧冻结**：任何人不得在 test 上选/调超参。
