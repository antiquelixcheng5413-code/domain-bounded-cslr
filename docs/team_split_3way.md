# ⚠️ 本文档已作废——请以 `docs/TEAM_GUIDE.md` 为准

> **唯一有效的最新说明是 [`docs/TEAM_GUIDE.md`](TEAM_GUIDE.md)（v4）。**
> 本文件保留仅供参考的背景信息（硬件、打包体积、任务动机），
> 但**里面的操作步骤已过时**，直接照做会出错。已知过时点：
>
> - ❌ 让队友先跑 **D1**：D1 我们已跑完，且需要 880MB 官方权重，你们跑不了
> - ❌ `... p156_official_ablation.py 0 fast`：`fast` 会跳过 224+RGB 门禁基线
> - ❌ `tar czf` 打包：7GB 内存下会 `Cannot allocate memory`，必须用 `tar cf`
> - ❌ 把训练脚本当成可直接重跑：`--resume` 默认不生效，
>   中断后必须 `RESUME=1 bash run_d3_vac224.sh`，否则从第 1 轮重来
>
> *下面保留原文。*

# 三人并行分工（我们 5060 8G + 队友两台 3060 8G）

> 3060 = Ampere（GA106, CC 8.6）⇒ **支持 bf16** ⇒ `--amp-bf16` 可原样使用，无需改精度。
> 3060 算力略低于 5060 ⇒ 预计同任务慢 20~40%，下面的时长按此估计。

---

## 分工总表

| 人 | 任务 | 数据 | 预计 | 产出 |
|---|---|---|---|---|
| **我们**（5060） | **P150** VAC+160+RGB | 已有 | ~24h（已跑 10h） | **通道**值多少 pp |
| **队友 A** | **D1**（先）+ **D3** | 轻量包 → 完整包 | 16min + ~30h | **分辨率**值多少 pp |
| **队友 B** | **D4**（+ 顺手 D2） | 完整包 | ~50h | **容量**值多少 pp |

四项来源（通道 / 分辨率 / 容量 / 训练配置）**同时开跑**，
墙钟时间从串行 ~3.7 天 ⇒ **约 2 天**（瓶颈是队友 B 的 TFNet）。

> 训练配置已排除（与官方逐项一致，wd/lr/LS 全判负），不占人手。

---

## 队友 A：D1 → D3

### 第一步 D1（16 分钟，轻量包 3.6GB）
```bash
./venv/bin/python -u tools/blank_gov/p156_official_ablation.py 0 fast
```
产出：官方权重上 224→160 掉多少、RGB→BGR 掉多少。
**意义**：这个数决定 D3 值不值得做 —— 若 160 只掉 2pp，D3 可以省 30 小时。
⇒ **D1 出结果后先发给我，再决定要不要开 D3。**

### 第二步 D3（VAC+224+RGB，需 train）
```bash
./venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag p157-vac224-rgb --module VAC --img-size 224 --batch 2 \
  --mem-budget 5.2e9 --amp-bf16 --epochs 55 --total-epochs 55 \
  --lr 1e-4 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad \
  --word-head 0.3 --word-boundary ctc_align --warmup-ep 3 --workers 4 \
  --fix-rgb-order
```

---

## 队友 B：D4（+ 可选 D2）

### D4（TFNet+224+RGB，需 train）
```bash
./venv/bin/python -u tools/blank_gov/p78_train_official_tfnet.py \
  --tag p158-tfnet224-rgb --module TFNet --img-size 224 --batch 1 \
  --mem-budget 6.5e9 --amp-bf16 --epochs 55 --total-epochs 55 \
  --lr 1e-4 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad \
  --word-head 0.3 --word-boundary ctc_align --warmup-ep 3 --workers 4 \
  --fix-rgb-order
```
⚠️ TFNet 首跑前**务必先小规模冒烟**（`--max-train 40 --epochs 2`）确认能跑通再上 55 轮，
免得跑几小时才发现 OOM 或结构不匹配。

### 可选 D2（30 秒算力）
排查 P155 异常：我们 P123 在 train 子集 WER 87.35%（比 dev 55.84 还差，不合理）。

---

## 数据交付

| 包 | 大小 | 给谁 | 内容 |
|---|---|---|---|
| `cecsl-dev-pkg.tar.gz` | ~2.5GB | 两人都要 | dev 帧（95891 张） |
| 官方权重 `TFNet-...-32.46.pth` | 880MB | 队友 A | 已在 `官方权重/` 目录 |
| `cecsl-train-pkg.tar.gz` | ~22GB | 跑 D3/D4 的人 | train 帧（930841 张） |

解压到仓库的 `artifacts/official_rgb/` 下即可（dev/ 和 train/ 两个目录）。

> train 包较大（约 22GB）。建议移动硬盘直拷；网盘上传耗时很长。
> ⚠️ 我们这边 train 的打包要等 P150 跑完再做（现在打包会抢 IO 拖慢训练）。

---

## 环境（队友机器）
- Linux/WSL + Python 3.11+
- `pip install torch torchvision`（需匹配 CUDA；3060 用 CUDA 11.8/12.x 均可）
- `pip install opencv-python numpy tqdm`
- `git clone` 仓库后，把帧放进 `artifacts/official_rgb/{dev,train}`

## 统一约定
1. **判据一致**：所有人与基线 P123（dev 55.84）比；配对只用 **ep32+**
   （P123 的 ep1–31 是坏解码截断的虚高值，实测跳变 −15.70pp，不可比）
2. **指标一致**：主指标 = `official_wer.py` 的 `WER_official`（语料级）
3. **test 冻结**：任何人都不得在 test 上选/调超参
4. **Git**：各自开分支 `team/<名字>-<任务>`；**只有用户说"推送"才推**
5. **训练脚本一律加 `-u`**（无缓冲，否则进程被杀时日志空白）
6. **谁先出结果就发群里**，尤其是 D1 —— 它可能让我们省掉 30 小时
