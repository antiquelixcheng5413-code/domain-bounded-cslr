# 可并行任务清单（队友领用）

> 前提：**不要碰 GPU**。主卡正跑 P150（VAC+160+RGB，约 24h，到 10-19 晚）。
> 并行跑 GPU 任务会触发 CUDA sticky error，把整轮训练毁掉。
> 下面的任务全部可以在 CPU / 纯读代码下完成。

---

## A 类：零 GPU，现在就能做（按优先级）

### A1. 把通道修正固化到抽帧侧（p77）🔴 最重要
**背景**：P149/P151 实测，我们落盘的 jpg 是「R/B 颠倒」语义，
现在靠 `p78 --fix-rgb-order`（读图时不再 cvtColor）兜底，属于补丁不是根治。

**要做**：
- 改 `tools/blank_gov/p77_extract_rgb.py`：抽帧时**去掉**
  `rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)`，直接对 cv2 的 BGR 数组做
  `resize(256)` + `imencode` ⇒ 这样磁盘 jpg 与官方**完全一致**（真 RGB）。
- 改完后，**只重抽 dev（515 条）做验证**，别动 train（train 重抽要 ~30min 且
  会影响在跑的实验所依赖的数据，等 P150 跑完再全量重抽）。

**验收标准**（写个 5 行的小脚本即可）：
```python
ref = cv2.resize(cv2.cvtColor(cv2.VideoCapture(v).read()[1], cv2.COLOR_BGR2RGB), (256,256))
ours = cv2.cvtColor(cv2.imread(jpg), cv2.COLOR_BGR2RGB)   # 与官方读图路径一致
MAE(ours, ref) < 2      # 现在是 10.30，修完应降到 ~1.4
```
**预计**：改代码 20min + 抽 dev 3min + 验证 10min

---

### A2. 核对「逐句等权」口径差异（P152 遗留）
**现象**：官方权重自报 dev 逐句等权 = **33.19**，
我们用 `official_wer.py` 算出 **187**，但语料级两边几乎一致（33.92 vs 33.19）。

**要做**：读 `external/TFNet/DataProcessMoudle.py` 里的 `WerScore` 函数，
和我们 `tools/blank_gov/official_wer.py` 的 `WER_per_sentence` 逐项比对：
分母是什么、空 ref / 空 hyp 怎么处理、是否含标点。

**验收标准**：能明确指出差异来自哪一行；若确认是我们的问题，顺手修正并注明
（⚠️ 不要改 `WER_official`，那是我们的主指标，已与官方对齐）。

**预计**：1~2h（纯读代码）

---

### A3. 清理死参数 `--w-kd`
**背景**：`p78` 的 `--w-kd`（默认 1.0，help 写"官方未给"）**实际从未生效** ——
SeqKD 权重在 `p78:895/897/903` 写死 `25.0`，与官方 `Train.py:196/210` 一致。
这个参数会误导后来的人（收据里一直记着 `w_kd: 1.0`）。

**要做**：删掉该参数或改成真正生效；改完**必须跑一次 `--help`** 验证
（argparse 的 help 里有字面 `%` 会直接崩，这是踩过的坑）。

**预计**：30min

---

### A4. 核实「语言模型解码」在 CSLR 的论文依据
**背景**：我们有个候选方向是 beam + gloss n-gram LM（官方没用 LM）。
SLRT（ref15, Camgoz CVPR2020）p.6 提到的是 **length penalty α**，不是 LM ——
所以目前 LM 方向**没有**直接论文依据。

**要做**：在本地论文库（`参考论文/papers/`、`新参考论文/`）里查有没有 CSLR 工作
明确用了外部语言模型做解码（可搜 "language model" / "n-gram" / "LM" / "shallow fusion"）。

**验收标准**：给出 0 篇就明确写"【无】依据"（那我会重新评估这个方向）；
有就给出篇名 + 页码 + 原句。

**预计**：1~2h

---

## B 类：需要 GPU，但要等 P150 跑完（排队）

### B1. 排查 P155 异常（我们 train WER 87.35%）
P155 测出我们 P123 在 train 子集 WER 87.35%、生成率 14.6%，比 dev（55.84）还差，
**不合理**。已排除：子集采样错位、greedy_decode 的 GPU/CPU 差异、数据读取。
需要继续查（150 条推理只要 30 秒，但要独占 GPU）。
**价值**：若确认是真"欠拟合"，整条改进路线要重排。

### B2. P156 官方权重消融（16min）—— 已在自动队列里，不用领

---

## C 类：为后续方向做准备（CPU 可做）

### C1. 实现 gloss n-gram LM + 带 LM 的 beam 解码
用 `train.csv` 的 4973 条 gloss 序列训一个 3/4-gram（纯 CPU，几分钟），
实现 shallow fusion：`score = logp_ctc + α·logP_LM + β·len`。
先不评估（评估要 GPU），把代码写好 + 用 CPU 跑通 20 条冒烟即可。

### C2. 写官方 ckpt → 我们格式的转换脚本
官方 `.pth` 键名是 `moduleNet_state_dict` / `bestWerScore` …
我们是 `model_state` / `optimizer` …。写个转换，以后能 `--resume` 官方权重
（虽然 P154 已证继续训练无收益，但作为工具备着）。

---

## 协作注意事项

1. **Git**：队友请开独立分支（如 `team/<名字>-<任务>`），不要直接动
   `experiment/blank-governance-p0-p38`。改 `p78` 这种主脚本前先说一声，
   我这边正在跑的实验依赖它。
2. **推送**：按用户约定，**只有用户说"推送"才推**，我们各自 commit 到本地即可。
3. **环境**：WSL 仓库在 `/home/su127/FYP/domain-bounded-cslr`，
   Python 必须用 `./venv/bin/python`（系统 python3 没有 torch）。
   ⚠️ 训练脚本一律加 `-u`（无缓冲），否则进程被杀时日志一片空白。
4. **不要**重抽 train 帧 —— 会影响正在跑的实验。等 P150 结束后再做全量。
