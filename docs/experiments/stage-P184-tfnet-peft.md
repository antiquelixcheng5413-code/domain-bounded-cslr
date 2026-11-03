# P184 — 冻结主干：TFNet@160 的参数高效适配（含 P184b / P185 / P186 / P187 / P188）

> 状态：已收口（2026-11-03）
> 定位：发布权重在 **160 轻量配置**下的最佳用法；并系统性排除同权重上的推理侧技巧。
> 结论一句话：**冻结 ResNet34MAM 主干、只适配时序分支，是本约束下唯一稳定正向的改动（dev 36.81→34.34，test 35.66→33.98）；其余手段均为负向或不显著。**

## 0. 背景

- 官方发布权重 `TFNet-CE-CSL-CSLDaily-32.46.pth`，在 CE-CSL **dev 224 零样本 33.92 / test 33.09**。
- 工程约束：单卡 8GB、160 输入、测试集冻结、只用训练集训练、官方 `official_wer`。
- P171（全量微调）已判死：best 39.90%@ep3，之后 16 轮退化到 43–46%，ctc 从 9.0 降到 7.72 而 WER 反升 ⇒ 灾难性遗忘。
- P184 的假设：**160 相对 224 的损失主要来自时序对齐，而非视觉编码器**；因此只解冻时序模块即可补回，且不破坏预训练特征。

## 1. 方法

- 模型：官方 `Net.moduleNet("TFNet")`，73.27 M。
- 初始化：发布权重 strict 载入（missing=0 / unexpected=0），`--word-head 0`。
- **冻结策略**：`model.conv2d`（resnet34MAM，21.53 M）`requires_grad=False`；每轮 `model.train()` 后追加 `model.conv2d.eval()`，防止 BatchNorm running stats 被 160 输入改写。可训练 51.74 M。
- 输入：160×160 真 RGB（`--fix-rgb-order`）、官方 collate 补帧（`dataLen=ceil(T/4)*4+12`）、hflip 0.5、temporal rescale 0.2。
- 损失：官方 TFNet 7 项（主 CTC + 3 分支 CTC + 2×SeqKD(25×KL) + 融合 CTC），`blank=0, zero_infinity=True`。
- 优化：Adam lr 2e-5 / wd 1e-4 / 梯度裁剪 5.0 / bf16 autocast；6 轮；MultistepLR `[4,5]` gamma 0.2。
- 评估：每轮 dev 官方 greedy + `official_wer`；测试集仅在 dev 明确超过噪声地板（1.556pp）时跑一次。
- 训练耗时 196.3 min（6 轮含逐轮 dev 评估）；`nan_steps_skipped=0`。

## 2. 正向结果

### 2.1 P184（冻主干，6 轮）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| train ctc | 5.412 | 5.049 | 4.934 | 4.838 | 4.789 | 4.671 |
| dev WER | 35.57 | 34.59 | 35.12 | **34.34** | 34.69 | 34.55 |

- best **dev 34.34%**（ep4），**test 33.98%**（官方口径；edits 1033 / ref_tokens 3040，exact 11.6%）。
- 曲线形态与 P171 相反：**ctc 与 WER 同向下降，无退化** ⇒ 冻结确实阻断了灾难性遗忘。

### 2.2 P184b（再冻 classifier11/33/55）

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dev WER | 34.59 | 34.41 | 34.38 | 34.31 | 34.66 | **34.24** |

- 冻结 32.33 M、可训练 40.94 M，dev 34.24 与 P184 的 34.34 差 0.10pp（**远低于噪声地板**）。
- 结论：**分类头无需重训**；更少参数即可等价，泛化更稳（ep6 仍在下降）。
- 按规范未跑 test（dev 增益不足）。

### 2.3 P184-cont（续训验证是否欠训）

- 从 best(ep4) 续训，lr 重置 1e-5、6 轮。
- ep5–ep8 = 34.38 / 34.55 / 34.59 / 34.41，**未破 34.34**，ep8 后停。
- 结论：34.3 附近是真实平台，不是训练不足。

## 3. 负向 / 不显著对照

| 实验 | 设置 | dev | 判定 |
|---|---|---|---|
| P186 | 224 + 冻主干冻头 PEFT | 34.13（ep1 最好，之后 34.2–34.4） | ❌ 不如 224 零样本 33.92；ctc 0.71→0.58 而 WER 不动 ⇒ 原生分辨率下继续训练只会拟合训练集 |
| P185 | 多尺度 CTC 中间监督 | 35.50 | ❌ 比 P184b 差 1.26pp；额外监督破坏对齐 |
| P187 | P184 权重扫推理分辨率 | 160: 34.34 / 192: 34.83 / 224: OOM | ❌ 160 自检复现成功；192 变差；224 触发 OOM（36.56 无效） |
| TTA | hflip / 192 / 224 多视平均 | 34.48–34.98 | ❌ 全部劣于单视 34.34 |
| P188 | P184+P184b+官方224 logit 等权集成 | 33.78 | ⚠️ 数值最低但**不显著**（见下） |

### 3.1 P188 配对检验（关键）

- 集成 dev 33.78 vs 最佳单成员 33.92，Δ = **−0.141pp**。
- 逐句：**更好 52 / 更差 49 / 持平 414**（515 句，仅 101 句发生变化）。
- 95% bootstrap CI（20000 次重采样）：**[−0.902, +0.630]**，**跨 0**。
- 结论：数值上最低，但统计上不可区分；且推理成本 3×（2×160 + 1×224，224 还会 OOM），不采用。
- 根因：三成员同源于同一发布权重，错误高度相关，无互补性。

## 4. 判定与部署建议

| 目标 | 配置 | dev | test |
|---|---|---|---|
| 精度优先 | **官方权重 224 零样本** | 33.92 | **33.09** |
| 速度优先 | **P184 冻主干 @160** | 34.34 | **33.98** |
| 更省参数 | P184b @160（冻头） | 34.24 | 未跑 |

- 160 的理论上限 = 追平 224 零样本（dev 差 0.42pp / test 差 0.89pp），**均小于噪声地板** ⇒ 160 侧继续调参已无可统计证明的收益。
- 与官方 README 32.46（同权重 test）相比：224 零样本差 0.63pp，P184@160 差 1.52pp——**尚未超过发布数字**；该 0.63pp 残差可能来自预处理/补帧/解码细节，未定位。
- 论文对照口径提醒：论文表中 TFNet 41.9 是**从头训练**，与本项目使用发布权重不可直接比较。

## 5. 局限

- 机制证据为行为层（ctc/WER 走势），未做 CKA / 线性探针来直接证明"视觉特征被保住"。
- dev 仅 515 句，最小可分辨效应约 ±0.77pp；0.14pp 级差异无法判定。
- 224 训练/推理在 8GB 上不稳（P187 已实测 OOM），224 侧改进受限。
- 未试：部分解冻 layer4、跨分辨率蒸馏 224→160、视觉对比损失。三者的预期上限均 ≤0.9pp（小于噪声地板），故未排期。

## 6. 复现命令

```bash
# 初始化（发布权重 → p78 可续训 ckpt，word_head=0）
venv/bin/python tools/blank_gov/make_p171_init.py \
  --word-head 0.0 --total-epochs 6 \
  --out artifacts/checkpoints/p184-init-tfnet160.pt

# P184
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p184-tfnet-freeze --module TFNet --img-size 160 --batch 2 \
  --mem-budget 7e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --no-resume-sched --epochs 6 --total-epochs 6 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt

# P184b（仅多一个 --freeze-heads）
# P186（--img-size 224，其余同 P184b）
# P187（推理分辨率扫描）
OFFICIAL_CKPT=artifacts/checkpoints/p184-tfnet-freeze-best.pt \
RES_LIST=160,192,224 venv/bin/python tools/blank_gov/p156_official_ablation.py 0 fast
# P188（dump 三次 + combine/compare）
venv/bin/python tools/blank_gov/p188_ensemble.py combine --inputs p184,p184b,official --weights 1,1,1
venv/bin/python tools/blank_gov/p188_ensemble.py compare --inputs p184,p184b,official --weights 1,1,1
```

## 7. 产物

- 权重：`artifacts/checkpoints/p184-tfnet-freeze-{best,last}.pt`、`p184b-frozen-heads-{best,last}.pt`、`p186-224-peft-*.pt`
- 收据：`artifacts/metrics/blank-gov/p184-tfnet-freeze-official-tfnet.json`、`p184b-frozen-heads-official-tfnet.json`、`p186-224-peft-official-tfnet.json`、`p188-ensemble-dev.json`、`p188-ensemble-paired.json`
- 日志：`artifacts/logs_P184_train.log`、`logs_P184_cont.log`、`logs_P184_test.log`、`logs_P184b_train.log`、`logs_P185_mlctc.log`、`logs_P186_224.log`、`logs_P187_res_sweep.log`、`logs_P188_ensemble.log`
