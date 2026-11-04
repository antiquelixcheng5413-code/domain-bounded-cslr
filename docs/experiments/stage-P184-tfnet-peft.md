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
- 未试：跨分辨率蒸馏 224→160、视觉对比损失。两者预期上限均 ≤0.9pp（小于噪声地板），故未排期。（部分解冻结 layer4 已于 P193 测过，见 §8.5，负向。）

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

## 8. P189–P192（2026-11-03 晚，收尾四问）

### 8.1 P189 协议对齐：0.63pp 的"差距"主要是**口径**，不是性能

先核对官方评测链路（`external/TFNet`）：

| 环节 | 官方实现 | 我们此前 |
|---|---|---|
| 解码 | `Train.py:152` → `ctcdecode.CTCBeamDecoder(beam_width=10, blank_id=0)` | greedy |
| 打分 | `Train.py:295` `WER = sum(batchWER)/len(loader)`；`WER.py:WerScore` 内部是**逐句 ed/len 求均值** | 语料级 ed 累加 |
| 预处理 | `cv2.resize(256,256)` + `CenterCrop(224)` + `/127.5-1` | 帧已存 256×256（`_dbg_size` 核对），直接 CenterCrop ⇒ 等价 |

本机无 `ctcdecode`（C++ 扩展未装），故自实现等价 prefix beam search（`p189_beam_eval.py`），并在 dev/test 各算四种口径：

| 集 | greedy 语料级 | greedy 逐句均值 | beam10 语料级 | beam10 逐句均值 | 官方 README |
|---|---|---|---|---|---|
| dev（515） | 33.92 | 33.55 | 33.88 | 33.54 | — |
| test（500） | 33.09 | **32.74** | 32.99 | **32.63** | **32.46** |

- 口径差（语料级 → 逐句均值）：dev −0.37pp，test −0.35pp；
- beam10 再加 −0.04 ~ −0.11pp；
- **与 README 的残差从 0.63pp 缩到 0.17pp（32.63 vs 32.46）**，剩余部分来自 ctcdecode 内部实现细节与 fp16 落盘，不再值得追。

⇒ 后续所有对外数字建议统一用**逐句均值**（官方口径），否则跨表比较会白丢 0.35pp。

### 8.2 P190 模型内部多头集成：**负向，零成本排除**

TFNet 推理时 `logProbs1 = logProbs5`（`Net.py:474-490`），故可用互异头 4 个（融合 / 时间支路原始 / 频域注意力 / 频域原始）。一次前向落盘后做 log 概率平均：

| 组合 | dev（逐句均值） |
|---|---|
| h0 融合（基线） | **33.55** |
| h2 频域注意力 | 36.11 |
| h1 时间支路原始 | 41.38 |
| h3 频域原始 | 41.66 |
| h0+h2 | 34.22 |
| h0+h1+h3 | 39.14 |
| 全 4 头 | 40.36 |

弱头的 blank 校准与融合头不一致，几何平均反而把 blank 概率拉低 ⇒ 比单头更差。**排除。**

### 8.3 P191 误差构成：67% 是替换，欠生成只占 1/4

官方权重 @224 test（官方口径 32.74）：

- 编辑构成：**替换 67.5% / 删除 24.5% / 插入 8.1%**（dev P184@160：71.0 / 19.8 / 9.2）；
- 生成率 94.6%（dev 96.4%），句子完全正确率 13.0%；
- 无 `<unk>`（词表覆盖没问题）；
- 高频混淆：`？→。` 28 次（占全部替换 4%）、`今天→现在`、`设计→计划`、`完了→完`（切分粒度）；
- 高频漏词：`喜欢`(10)、`最`(7)、`介绍`(7)、`而且`(7)；
- 句长分桶（test）：1–3 词 WER 23.2%（生成率 100%），4–6 词 30.2%，7–9 词 36.3%（生成率 92%），10+ 词 39.4%（生成率 81%）。

⇒ 长句的欠生成确实存在，但即使全部补回也只占 1/4 的误差；**主矛盾是同类手势的细粒度区分（替换）**，不是长度校准。这也解释了为什么 blank 偏置 / 多尺度 CTC 都无效。

### 8.4 P192 权重插值：数值最低但不显著

`soup = 0.5*P184 + 0.5*P184b`，dev **34.10**（P184 34.34 / P184b 34.24）。−0.24pp，远小于噪声地板 1.556pp ⇒ 记"不显著"，不改变部署结论。

### 8.5 P193 layer4 部分解冻 @224：**负向，与零样本统计无差别**

这是"全冻（P186，34.13）↔ 全解（P171，退化）"之间唯一未测的中点：`--freeze-backbone --freeze-heads --unfreeze-layer4 --bb-lr 1e-6`，从 `p184-init-tfnet160.pt` 起、224 输入、6 轮（冻结 19.22 M / 可训练 54.05 M），共 324.7 分钟。

| 轮次 | train ctc | dev 语料级 WER |
|---|---|---|
| ep1 | 0.7279 | 34.27 |
| **ep2** | 0.6932 | **33.81（best）** |
| ep3 | 0.6919 | 34.38 |
| ep4 | 0.6962 | 34.17 |
| ep5 | 0.6315 | 33.95 |
| ep6 | 0.6214 | 34.03 |

best 检查点按官方协议重跑（`p188 dump` + `p189` + 逐句配对）：

| 模型 | 语料级 | 逐句均值 |
|---|---|---|
| 224 零样本（official） | 33.92 | 33.55 |
| P193 best | 33.81 | **33.50** |

逐句配对（20000 次 bootstrap，n=515）：P193 更好 52 / 更差 51 / 持平 412；Δ语料级 +0.102pp，95% CI [−0.633, +0.840]；Δ逐句均值 +0.043pp，95% CI [−0.723, +0.809]，**两条 CI 均跨 0**。

⇒ 名义上逐句均值 33.50 略低于 33.55 的判据线，但 0.05pp 远小于噪声地板（dev 最小可分辨效应 ±0.77pp），**判定为无差别，不构成有效提升**。同时 train ctc 从 0.728 单调降到 0.621 而 WER 原地震荡，是"训练损失下降、识别不变"的典型过拟合形态：224 域内的适配已经被 73 M 参数的容量吃掉了。

至此 160/224 两侧可低成本尝试的方向（续训、冻头、多尺度 CTC、TTA、logit 集成、权重插值、部分解冻、内部多头、beam/口径对齐）已全部闭环，**部署仍维持双轨：P184@160（dev 34.34 / test 33.98，语料级，速度优先）与官方权重 @224 零样本（test 32.74，官方口径，精度优先）**。

### 8.6 复现命令（P189–P193）

```bash
# P189：dev 四口径（test 需先把 dump 的 --split 换成 test）
venv/bin/python tools/blank_gov/p189_beam_eval.py --tag official --p184b --beam 10
venv/bin/python tools/blank_gov/p189_beam_eval.py --tag official_test --split test
# P190：一次前向落盘 4 头 + 组合评估
venv/bin/python tools/blank_gov/p190_head_ensemble.py --ckpt <官方权重> --img 224 --tag official224
# P191：误差构成（读已有 dump，无需 GPU）
venv/bin/python tools/blank_gov/p191_error_analysis.py --tag official_test --split test
# P192：权重插值
venv/bin/python tools/blank_gov/p192_soup.py --split dev
# P193：layer4 部分解冻 @224（6 轮，约 5.4 h）
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p193-layer4-224 --module TFNet --img-size 224 --batch 2 \
  --mem-budget 7e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --freeze-heads --unfreeze-layer4 --bb-lr 1e-6 \
  --epochs 6 --total-epochs 6 --resume artifacts/checkpoints/p184-init-tfnet160.pt
# P193 官方口径复评 + 逐句配对
venv/bin/python tools/blank_gov/p188_ensemble.py dump \
  --ckpt artifacts/checkpoints/p193-layer4-224-best.pt --img 224 --tag p193
venv/bin/python tools/blank_gov/p189_beam_eval.py --tag p193
```

### 8.7 新增产物

- 收据：`p189-beam-protocol-{dev,test}.json`、`p190-head-ensemble-dev.json`、`p191-error-{test-official_test,dev-p184}.json`、`p192-soup-dev.json`、`p193-layer4-224-official-tfnet.json`
- dump：`p188_logp_official_test.pt`、`p190_heads_official224.pt`、`p188_logp_p193.pt`
- 权重：`p193-layer4-224-{best,last}.pt`
- 日志：`logs_P189_beam*.log`、`logs_P189_test*.log`、`logs_P190_heads.log`、`logs_P191_*.log`、`logs_P192_soup.log`、`logs_P193_layer4_224{,_resume}.log`、`logs_P193_dump.log`、`logs_P193_p189.log`、`logs_P193_paired.log`
