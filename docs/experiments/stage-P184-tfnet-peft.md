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

## 9. P195–P197（2026-11-05 凌晨，按文献继续三连）

### 9.1 P195 跨分辨率蒸馏（224 教师 → 160 学生）

同骨架 TFNet@224（官方权重）作为教师在 4973 条 train 上产缓存 log-prob，学生在 160 上做 KD（w=0.50，T=1），6 轮。

| 轮 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dev WER | 34.59 | 34.48 | 34.73 | 34.24 | 34.48 | **34.03** |

配对终评（同管线、同一 dev、20000 次 bootstrap，n=515）：p184 38.11 → p195 37.72（语料级），37.76 → 37.32（逐句均值）；Δ逐句 **+0.442**，CI [−0.292, +1.185]，更好 49 / 更差 37 / 持平 429。**CI 跨 0，不显著**。

> ⚠️ 该配对脚本的绝对值（38.1）与官方口径（34.34）不同：脚本按 `/` 切 ref，而官方 `wer_official` 还会做 `split_glued` 拆粘连词。**只可用于相对比较**；后续 P196/P197 已改用官方 `evaluate`。

### 9.2 P196 WE/WC 辅助约束（词存在性 + 词数回归）

对 `main_lp` 的逐帧 max 做多标签 BCE（WE）+ 对 `Σ(1-p_blank)` 与真实词数做相对误差回归（WC），权重 0.05；零新增参数、推理零开销。

| 轮 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dev WER | 34.69 | 34.55 | 34.69 | 34.41 | 34.52 | **34.20** |

配对：语料级 38.11 → 37.83，逐句均值 37.76 → 37.42；Δ逐句 **+0.341**，CI [−0.469, +1.158]，更好 58 / 更差 52 / 持平 405。**CI 跨 0，不显著**。

与 P195 同量级（+0.34 ~ +0.44pp），说明"词级辅助约束"这条线在 160 上同样到顶。

### 9.3 P197 解码侧：前缀 beam + N-best + 词级 bigram LM 重排（零训练）

- beam 实现先用**穷举全部对齐路径**的精确前缀概率做单元校验（`ALL_MATCH`），修掉两处 bug：同字符延展重复计入 `p_b`、`argpartition` 的 `kth` 越界/漏掉一类。
- 结果（p184-best @160，官方 corpus 口径，dev n=515）：

| 方案 | WER_official |
|---|---|
| greedy | 34.41 |
| blank bias −0.6 | 34.20 |
| blank bias +0.6 | 34.94 |
| beam(12/12) 1-best | 34.34 |
| **beam N-best=8 oracle（上界）** | **25.44** |
| beam + bigram LM λ=0.3 | 34.87 |
| beam + bigram LM λ=1.0 | 37.12 |

⇒ beam 与 greedy 打平（0.07pp），LM 重排单调变差（4974 句统计出的 bigram 太稀疏，未登录回退惩罚仍压过声学项），blank bias 双向都在噪声内（±0.2pp）。**Oracle 25.44 说明 8-best 里确实含更好的句子，但现有 n-gram 打分无法把它们排到前面。**

### 9.4 判定

三条线（跨分辨率蒸馏、词级辅助约束、解码侧重排）均落在 dev 噪声地板（±0.77pp）内。结论：**160×160 / 73 M 的 TFNet 在本语料上已到容量-分辨率联合上限**，继续在损失项与解码策略上做增量的边际收益接近 0；若还要推进，应换模型侧变量（更高输入分辨率下的容量扩展、更强时序主干、或数据增广/半监督），而不是继续微调 decode。

### 9.5 复现命令（P195–P197）

```bash
# P195 教师缓存（断点续跑）+ 训练
venv/bin/python tools/blank_gov/p195_teacher_cache_224.py
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p195-kd160 --module TFNet --img-size 160 --batch 2 \
  --mem-budget 7e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --freeze-heads --no-resume-sched \
  --kd-dir artifacts/kd/teacher224 --kd-weight 0.5 \
  --epochs 6 --total-epochs 6 --resume artifacts/checkpoints/p184-init-tfnet160.pt
# P196 WE/WC 辅助约束
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p196-we-wc --module TFNet --img-size 160 --batch 2 \
  --mem-budget 5e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --freeze-heads --no-resume-sched --aux-ww 0.05 \
  --epochs 6 --total-epochs 6 --workers 2 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt
# 配对终评（通用化）
venv/bin/python tools/blank_gov/paired_eval.py \
  --base p184:artifacts/checkpoints/p184-tfnet-freeze-best.pt \
  --cand p196:artifacts/checkpoints/p196-we-wc-best.pt \
  --out artifacts/metrics/blank-gov/p196-paired-dev.json
# P197 beam + LM 重排
venv/bin/python tools/blank_gov/p197_beam_lm.py \
  --ckpt artifacts/checkpoints/p184-tfnet-freeze-best.pt --tag p197 \
  --lambdas 0 0.3 0.6 1.0 1.5 2.0 --blank-bias-grid -0.6 0 0.6
```

### 9.6 新增产物

- 收据：`p195-kd160-official-tfnet.json`、`p196-we-wc-official-tfnet.json`、`p195-paired-dev.json`、`p196-paired-dev.json`、`p197-beam-lm-dev.json`
- 权重：`p195-kd160-{best,last}.pt`、`p196-we-wc-{best,last}.pt`
- 缓存：`artifacts/kd/teacher224/`（4973 条 `.npy` + `teacher_meta.json`）
- 日志：`logs_P195_train.log`、`logs_P195_paired.log`、`logs_P196_train.log`、`logs_P196_paired.log`
- 代码：`tools/blank_gov/p195_teacher_cache_224.py`、`tools/blank_gov/paired_eval.py`、`tools/blank_gov/p197_beam_lm.py`；`p78` 新增 `--aux-ww`

## 10. P198 Conformer 时序适配器（2026-11-05，负向）

### 10.1 动机与结构

§9.4 的结论是「应换模型侧变量」，但换分辨率要重跑 224 全链、换主干要重训 35 轮，算力代价都高。P198 选最省的一条：**冻结主干，只在时序表征上挂轻量适配器**，其余配方与 p184 完全一致（160×160、batch 2、lr 2e-5、6 轮，从 `p184-init-tfnet160.pt` 起步），唯一差别是适配器。

- 结构（`tools/blank_gov/conformer_adapter.py`）：瓶颈 1024→256→1024，两层轻量 Conformer，每层 = 前馈 + 深度可分离 conv(k=15) + 4 头自注意力；正弦位置编码（`persistent=False` buffer）。
- **恒等起步**：`out_proj` 零初始化，forward 返回残差增量（调用方 `pred + adapter(pred, lens)`），因此接上适配器的那一刻输出与 p184 逐位相同（单测 `|Δ|_max == 0.0`）。
- 全程 padding 掩码，注意力后 `nan_to_num`，避免 dev 尾部补齐帧污染统计量；参数量 3.563 M/个 × 2（`temporal_model` / `temporal_model1`）= **7.13 M**。
- 训练侧：主干 21.53 M 冻结，`temporal_model*` + `classifier11` 照常训练（51.74 M 可训练），适配器单独 lr **5e-4**（其余 2e-5）。

> ⚠️ 踩坑：`paired_eval.py` 原先只按 ckpt 的 `state_dict` 建裸模型，适配器权重会被 `load_state_dict` 静默丢弃，终评等于「没接适配器的 p184」。已改为先读 ckpt 内 `config`，按 `adapter/adapter_cfg` 重建并 attach 后再加载（`strict=False` 仅允许 `temporal_adapter*` 缺失）。

### 10.2 方案 C（训练侧速度扰动 0.9/1.1）不需要实现

核对了训练链与 receipts：

- `p78_train_official_tfnet.py` 的 `train_ops` 默认带 `VA.TemporalRescale(0.2)`（帧率随机重采样 ∈[0.8, 1.2]，只在训练侧，dev/test 不加）；
- `p184-tfnet-freeze-official-tfnet.json` 里 `no_temporal_rescale: false`，即拿到 34.34 的基线本身就带这层扰动。

0.9/1.1 是 [0.8, 1.2] 的子集，重复施加无新增信息量。**所以 C 撤回，本轮只做 A。**

### 10.3 结果

| 轮 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dev WER | 36.88 | 36.31 | 36.45 | 35.86 | **34.98** | 35.36 |

- best 34.98（ep5）；对照 p184@160 = 34.34；判据 ≤ 33.5 ⇒ **未达标**（差 0.64pp，仍在 ±0.77pp 的 dev 噪声地板内，属"无增益"而非明确劣化）。
- 配对终评（dev n=515，同管线，20000 次 bootstrap）：

| | 语料级 | 逐句均值 |
|---|---|---|
| p184 | 38.11 | 37.76 |
| p198 | 38.78 | 38.49 |
| Δ（p184−p198） | **−0.667**，CI [−1.507, +0.143] | **−0.735**，CI [−1.595, +0.101] |

逐句更好 47 / 更差 64 / 持平 404。**CI 跨 0，不显著。**

> ⚠️ 与 §9.1 同一条备注：该脚本绝对值（38.1）与官方口径（34.34）不同（切分方式未做 `split_glued`），只可用于相对比较。

### 10.4 判定

未达判据，且配对 CI 跨 0 —— 冻结主干的轻量时序适配器在 160×160 上**没有可测增益**。两条可能原因：

1. 表征已饱和（§9.4）：主干冻结后，偏差不是"时序结构缺失"，而是 frontend 分辨率与容量的信息上限，适配器再深也变不出新信息；
2. 零初始化把适配器最优解锚在恒等附近，6 轮 ≈ 2.9 万步、lr 5e-4 不足以让它离开恒等足够远（ep1 反而回落到 36.88，之后逐轮爬回，末轮 35.36 仍未回到起点）。

结论不变：要再推，得动**输入分辨率 + 容量**或**主干部件**（而非在冻结主干上挂 adapter），或者转向数据侧（半监督/更多标注）。

### 10.5 复现命令

```bash
# 训练（6 轮，约 3.5 h，单卡 2.2 GB）
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p198-adapter-160 --module TFNet --img-size 160 --batch 2 \
  --mem-budget 5e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --no-resume-sched --workers 2 \
  --epochs 6 --total-epochs 6 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt \
  --adapter conformer --adapter-dim 256 --adapter-layers 2 \
  --adapter-heads 4 --adapter-lr 5e-4
# 配对终评
venv/bin/python tools/blank_gov/paired_eval.py \
  --base p184:artifacts/checkpoints/p184-tfnet-freeze-best.pt \
  --cand p198:artifacts/checkpoints/p198-adapter-160-best.pt \
  --out artifacts/metrics/blank-gov/p198-paired-dev.json
```

注：日志里的「适配器权重未在上游 ckpt 中（128 项）」「优化器状态跳过（参数组数量不一致）」是预期行为（零初始化恒等起步 + 新增 lr 组）。

### 10.6 新增产物

- 代码：`tools/blank_gov/conformer_adapter.py`（新）；`p78_train_official_tfnet.py` 新增 `--adapter/--adapter-dim/--adapter-layers/--adapter-heads/--adapter-lr` 与 `attach_conformer_adapters()`；`paired_eval.py` 修复适配器重建
- 收据：`p198-adapter-160-official-tfnet.json`、`p198-paired-dev.json`
- 权重：`p198-adapter-160-{best,last}.pt`
- 日志：`logs_P198_train.log`、`logs_P198_paired.log`

## 11. P199 224 全参微调（2026-11-06，负向，按预注册规则提前终止）

### 11.1 设置

§9.4 指出的"换模型侧变量"里，唯一尚未试过的组合是 **224 输入 + 全参数微调**（此前 224 侧只测了冻主干冻头 P186 和部分解冻 layer4 P193；160 侧全参 P171 已判死）。

- 起点：`p184-init-tfnet160.pt`（P184 的 160 训练初始化）；**不冻结任何模块，73.27 M 全部可训练**。
- 配方：224×224 真 RGB、官方 collate 补帧、batch 2（显存预算 6.6 GB）、bf16、lr 2e-5 / wd 1e-4、6 轮、MultistepLR `[4,5]`。
- 预注册判定线：ep1 ≤35.0 继续；35.0–36.5 看 ep2–3；>36.5 立即停。ep1 出分后细化为：ep2 <34.5 跑 ep3；34.5–35.2 停；**>35.2 立即停**。

### 11.2 结果

| 轮 | train ctc | dev WER（语料级） | 单轮耗时 |
|---|---|---|---|
| ep1 | 1.0194 | 35.12 | 5 h 55 min |
| ep2 | 1.3059 | **35.57** | 10 h 32 min |

- ep2 = 35.57% > 35.2% 的预注册停止线 ⇒ **进程在第 2 轮结束后终止，未跑 ep3–ep6**。
- 两个负向信号：
  1. ep1 的 35.12 已经差于同分辨率下**所有**冻结方案——224 零样本 33.92、P186 冻主干冻头 34.13、P193 layer4 部分解冻 33.81；也差于 160 的 P184（34.34，但口径同为语料级，可横向比）；
  2. train ctc 从 1.0194 **上升**到 1.3059，而 WER 同步从 35.12 恶化到 35.57，与 P171@160 全参微调"ep3 见顶后崩"的前期形态一致：更新已经开始偏离预训练分布，而不是在拟合本任务。

### 11.3 判定

- **224 + 全参微调在本约束下不可用**：既没有超过冻结基线，又表现出遗忘前兆。
- 算力代价不可接受：自适应分桶实际 4965 个 train batch（多数 batch=1），显存贴顶 7454 / 8151 MiB；单轮 5.9–10.5 h（ep2 明显变慢，原因未定位），完整 6 轮 ≥ 35 h，与可能的收益（上限也只有 ~1pp）完全不匹配。
- 部署结论不变：精度优先用官方权重 @224 零样本（test 32.74，官方口径），速度优先用 P184@160（test 33.98）。

### 11.4 复现命令

```bash
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p199-full-224 --module TFNet --img-size 224 --batch 2 \
  --mem-budget 6.6e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --no-resume-sched --workers 2 --epochs 6 --total-epochs 6 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt
```

### 11.5 产物

- 权重：`p199-full-224-{best,last}.pt`（best = ep1，35.12；last = ep2，35.57）
- 日志：`artifacts/logs_P199_train.log`
- 未跑 test 与配对终评：两轮均未达判据，按规范不进入终评

## 12. P200 Conformer 时序主干替换（2026-11-06，负向）

### 12.1 动机与结构

§9.4 / §10.4 的结论都指向"换模型侧变量"，而 224 容量扩展（P199）已经关闭。最后一条低成本可验证的方向是把 TFNet 原生 **BiLSTM 时序主干整段换成 Conformer**：

- 新增 `tools/blank_gov/conformer_backbone.py`：接口与 `BiLSTMLayer` 完全一致（`x[T,B,D] + lgt -> dict['predictions']`），官方 `Net.py` 零改动；
- 结构：1024 →**bottleneck 256**→ 4 个 Conformer block（复用 P198 的 `_Block`：macaron FFN / 4 头 MHSA / DWConv k=15）→ 1024；正弦位置编码；全程 padding mask；
- 每支路 6.60 M，`temporal_model` + `temporal_model1` 共 13.20 M；替换后模型 73.27 M → **61.28 M**，冻结 conv2d 21.53 M，可训练 **39.75 M**；
- 配方：160×160、batch 2（mem 5 GB）、bf16、6 轮、冻结主干；Conformer 单独 lr **1e-4**，其余 2e-5；从 `p184-init-tfnet160.pt` 起步（BiLSTM 权重整段失效，Conformer 随机初始化）。
- 冒烟测试（32 条 / 1 轮）通过：形状、mask、续训 key 校验均正常。

### 12.2 结果

| 轮 | train ctc | dev WER（语料级） |
|---|---|---|
| ep1 | 18.4774 | 44.79 |
| ep2 | 11.0472 | 41.66 |
| ep3 | 9.4019 | 42.65 |
| ep4 | 8.4737 | 42.51 |
| ep5 | 7.1495 | 40.53 |
| ep6 | 6.7744 | **40.04** |

- 6 轮共约 3 小时 13 分（08:35→11:48），显存与 P184 同级。
- train ctc 从 18.48 单调降到 6.77，dev WER 从 44.79 缓慢降到 40.04；ep3–ep4 出现平台（42.6 / 42.5），最后两轮靠 lr 衰减才继续下探。
- **best 40.04% 比 P184@160 的 34.34 差 5.70pp**，远大于 dev 噪声地板（±0.77pp），也不如 224 零样本（33.92）——不构成可比较的候选。

### 12.3 判定

- **整段替换 BiLSTM 在本数据规模 / 6 轮预算下明确负向**：只需 6 轮就能确认它不在同一量级，无需再跑 35 轮（预计 20 h+）。
- 曲线形态说明问题不是"还没训够"：ctc 持续下降而 WER 只降 4.75pp 后趋缓，与 P193 的"损失降、识别不动"同构——随机初始化的大时序模块在 4973 条数据上主要是在拟合训练集，泛化没跟上。
- 结构性原因推测：① BiLSTM 的强序列先验在中小语料上本就优于 Transformer 类结构；② 256 维 bottleneck 对 3516 类 CTC 可能偏窄；③ 从零训练需要更多轮次/数据，而这两者都是本项目的硬约束。
- **方向 B 关闭**。若仍想利用 Conformer，只能走"预训练 + 大规模数据"路线，已超出当前 8 GB / 单卡 / CE-CSL 语料的约束。

### 12.4 复现命令

```bash
# 冒烟（32 条、1 轮）
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p200-smoke --module TFNet --img-size 160 --batch 2 \
  --mem-budget 5e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --temporal-backbone conformer \
  --tb-dim 256 --tb-layers 4 --tb-heads 4 --tb-lr 1e-4 \
  --no-resume-sched --workers 0 --epochs 1 --total-epochs 6 \
  --max-train 32 --max-dev 16 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt

# 正式 6 轮（约 3.2 h）
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p200-conformer-160 --module TFNet --img-size 160 --batch 2 \
  --mem-budget 5e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --temporal-backbone conformer \
  --tb-dim 256 --tb-layers 4 --tb-heads 4 --tb-lr 1e-4 \
  --no-resume-sched --workers 2 --epochs 6 --total-epochs 6 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt
```

### 12.5 新增产物

- 代码：`tools/blank_gov/conformer_backbone.py`（新）；`p78_train_official_tfnet.py` 新增 `--temporal-backbone/--tb-dim/--tb-layers/--tb-heads/--tb-kernel/--tb-lr` 与 `attach_temporal_backbone()`；`paired_eval.py` 同步支持按 config 重建时序主干
- 权重：`p200-conformer-160-{best,last}.pt`、`p200-smoke-{best,last}.pt`
- 收据：`p200-conformer-160-official-tfnet.json`、`p200-smoke-official-tfnet.json`
- 日志：`artifacts/logs_P200_train.log`
- 未跑 test / 配对：best 差 5.70pp，远超噪声地板，按规范不进入终评

---

## 13. P202 / E1 强空间增广（数据侧主线，2026-11-06）

### 13.1 动机

D0 诊断（`tools/blank_gov/p201_d0_diagnostics.py`，权重 `p184-tfnet-freeze-best.pt`）：

| split | n | WER | S / D / I |
|---|---|---|---|
| train | 4973 | **1.03%** | 1796 / 106 / 75 |
| dev | 515 | **34.34%** | 798 / 193 / 90 |

泛化间隙 **33.3pp** ⇒ 强过拟合，按预设门槛（train WER < 15% 即可投入增广）启动 E1。

### 13.2 改动

只动训练期增广，其余**完全沿用 P184 配方**（160×160 / TFNet / 冻结 conv2d / batch 2 / bf16 / lr 2e-5 / 6 轮 / 官方补帧 / hflip 0.5 / 时序 ±20% / seed 42）：

- `p78_train_official_tfnet.py` 新增 `ClipSpatialAug` 与 `--spatial-aug`
- 组成：`RandomResizedCrop(160, scale=(0.7,1.0), ratio=(0.85,1.18))`、旋转 ±8°、`ColorJitter(b/c/s ±0.2, hue ±0.05)`、`RandomErasing(p=0.25, scale=(0.02,0.10))`
- 关键实现：**随机参数按 clip 采样一次、整段帧共用**；逐帧独立采样会给同一段视频制造闪烁，等于往 CTC 对齐里注入噪声
- dev/test 分支**不变**（仍为 `CenterCropShift`，无随机性）；收据新增 `spatial_aug: true`

### 13.3 结果（6 轮，309.8 min）

| ep | train ctc | dev WER | P184 同轮 | 差值 |
|---|---|---|---|---|
| 1 | 10.3536 | 36.28 | 35.57 | +0.71 |
| 2 | 9.6718 | 36.03 | 34.59 | +1.44 |
| 3 | 9.5043 | 35.86 | 35.12 | +0.74 |
| 4 | 9.4360 | 36.03 | 34.34 | +1.69 |
| 5 | 9.0901 | **35.54** | 34.69 | +0.85 |
| 6 | 8.8944 | 35.57 | 34.55 | +1.02 |

- best **35.54%**（ep5），P184 为 **34.34%** ⇒ **+1.20pp**，超过 dev 噪声地板 ±0.77pp。
- 距 E1 筛选线（dev ≤ 33.6）差 **1.94pp**。
- 无 NaN；数据管线为 CPU 瓶颈（4 worker 占 ~350% CPU，GPU 利用率仅 0–25%）。
- 曲线形态：dev 在 36.28→35.54 之间缓慢下滑，ep4 的 lr 里程碑退火**没有**出现 P184 那样的下探（P184 ep3→ep4 降 0.78pp，本实验反而从 35.86 回到 36.03）。

### 13.4 判定

- **E1 判负**，按硬约束不进入 `paired_eval` / 2 seed / test。
- 机理：P184 基线已含 hflip + 时序 ±20%，再叠强空间增广后**收敛速度显著变慢**——train ctc 全程约为 P184 的 2 倍（9.4 vs 4.8），6 轮预算内追不上；每轮 dev 都比 P184 同轮差 0.3–1.4pp，属于系统性落后而非单点抖动。
- 推论：在「6 轮 × 冻结主干」这一预算下，**增强强度不是提分杠杆**；继续加码同一族（空间）增广没有意义。数据侧剩余候选只有 E2（时序）与 E4（landmark），且应先做小档验证再投 6 轮。

### 13.5 复现命令

```bash
# 冒烟（32 条 / 16 条、1 轮）
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p202-smoke --module TFNet --img-size 160 --batch 2 \
  --mem-budget 5e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --spatial-aug --no-resume-sched \
  --workers 0 --epochs 1 --total-epochs 6 --max-train 32 --max-dev 16 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt

# 正式 6 轮
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p202-e1-spatial --module TFNet --img-size 160 --batch 2 \
  --mem-budget 7e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --spatial-aug --no-resume-sched \
  --epochs 6 --total-epochs 6 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt
```

### 13.6 产物

- 代码：`p78_train_official_tfnet.py` 新增 `ClipSpatialAug` + `--spatial-aug`（收据字段 `spatial_aug`）
- 权重：`p202-e1-spatial-{best,last}.pt`、`p202-smoke-{best,last}.pt`
- 收据：`p202-e1-spatial-official-tfnet.json`、`p202-smoke-official-tfnet.json`
- 日志：`artifacts/logs_P202_train.log`、`artifacts/logs_P202_smoke.log`

---

## 14. P203 / E2 强时序增广（数据侧主线，2026-11-07）

### 14.1 改动

同样只动训练期，其余沿用 P184 配方（160 / TFNet / 冻结 conv2d / batch 2 / bf16 / lr 2e-5 / 6 轮 / 官方补帧 / hflip / seed 42）：

- `p78_train_official_tfnet.py` 新增 `ClipTemporalAug` 与 `--temporal-aug`（收据字段 `temporal_aug`）
- 随机时序裁剪：保留 70–100% 连续帧段，首/尾用该段边界帧补回原长（长度不变 ⇒ CTC dataLen 口径不变；< 48 帧样本跳过裁剪）
- 时间掩码：1 段、≤10% 帧，用**前邻帧复制**而非置零（置零会让 CTC 对齐出现整段 blank）
- 速度扰动：`TemporalRescale` 由 0.2 改 0.3 ⇒ 0.7–1.3×（官方语义 ±temp_scaling），不是叠加两套抖动
- 与 E1 不叠加（不传 `--spatial-aug`）

### 14.2 结果（6 轮，186.6 min）

| ep | train ctc | dev WER | P184 同轮 | 差值 |
|---|---|---|---|---|
| 1 | 8.9745 | 36.31 | 35.57 | +0.74 |
| 2 | 8.4200 | 35.43 | 34.59 | +0.84 |
| 3 | 8.2145 | 35.40 | 35.12 | +0.28 |
| 4 | 8.2750 | 35.12 | 34.34 | +0.78 |
| 5 | 7.9827 | **34.55** | 34.69 | **−0.14** |
| 6 | 7.9103 | 34.83 | 34.55 | +0.28 |

- best **34.55%**（ep5），P184 **34.34%** ⇒ **+0.21pp**，**落在 dev 噪声地板 ±0.77pp 内**。
- 距 E2 筛选线（dev ≤ 33.6）差 **0.95pp**。
- 无 NaN；用时 186.6 min（时序增广为轻量 list 操作，GPU 利用率 ~60%，明显快于 E1 的 309.8 min）。

### 14.3 判定

- 按硬约束「单次 6 轮 dev 必须比 34.34 好 ≥0.7pp 才进入复跑」⇒ **E2 判负（无差别）**，不进入 paired_eval / 2 seed / test。
- 与 E1 相比机制更对路：ep1/ep2 的 dev 比 E1 同轮好约 0.6pp，ep5 甚至比 P184 同轮好 0.14pp；但**最佳轮仍与基线持平**，且 ctc 全程高于 P184（7.9–9.0 vs 4.7–5.4），说明增广只是让收敛变慢、并未带来可测的泛化收益。
- 结合 §13（E1 +1.20pp 明确负向）与本节（E2 ±0.21pp 噪声内），**数据增广这条线可以关闭**：在「6 轮 × 冻结主干」预算下，通用增广无法缩小 33.3pp 的训练–dev 间隙。剩余未证伪的只有 E4（landmark 辅助监督）。

### 14.4 复现命令

```bash
# 冒烟
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p203-smoke --module TFNet --img-size 160 --batch 2 \
  --mem-budget 5e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --temporal-aug --no-resume-sched \
  --workers 0 --epochs 1 --total-epochs 6 --max-train 32 --max-dev 16 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt

# 正式 6 轮
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p203-e2-temporal --module TFNet --img-size 160 --batch 2 \
  --mem-budget 7e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --temporal-aug --no-resume-sched \
  --epochs 6 --total-epochs 6 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt
```

### 14.5 产物

- 代码：`p78_train_official_tfnet.py` 新增 `ClipTemporalAug` + `--temporal-aug`
- 权重：`p203-e2-temporal-{best,last}.pt`、`p203-smoke-{best,last}.pt`
- 收据：`p203-e2-temporal-official-tfnet.json`、`p203-smoke-official-tfnet.json`
- 日志：`artifacts/logs_P203_train.log`、`artifacts/logs_P203_smoke.log`

---

## 15. P206 / E4 landmark 时序融合（数据侧收尾，2026-11-07）

### 15.1 动机与前置探针

D0 的替换分析显示 dev 的替换不集中在单一手形词族，通用增广因此失效（§13、§14）。E4 用已落盘的 MediaPipe landmark（`data/processed/ce-csl-landmark48`，每条 **固定 (48,368)** float32）作为第二视图。

**探针先做两件事**（`tools/blank_gov/p204_landmark_probe.py`、`p205_landmark_only.py`）：

| 项 | 结果 |
|---|---|
| landmark48 覆盖 | **5487** 条（train 4973 + dev，dev 缺 `dev-00403`；**无 test**） |
| landmark96 | 目录为空（未落盘） |
| 单模态 WER（独立 LSTM+CTC，1.39M 参数，60 轮 2 min） | 100% → **64.12%** 后停滞（train ctc 0.72） |
| 标签长度 | mean 5.7，max 16（48 步足够 aux CTC） |

⇒ 单模态远弱于 RGB（34.34%），但携带 gloss 信息（随机水平 100%）。
**同时修正了原方案的一个逻辑漏洞**：原计划的「独立辅助 CTC 头 + 推理丢弃」不共享参数（conv2d 冻结），对 test WER 恒等于零；要让 landmark 影响指标只能做**时序级融合**。

### 15.2 实现

- `tools/blank_gov/landmark_fusion.py`
  - `LandmarkBank`：按 sid 懒加载 + dict 缓存，train 200 条确定性统计的 mu/sd 归一化，缺失文件返回全 0 行并计数
  - `LandmarkFusion`：`Linear(368→256) + LayerNorm + GELU + Dropout(0.1) + BiLSTM(256→128×2)` → `proj(256→512)`，**proj 权重/bias 全零初始化 ⇒ 初始 delta≡0，ep0 与 P184 严格等价**（冒烟实测 conv2d 输出差 0.000e+00）
  - 注入：`model.conv2d` forward hook，`F.interpolate` 把 48 步插到 `[B*temp,512]` 的 temp 步后残差相加；融合段强制 fp32（cuDNN LSTM 在 bf16 autocast 下不可靠）
- `p78_train_official_tfnet.py`：`--lm-fusion` / `--lm-hidden`(256) / `--lm-lr`(5e-4，独立 lr 组)；检查点新增 `lm_fusion_state`；未加开关时行为完全不变
- 参数量：融合 **0.62M** / 总 73.27M

### 15.3 结果（6 轮，225.1 min）

| ep | train ctc | dev WER | P184 同轮 | 差值 |
|---|---|---|---|---|
| 1 | 5.2453 | 35.26 | 35.57 | −0.31 |
| 2 | 4.7492 | **34.55** | 34.59 | −0.04 |
| 3 | 4.6147 | 35.08 | 35.12 | −0.04 |
| 4 | 4.5213 | 34.83 | 34.34 | +0.49 |
| 5 | 4.2888 | 34.73 | 34.69 | +0.04 |
| 6 | 4.1165 | **34.10** | 34.55 | **−0.45** |

- best **34.10%**（ep6，最后一轮），P184 **34.34%** ⇒ **−0.24pp**：是四条候选里**唯一在 best 上优于 P184**的实验，但幅度**仍在噪声地板 ±0.77pp 内**。
- 距筛选线（dev ≤ 33.6）差 **0.50pp**。
- bank 缺失 1 条（dev-00403）；无 NaN。

### 15.4 判定

- 按硬约束「单次 6 轮 dev 必须比 34.34 好 ≥0.7pp」⇒ **P206 判负（无差别）**，不进入 paired_eval / 2 seed / test。
- 但曲线形态与此前三条候选不同：ep3 有一个反弹，随后 35.08 → 34.83 → 34.73 → **34.10** 持续下降，且 best 落在**最后一轮**——说明 6 轮预算内**尚未收敛**（landmark 分支随机初始化、lr 5e-4，起步更慢）。这是唯一一条「延长训练可能过线」的线索，但在现有预算下不能作为结论。
- 另有两个硬限制：(1) **test landmark 不存在**，即使 dev 过线也无法做终评，需先补 extract；(2) 推理侧必须保留 MediaPipe 分支，部署成本上升，与「速度优先 160 档」的定位相冲突。
- 综合 §13（E1 +1.20pp）、§14（E2 +0.21pp）、本节（P206 −0.24pp）⇒ **数据侧全部到顶**，项目维持双轨：P184@160（test 33.98）+ 官方 @224（test 32.74）。

### 15.5 复现命令

```bash
# 冒烟（32/16，1 轮）
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p206-smoke --module TFNet --img-size 160 --batch 2 \
  --mem-budget 5e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --lm-fusion --no-resume-sched \
  --workers 0 --epochs 1 --total-epochs 6 --max-train 32 --max-dev 16 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt

# 正式 6 轮
venv/bin/python tools/blank_gov/p78_train_official_tfnet.py \
  --tag p206-lm-fusion --module TFNet --img-size 160 --batch 2 \
  --mem-budget 7e9 --amp-bf16 --lr 2e-5 --weight-decay 1e-4 \
  --hidden 1024 --seed 42 --official-pad --fix-rgb-order \
  --freeze-backbone --lm-fusion --no-resume-sched \
  --epochs 6 --total-epochs 6 \
  --resume artifacts/checkpoints/p184-init-tfnet160.pt
```

### 15.6 产物

- 代码：`tools/blank_gov/landmark_fusion.py`（新增）、`p78_train_official_tfnet.py`（`--lm-fusion`）
- 探针：`p204_landmark_probe.py`、`p205_landmark_only.py`
- 权重：`p206-lm-fusion-{best,last}.pt`（含 `lm_fusion_state`）、`p206-smoke-{best,last}.pt`
- 收据：`p206-lm-fusion-official-tfnet.json`（`lm_fusion=true`、`lm_bank_miss=1`）、`p206-smoke-official-tfnet.json`
- 日志：`artifacts/logs_P206_train.log`、`artifacts/logs_P206_smoke.log`
