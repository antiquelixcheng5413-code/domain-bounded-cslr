# P208 / D2 — 多视图一致性正则（E3 / CR-CTC 风格，判负）

> 日期：2026-11-07｜状态：**负向（无差别），不进 paired_eval / 2 seed / test**
> 关联：`OPT-DIRECTIONS-2026-11-07.md` D2、`DATA-SIDE-PLAN-2026-11-06.md` E3、
> P184 / P202(E1) / P203(E2) / P206(E4)
> 依据等级：【有】R-Drop（NeurIPS 2021, arXiv:2106.14448）、CR-CTC（Yao et al.）；
> 【间接】本地 ref12 CCL-SLR

## 1. 动机与预注册判据

D0 诊断显示 train WER 1.03% vs dev 34.34%（**33.3pp 泛化间隙**），
但 E1(P202, +1.20pp)、E2(P203, +0.21pp) 通用增广均失败。
E3 假设：问题不是「数据不够多样」，而是**两次前向的输出分布不一致 / CTC 峰值塌缩**，
故用一致性约束（正则项，不是增广）压低过拟合。

**判据（启动前写死）**：dev ≤ **33.80** 且 train–dev 间隙缩小；
另遵守项目硬约束「比 34.34 好 ≥0.7pp 才进 paired_eval / 2 seed / test」。

## 2. 方法（最小侵入，默认关闭）

- `p78_train_official_tfnet.py` 新增 `--cons-weight`（默认 0.0）/`--cons-noise`（默认 0.05），
  **不加开关时行为与 P184 逐位一致**。备份 `/tmp/p78_pre_d2.py`。
- 第二视图 = `vid + 0.05·N(0,I)` 后 clamp 到 [-1,1]；**不改帧数 ⇒ CTC 对齐与 lgt 不变**。
- 主 CTC（官方 7 项 loss）仍只用第一视图；额外加对称 KL：
  `0.5·[KL(p1||p2) + KL(p2||p1)]`，对方分布 detach（标准 R-Drop 做法，双向都有梯度）。
- **按非 blank 帧加权**：`w = 1 − ½(p1_blank + p2_blank)`，乘有效帧 mask，
  避免 CTC 峰值分布中 ~87% blank 帧主导 KL。λ=0.3。
- 配方逐项沿用 P184：160 输入 / 冻结 conv2d + 每轮 `conv2d.eval()` / batch 2 /
  bf16 / lr 2e-5 / wd 1e-4 / 6 轮 / MultistepLR[4,5] / seed 42 / hflip 0.5 /
  TemporalRescale 0.2 / 官方补帧。从 `p184-init-tfnet160.pt` 起。
- 冒烟（32/16 条 1 轮）PASS：两次前向未 OOM，参数生效。
- 正式 6 轮耗时约 3 h 35 min（14:13→17:48），`nan_steps_skipped=0`。

## 3. 结果

| ep | train ctc | dev WER | P184 同轮 | 差 |
|---|---|---|---|---|
| 1 | 5.4102 | 35.61 | 35.57 | +0.04 |
| 2 | 5.0464 | 34.45 | 34.59 | −0.14 |
| 3 | 4.9422 | 34.98 | 35.12 | −0.14 |
| 4 | 4.8354 | **34.17★** | **34.34** | **−0.17** |
| 5 | 4.7971 | 35.05 | 34.69 | +0.36 |
| 6 | 4.6649 | 34.41 | 34.55 | −0.14 |

- best **34.17%**（ep4）vs P184 **34.34%** ⇒ **−0.17pp**，
  落在 dev 噪声地板 ±0.77pp 内；距筛选线 33.80 差 0.37pp，距硬约束 33.64 差 0.47pp。
- **train ctc 与 P184 几乎逐位相同**（5.410/5.046/4.942/4.835/4.797/4.665
  vs 5.412/5.049/4.934/4.838/4.789/4.671）⇒ **33.3pp 的 train–dev 间隙没有缩小**，
  判据第二条不满足。

**判定：NEGATIVE（无差别）**，不进 paired_eval / 2 seed / test，不跑 test。

## 4. 机制诊断：正则到底施加了没有？

为避免把「实现没生效」误判成「机制无效」，对 `p208-d2-cons-best.pt` 与
`p184-tfnet-freeze-best.pt`（同 init、同 seed、同为 ep4 best）做了权重差异分析：

| 模块 | 张量数 | 完全相同 | 说明 |
|---|---|---|---|
| `conv2d` | 212 | **212 (100%)** | P184 本来就冻结 ⇒ 符合预期 |
| `temporal_model` / `temporal_model1` | 32 | 0 | 全部变化 |
| `conv1d` / `conv1d1` | 24 | 0 | 全部变化 |
| `classifier11~55` | 5 | 0 | 全部变化 |

- **61 个可训练张量全部变化**（如 `temporal_model.rnn.weight_hh_l0` 差 3.38e-02、
  `classifier33.weight` 差 1.63e-02）⇒ **一致性项确实产生了梯度、不是空操作**。
- ⚠️ 初看「212/273 张量完全相同」容易误判为「正则没生效」，实为冻结主干，
  **已纠正**。
- ⚠️ **附带发现（诚实记录）**：第二视图前向会使 `conv1d/conv1d1` 的 **BN running stats
  额外更新一次**（`conv1d.temporal_conv.1.running_var` 差 4.899、
  `running_mean` 差 1.032，是所有张量里差异最大的）。因此 D2 严格来说**不是纯单变量**：
  除了一致性项，还多了一次 BN 统计更新。这是 R-Drop / CR-CTC 类「双前向」方法的固有实现特性
  （两次 forward 都会过 BN），不是代码 bug，但后续若复跑应在第二视图用 `no_grad` +
  冻结 BN 才能完全隔离变量。

## 5. 判定与解读

- **E3 判负（无差别）**。与 E1(+1.20pp 明确负)、E2(+0.21pp 噪声内)、
  P206(−0.24pp 噪声内) 相比，D2 的 −0.17pp 量级相当，
  **是四条候选里最好的一条，但仍远不足以越过噪声地板**。
- 机制层面：一致性约束确实改变了可训练参数，却几乎没有改变 train ctc 轨迹、
  也没缩小 train–dev 间隙 ⇒ 在「冻结主干 + 6 轮 + 4973 条」预算下，
  **过拟合不是由「两次前向的输出不一致」主导**，或 λ=0.3 的梯度相对主 CTC 太弱，
  6 轮内不足以产生可测的泛化收益。
- 与 P198(Conformer 适配器)、P196(WE/WC) 同属「在冻结主干上加轻量训练项」族群，
  结果一致地落在噪声内 ⇒ **这条路线可以关闭**。

## 6. 产物

| 类型 | 路径 |
|---|---|
| 代码 | `tools/blank_gov/p208_patch_d2.py`（补丁）、`p78_train_official_tfnet.py`（`--cons-*`） |
| 权重 | `artifacts/checkpoints/p208-d2-cons-{best,last}.pt` |
| 收据 | `artifacts/metrics/blank-gov/p208-d2-cons-official-tfnet.json`、`p208-d2-judge.json` |
| 日志 | `artifacts/logs_P208_d2_smoke.log`、`logs_P208_d2_train.log` |
| 诊断 | `tools/blank_gov/_d2_weight_diff.py`、`_d2_weight_diff2.py` |

## 7. 净结论（不变）

部署双轨：精度优先 = 官方 @224 零样本 test 32.74（官方逐句均值）；
速度优先 = P184@160 test 33.98（语料级）。dev 噪声地板 ±0.77pp。
D1 / D2 / D3 均判负，D4 训练中。
