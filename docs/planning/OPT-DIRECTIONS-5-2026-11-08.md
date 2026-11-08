# 第五轮优化方向（2026-11-08 18:55，仅方案，未启动实验）

> 输入：前四轮全部实验记录 + H2-a 新结果 + 本地 25 篇论文库 + 2026-11-08 系统性文献检索
> 依据等级（用户约定）：【有】能指出是哪篇哪句｜【间接】论文讲相近机制｜【无】无支撑，我拍的
> 状态：**仅方案**。任何方向启动前须按项目规范写死判据。

---

## 0. 一句话结论

前四轮把「类别信息 / 后验融合 / 参数自适应」三个维度系统穷尽（13+ 判负），
但有一个自由度**从头到尾没被碰过**：**输入时间分辨率 / CTC 对齐密度**。
它不是「再加一个损失项」，而是改模型看到的时间轴本身；
且有直接文献消融 + 本地两个硬诊断支撑，且可以**零训练、参数无关、纯推理**验证。
若它也判负，按累计证据应接受天花板收尾。

## 1. 现状锚点（本轮新核算的硬数字）

| 量 | 实测 | 说明 |
|---|---|---|
| dev 输入帧数 | min 81 / 中位 **181** / 均值 186.2 / max 464 | 差异 5.7 倍，天然诊断变异 |
| 参考 gloss 数 L | 中位 5 / 均值 **5.52** / max 13 | 515 句 |
| **输入帧/词** | **≈ 32.8**（181 / 5.52） | 介于 CSL-Daily 18 与 MSLR 58 之间 |
| **编码器时间步/词 `T_conv/L`** | p10 6.00 / **中位 8.50** / 均值 10.07 / p90 11.80 | `p188_logp_p184.pt` 的 `logp.shape[0]` 实测 |
| 非 blank 主导帧占比 | **12.35%**（clean，H2-a 同管线） | 87.65% 是 blank |
| **每个 gloss 被监督帧数** | **1.06**（D3 / P209） | CTC 峰值对齐坐实 |
| 错误构成 | 替换 **67.5%** / 删除 24.5% / 插入 8.1%（P191） | **主矛盾仍是替换** |
| H2-a 关键耦合 | 中心遮挡使非 blank 12.35→**6.10%**、`del` 193→**1378**（×7.1） | 非 blank 帧数腰斩 ⇒ 删除爆炸 |

**本地从未试过这个维度**（三条独立核查）：
1. `stage-P120-beam-search.md` §L283 当年把「CTC 空白率调节 / **帧率下采样**」列为
   「值得试的两个方向（都低成本）」之一，**从未执行**，之后再无人提起；
2. `LIT_PLAN_论文逐篇对照.md` §L234–235：本地 25 篇论文里
   `frame rate` **0 次**、`temporal resolution` **0 次**、`frame stride` **0 次**、
   `number of frames` 0 次，官方论文也无任何帧率/下采样表述；
3. 已判死项逐一排除：P187 是**空间**分辨率 160/192/224 + hflip；P185 是**损失层**多尺度 CTC；
   `TemporalRescale(0.2)` 只在**训练侧**随机 0.8–1.2×；P172 是**跨帧率蒸馏**的师生对齐问题；
   P183 是解码期长度惩罚；P203 强时序增广（时间掩码）判负。
   ⇒ **没有任何实验在推理期改变过输入时间分辨率。**

## 2. 文献依据（逐条标级）

- 【有】**Min et al., ICCV 2025W《A Closer Look at Skeleton-based CSLR》**（MSLR Workshop）：
  Table 3 **downsampling ratio 消融，ratio = 0.5 最优**；原文
  "framerate plays a critical role in sign language recognition"、
  "MSLR ~58 frames per gloss ... **such a high frame rate does not lead to better performance**"、
  "the importance of **selecting an appropriate input frame rate**"。
  参照带：Phoenix-T ≈17、CSL-Daily ≈18 frames/gloss。
  ⇒ **帧率是可调超参，且不是越高越好，存在最优点。**
- 【有】**SignFlow, IEEE Access 2025**（DOI 10.1109/ACCESS.2025.3554618）：
  动态下采样到 **6 fps** 是精度/速度最优平衡；消融明确写
  **"rapid signing, which led to increased deletion errors"**（打手语越快，删除错误越多）
  ⇒ 直接把「时间分辨率」与 `del` 挂钩。
- 【有】**RAIDAL, BMVC 2026**（github.com/verlab/RAIDAL）：CTC 每个 gloss 只在
  1 或少数 timestep 出尖峰、其余几乎全 blank；峰值窗口半径
  `R = ½(T_sign × FPS / s − 1)` 显式由帧率与下采样 stride 决定，实测只保留 **20–40%** 时间步。
- 【有】**arXiv:2406.02560《Less Peaky and More Accurate CTC Forced Alignment by Label Priors》**：
  训练期把后验除以 `prior^α` 后，音素平均预测时长 **21 ms（≈1 帧）→ 74 ms**（真人标注 82 ms），
  边界误差降 **12–40%** ⇒ 证明「把尖峰摊成区间」在 CTC 上是可行的（但见 §4 高重合风险）。
- 【有】**MTCNet, Springer 2025**：基线 deletion 占全部错误 **54.3%**，
  其 CFMR/TCAR 把它降到 38.5%，错误结构从 del 主导转为 sub 主导。
  ⚠️ **本地前提不同**：我们是替换 67.5% 主导、删除仅 24.5%，不可照搬其动机。
- 【间接】**Zhao & Bell, IEEE SLT 2024**（CTC 拓扑）：增加 CTC 状态数改善对齐、缓解 peaky；
  **增大 minimum traversal frame 反而恶化对齐** ⇒ 时间尺度是敏感量。
- 【有】本地 **D3 / P209**（每词 1.06 帧）、**H2-a / P227**（非 blank 腰斩 → del ×7.1）、
  **P191**（错误构成）、**P120**（当年列为待试方向）。
- 【无】「推理期帧步长扫描」在 CE-CSL / 冻结主干 PEFT 设定下无直接先例；
  文献里的 U 型帧率曲线是跨数据集推导，不是实测（domain_search 已如实标注该证据缺口）。
- 【无】本方案里两个相反假设的提出与扫描档位，属工程判断。

## 3. 两个方向相反的假设 —— 让扫描自己裁决

| | H_low（应降采样） | H_high（应增密） |
|---|---|---|
| 依据 | 本地**输入帧/词 ≈32.8** 高于 CSL-Daily 18 / Phoenix-T 17 经验带；ICCV2025W 的 ratio=0.5 最优 ⇒ 减半到 16.4 正好入带 | 本地**`T_conv/L` 仅 8.50**、每词仅 **1.06** 个非 blank 帧、blank 87.65% ⇒ 可用 spike 位置严重不足，增密能给 CTC 更多发射机会 |
| 支持文献 | ICCV2025W、SignFlow 6fps | D3、RAIDAL、arXiv:2406.02560、EMNLP26 Beyond CTC Spikes |
| 预测 | `step=2` 跳帧应改善 | 每帧重复 ×2 应改善 |

**两者不能同时成立，且各自都有【有】级依据 ⇒ 不预设立场，一次扫描同时覆盖。**
这正是本方向值得做的原因：它是本项目少见的「有可证伪对立假设」的实验，
无论结果指向哪边都能给出机制结论，而不是又一次噪声内抛光。

⚠️ **口径诚实说明**：ICCV2025W 的 17/18/58 是**输入帧/词**（其 ratio 消融在数据管线），
本地 8.50 是**编码器时间步/词**（conv2d 约 4× 时间下采样后），两者**不可直接比大小**。
上表只是把两个口径各自能推出的假设并列，绝不把「8.50 < 17」当成已证实的缺口。

## 4. 候选方向（按性价比）

### I0. 推理期输入时间分辨率扫描 —— 优先级 1，零训练 / 参数无关 / 纯推理

- **唯一改动**：`RGBSeqDataset` 读出的帧序列在时间维做确定性重采样，其余逐项沿用 P184
  （`p184-tfnet-freeze-best.pt` / 160 / `OFFICIAL_PAD=True` / `RGB_FIX=True` /
  `BLANK_BIAS=0` / dev 515 句全量 / `eval()` + `no_grad` / 主干全冻结 / `batch=1` 不跳句）。
- **固定档位（启动前写死，不事后增删）**：
  `step=2`（每 2 帧取 1，输入帧/词→16.4）、`step=1`（基线）、
  `repeat×2`（每帧重复一次，`T_conv`≈92）、`step=3`（输入帧/词→10.9）。
  另可选 `0.75×` 线性插值作稳健性参照（不参与主判据）。
- **每组报**：`WER_official` / `uWER` / 低频桶 / 高频桶 / `ins,sub,del` / `micro` /
  **非 blank 帧占比** / **每词非 blank 帧数** / **`T_conv/L`** /
  `per_item`（配对 bootstrap 用，句序 sid 必须逐句同序断言）。
- **判据（预注册）**：任一档同时满足
  ① `WER_official` 比 `step=1` 好 **≥2.00pp**；
  ② 配对 bootstrap（20000 次）95% CI 下界 **>0**；
  ③ 护栏：高频桶恶化 ≤1pp（R3）、插入增幅 ≤10%（R5）、micro 恶化 ≤0.30pp（R6）；
  ④ 低频桶不显著恶化（单独 CI 下界不跨到明显负向）。
  ⇒ 才算 `SIGNAL`，才值得进入 2 seed 复跑；否则 `FAIL`，**关闭该线、不投训练、不进 test**。
- **为什么门槛 2.00pp**：本地 dev 噪声地板 ±0.77pp（配对 1.556pp）；
  这是前提扫描，只有明显信号才值得后续投入；也避免 H2-a 那种「单组极端值刷过线」的误判
  —— 故额外要求 CI 下界 >0 与三项护栏。
- **与已判死项的区别**：见 §1 第 3 条。核心是「改输入时间轴」而非「改损失 / 改后验 / 改参数」。
- **成本**：4–5 档 × 515 句 ≈ **20–30 min GPU**，零训练；纯推理，不碰 test。
- **诚实预期**：【无】大概率全落在噪声内；乐观情形 ≤1–2pp，
  因为主矛盾是替换 67.5%，帧密度主要作用于对齐/删除（24.5%）。

### I1. 删除错误与 `T/L` 的相关性诊断 —— 优先级 1（并行），零成本 CPU

- 用**已落盘**的 `p188_logp_p184.pt`、`p227-h2a-clean-uwer.json`（含 `per_item`）与 dev 标注，
  纯 CPU 算：每句 `T_conv/L`、每句 `del` 占比、非 blank 帧数，检验
  「`T/L` 越低是否 `del` 越多」及分项（高/低频桶）差异。
- 判据（预注册）：`T/L` 与 `del_rate` 的 Spearman |ρ| ≥0.20 且 bootstrap CI 不含 0
  ⇒ 对齐密度与删除确有耦合，为 I0 结果提供机制解释；否则判「不相关」，I0 的立论减弱。
- **不产生候选模型**，只做解释；成本 ~1 min，可与 I0 并行（不占 GPU）。

### I2. 训练期对齐增密 —— 优先级 3，**仅在 I0 出现 SIGNAL 后才考虑**

- 依据【有】arXiv:2406.02560 label-prior CTC、EMNLP26《Beyond CTC Spikes》GTM、
  DPLR 稠密伪标签、EnCTC/EsCTC（NeurIPS18）、本地 `06_SMART`。
- ⚠️ **高重合风险，必须写明**：
  - **F1（训练期 logit adjustment）已明确负向**（35.47，train ctc 5.41→5.77、
    插入 90→39），与 label-prior CTC 机制高度相近（都是拿类先验压 blank）；
  - P185 多尺度 CTC 结论「**额外监督破坏对齐**」；P196 WE/WC、D2 一致性正则、
    D3 TFD 边界 `corr=0.29` STOP、P202/P203 增广全部判负；
  - ⇒ 若 I0 无 SIGNAL，**不投这 6 轮**，不要重复第四轮的失败模式。
- 若 I0 有 SIGNAL：须新写预注册判据（含 R1–R8 + 非 blank 帧占比作为机制指标），再启动。

### I3. 不推荐 / 已排除

| 方向 | 依据 | 不推荐理由 |
|---|---|---|
| 换骨干 / 多尺度时序融合（Signer-Invariant Conformer Auxiliary Block、DFConv 上下半区） | 【有】arXiv:2508.09372、DPLR | P198 Conformer 适配器、P200 整段替换（40.04，差 5.70pp）已证 6 轮内随机初始化新模块拟合失败；8G 不匹配。注：DFConv 论文称分区对缩放/位移更鲁棒，可解释 H2-a 的遮挡坍塌，但不构成投入理由 |
| CTC 拓扑改状态数 | 【间接】Zhao & Bell SLT24 | 需改损失递推 + 重训；P185 已证额外对齐监督破坏对齐 |
| TTA / 参数自适应 H2-b | 【有】H2-a 机械 PASS | 三类自然 mild 偏移全部 <2pp、统计量未漂移；唯一过线的 `occlud_mild` 是像素被物理删除，无监督自适应补不回（见 `stage-P227...` §4.1） |
| 检索增强 / LM 重排 / 类别重加权 / 蒸馏 / 通用增广 | 【有】本项目实测 | H1、D1、P197、G1、D4、E0/F0/F1/F2、P172/P174/P175、P202/P203 全部判负 |
| 为 H2-a 的遮挡坍塌做输入修复（相邻帧补洞等） | 【无】 | CE-CSL dev/test 同分布、test 无遮挡 ⇒ 优化一个不存在的场景，无部署价值 |

## 5. 执行顺序（全部预注册后再启动）

```
I1  零成本 CPU 相关性诊断（~1 min，可与 I0 并行）
I0  推理期时间分辨率扫描（4–5 档，20–30 min GPU，零训练）
     ├─ SIGNAL（≥2pp + CI 下界>0 + 三项护栏）──► 2 seed 复跑 → 才考虑 I2
     └─ FAIL ──► 关闭该线；按累计 13+ 判负接受天花板收尾
I2  仅当 I0 有 SIGNAL：训练期对齐增密（新写预注册，6 轮）
```

任一步落在 dev 噪声地板 ±0.77pp 内即判「无差别」，不进 test。
所有新实验仍必须同时报 `WER_official + uWER + 低频桶 + 高频桶 + ins/sub/del + micro`，
低频桶单独配对 bootstrap（R8，692 token，不套 ±0.77pp）。

## 6. 诚实预期与收尾条件

- 主矛盾是**替换 67.5%**（近义 gloss 判别边界，低频实词），删除只 24.5%；
  帧密度即使改善对齐，总量收益上限有限（乐观 ≤1–2pp）。MTCNet 的「deletion 主导」前提在本地不成立。
- 但 I0 是**目前唯一**同时满足「零训练 / 参数无关 / 不占 6 轮预算 / 有直接文献消融 /
  本地从未试过 / 有可证伪对立假设」的方向，性价比高于再投任何训练项。
- **若 I0 判负**：与 `OPT-DIRECTIONS-3` §6、`OPT-DIRECTIONS-4` §4 一致，
  合理结论是**接受天花板收尾**，而不是继续找新损失项。
  天花板不变：官方 @224 零样本 test **32.74** / P184@160 test **33.98** / P150 **53.73**；
  唯一稳定正向仍是 P184 冻结主干 PEFT。

## 7. 附注：一个仍未做的零成本口径复评（不算性能提升）

P189 确认官方口径 = **逐句均值 + beam10**；但 **P184@160 的 test 33.98 报的是语料级**，
从未在同一官方逐句均值口径下复评。纯推理即可统一双轨报数，不改变模型、不得当作收益。
（此前 `OPT-DIRECTIONS-2026-11-07` §3 已提，一直未执行。）

## 8. 来源清单

- 本地：`docs/experiments/stage-P120-beam-search.md` §L283、
  `stage-P209-D3-tfd-probe.md`、`stage-P227-H2a-tta-precondition.md`、
  `stage-P191-*`、`stage-P184-tfnet-peft.md`（P185/P187）、
  `docs/planning/LIT_PLAN_论文逐篇对照.md` §L234、
  `docs/planning/OPT-DIRECTIONS-{1,2,3,4}-2026-11-0{7,8}.md`、
  `artifacts/metrics/blank-gov/p227-h2a-probe.json`、`p188_logp_p184.pt`。
- 外部（2026-11-08 检索）：Min et al. ICCV 2025W (MSLR, cvf openaccess)；
  SignFlow IEEE Access 2025 DOI 10.1109/ACCESS.2025.3554618；
  RAIDAL BMVC 2026 (github.com/verlab/RAIDAL)；
  arXiv:2406.02560 Less Peaky CTC Forced Alignment by Label Priors；
  MTCNet Springer 2025 DOI 10.1007/s40747-025-02156-5；
  Zhao & Bell IEEE SLT 2024 CTC topology；
  Zhu et al. EMNLP 2026 (2026.emnlp-main.578)；
  DPLR / Self-Sufficient Framework；Zhu et al. arXiv:2508.09372；
  FSR (Transducer blank-skip, AISHELL-1)。
