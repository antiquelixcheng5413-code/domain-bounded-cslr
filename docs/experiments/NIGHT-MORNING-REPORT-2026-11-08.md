# CSLR 夜间自动实验链 · 早间汇报（2026-11-08 08:35）

> 数据来源：WSL `~/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov/` 原始收据，
> 以及 `artifacts/logs_NIGHT_chain.log`、`artifacts/cache/night_state.txt`。
> 所有 verdict 均按收据字段原样转述，未重新判定；缺失项一律写「未生成」。
> 未跑 test、未 git commit / push、未启动新训练、未修改预注册判据。

## 1. 链状态：正常完成（不触发补做汇总）

| 检查项 | 结果 |
|---|---|
| `NIGHT CHAIN START` | 2026-11-07 23:40:38 CST，deadline 11-08 09:00（remaining 33562s） |
| `NIGHT CHAIN END` | 2026-11-08 08:07:20 CST，日志命中 1 次，remaining 3160s（比硬停早 52m40s） |
| `night_state.txt` | `FINALIZE` |
| 阶段轨迹 | INIT → WAIT-E1 → S0-BASE-EVAL → F0-DECODE → F1（补丁/冒烟/6轮/评估）→ F2（文本嵌入/补丁/冒烟/6轮/评估）→ E3-PROBE → S5-POSTHOC → FINALIZE，全部执行、无跳过 |
| `night_chain.sh` 进程 | 无 |

⇒ 链已完成，**不触发步骤 2（链仍在运行）与步骤 3（异常中断）**：
未运行 `tools/blank_gov/night_report.py --summary`，未启动任何训练。
总汇总此前已落盘 `docs/experiments/NIGHT-RUN-2026-11-08.md`（WSL 主副本 + 桌面镜像）。

⚠️ **现场另有非夜间链训练进程，未干预**：`tools/blank_gov/p222_g0_train_confusion.py`
（PID 662 / 671，STAT `Rl+`；GPU 7736 MiB / 利用率 100%）。`artifacts/logs_P222_g0.log` 尾部：
`dev unique confusion pairs = 606`、`train usable samples = 4973`、`forward 300/2487 samples=602`；
日志中还有一条 08:32:05 CUDA OOM 警告（申请 2,753,560,576 bytes，free 0），进程仍在运行。
按「不打断、不启动新训练」只记录当前阶段，未做任何处置。

## 2. 统一基线（所有 judge 的 `base` = `p184-base`，dev n = 515，语料级 WER_official）

| WER_official | uWER（774 类等权） | 低频桶（≤10，692 token） | 高频桶（2150 token） | 插入 | 替换 | 删除 | micro_subdel | n_ref |
|---|---|---|---|---|---|---|---|---|
| 34.41 | 60.41 | 70.66 | 18.56 | 90 | 696 | 192 | 31.25 | 2842 |

## 3. 逐方向汇总（Δ 为 `delta.*_improve_pp`，正数＝改善）

| 方向 / tag | kind | dev WER（Δ） | uWER（Δ） | 低频桶（Δ） | 插入数（90 → cand） | verdict | 未过判据 |
|---|---|---|---|---|---|---|---|
| E1 CSGC `p212-e1-night` | train | 34.41（0.00） | 59.26（+1.15） | 69.08（+1.58） | 89（−1.11%） | **FAIL** | R1、R2 |
| F0 τ×β 解码 `p215-f0` | decode | 34.31（+0.10） | 58.74（+1.67） | 68.06（+2.60） | 89（−1.11%） | **FAIL** | R1（唯一） |
| F1 训练期 LA-CTC `p213-f1` | train | 35.54（−1.13） | 65.76（−5.35） | 76.59（−5.93） | 39（−56.67%） | **FAIL** | R1、R2、R3、R4、R6、R7 |
| F2 视觉-文本对齐 `p214-f2` | train | 34.31（+0.10） | 60.85（−0.44） | 71.10（−0.44） | 66（−26.67%） | **FAIL** | R1、R2、R6 |
| E3 手形聚类探针 `p216-e3` | probe | 未生成 | 未生成 | 未生成 | 未生成 | **STOP** | 不适用（探针） |
| P218 门控 LA `p218`（链后追加） | decode | 34.55（−0.14） | 59.17（+1.24） | 68.64（+2.02） | 88（−2.22%） | **FAIL** | R1 |

**E1 / `p212-e1-night-judge.json`**：`verdict=FAIL`。训练 6 轮 best dev 34.41（ep2/ep6），
train ctc 5.4231 → 4.6696；高频桶恶化 0.56pp、micro 恶化 0.03pp；
配对 bootstrap delta 0.0pp、CI95 `[-0.742, 0.733]` 含 0；R3–R7 全部 true。
另有白天旧口径 `p212-e1-judge.json`，同为 `verdict=FAIL`。

**F0 / `p215-f0-judge.json` + `p215-f0-scan.json`**：代表组合 τ=0.30 / β=2.00（预注册选择规则）。
R2 达标（低频 +2.60 ≥ 2pp）、R3–R6 全过，仅 R1（uWER +1.67 < 2pp）未过 ⇒ 差 0.33pp；
高频桶恶化 0.74pp、micro −0.07pp；CI95 `[-1.638, 1.869]` 含 0。
scan 收据：25 组网格、`chosen=0.30/2.00`、`guard_count=2`、
`selection_rule="guard R3/R4/R5/R6 first, then min uWER; if none pass -> baseline (auto FAIL)"`；
网格自身基点 `0.00/0.00` = WER 34.34 / uWER 60.28 / 低频 70.52 / ins 90
（与 judge 的 base 34.41 / 60.41 / 70.66 并存，两份收据各自口径，本次未作调整）。

**F1 / `p213-f1-judge.json`**：`verdict=FAIL`，7 项 checks 中 6 项 false（仅 R5 为 true）。
高频桶恶化 1.95pp、micro 恶化 2.92pp；train best dev 35.47，train ctc 5.7772 → 4.9301；
nonblank 帧比 12.52% → 11.37%；配对 bootstrap delta −1.126pp、CI95 `[-1.879, -0.384]` **不含 0**。
已落盘报告结论（原样转述）：「类先验校正不适合在 CTC 训练期直接做」。

**F2 / `p214-f2-judge.json`**：`verdict=FAIL`，R1/R2/R6 false；高频桶恶化 0.84pp、micro 恶化 0.73pp；
train best dev 34.27，train ctc 5.4150 → 4.6635；CI95 `[-0.496, 0.726]` 含 0（噪声内）。

**E3 / `p216-e3-probe.json`**：`verdict=STOP`，预注册判据 `1-NN top1>=5% AND AMI>=0.10`。
字段：`n_files=514`、`n_usable=514`、`n_distinct_main_gloss=162`、
`one_nn_top1_pct=27.44`、`random_baseline_pct=0.617`、`lift_over_random=44.45`、`kmeans_ami=0.0599`。
**dev WER / uWER / 低频桶 / 插入数 = 未生成**（该 probe 收据不含这些字段），
且目录内无 `p216-e3-judge.json`。已落盘报告结论（原样转述）：「检索式优于聚类、不投 GTM」。

**S5 / `p217-posthoc.json`**：`kind=exploratory_no_verdict`，
note 原文「探索性诊断；不做 PASS/FAIL，不进 test，不用于报性能」⇒ **PASS/FAIL = 未生成**。
网格 tau `[0.0, 0.1, 0.2, 0.3]` × beta `[0.0, 1.0, 2.0]`，覆盖 4 个 ckpt
（`p184-base` / `p212-e1` / `p213-f1` / `p214-f2`）；各 ckpt 的 τ=0/β=0 点与 judge 一致：
WER 34.41 / 34.41 / 35.54 / 34.31，uWER 60.41 / 59.26 / 65.76 / 60.85，
低频桶 70.66 / 69.08 / 76.59 / 71.10，插入 90 / 89 / 39 / 66。
夜链日志与 `NIGHT-RUN-2026-11-08.md` 的原结论（原样转述，非本次新判）：
「所有 ckpt guard-best 均为 τ=0/β=0 ⇒ 训练期改动与解码期补偿无互补」。

**P218 / `p218-judge.json`**（08:13 落盘，属目录内 `*-judge.json`，非本次链 tag）：
代表组合 τ=0.50 / β=1.00 / θ=0.80；高频桶恶化 0.93pp、micro 0.21pp；
CI95 `[-1.927, 1.671]` 含 0；`verdict=FAIL`。已落盘报告：「门控版反而弱于 F0 全局版」。

**目录内其余 judge 收据（非本次链 tag，verdict 原样）**：
`p212-e1-judge.json` → `FAIL`（旧口径 dev 34.41 / uWER 59.26 / 低频 69.08）；
`p210-d4-judge.json` → `FAIL`（dev 34.27 / uWER 59.85 / 低频 69.51）；
`p208-d2-judge.json` → `NEGATIVE`（best ep4 dev 34.17、delta 0.17pp、noise_floor 0.77pp、
`screen_passed=false`、`hard_passed=false`）。

## 4. 最重要的发现

1. **链内五个方向 0 个 PASS**：E1 / F0 / F1 / F2 全部 `FAIL`，E3 为 `STOP`；S5 明确不做判定；
   链后追加的 P218 同样 `FAIL`。全部判据均为 23:30 预注册版本，未事后调整。
2. **F0 最接近过线**：唯一 R2 达标（低频桶 +2.60pp）且 R3–R6 全过，只差 R1 的 0.33pp，
   但配对 bootstrap CI 跨 0；P218 门控版 uWER +1.24 / 低频 +2.02 均小于 F0 全局版，同判 `FAIL`。
3. **F1 是唯一配对 CI 不含 0 的方向，且方向为负**：delta −1.126pp、CI `[-1.879, -0.384]`，
   插入 90 → 39、micro +2.92pp、6 项判据未过 ⇒ 明确负向；对比 F2 的 CI `[-0.496, 0.726]` 含 0（噪声内）。
4. **总量与尾部指标背离（base 字段直读）**：WER_official 34.41 vs uWER 60.41；
   低频桶 70.66（692 token）vs 高频桶 18.56（2150 token）。E1/F0 尾部改善而总量几乎不动，
   F2 总量 +0.10pp 而尾部恶化 ⇒ 所有 judge 在同一 base 下呈现尾部/总量不同向。
5. **E3 探针**：AMI 0.0599、1-NN top1 27.44%（44.45× 随机），`verdict=STOP`。
6. **S5**：训练期改动与解码期补偿无互补（日志与总汇总原结论）。

## 5. 收据清单与缺失项

已读取（11 份，全部存在）：`p212-e1-night-judge.json`、`p215-f0-judge.json`、
`p213-f1-judge.json`、`p214-f2-judge.json`、`p218-judge.json`、`p212-e1-judge.json`、
`p208-d2-judge.json`、`p210-d4-judge.json`、`p215-f0-scan.json`、`p216-e3-probe.json`、
`p217-posthoc.json`。

明确「未生成」的项：E3 的 dev WER / uWER / 低频桶 / 插入数（probe 收据无此字段）、
`p216-e3-judge.json`（不存在）、S5 的 PASS/FAIL verdict（收据明确不做判定）、
任何 test 集评估收据（按禁令未跑 test）。

## 6. 合规与现场状态

- 未 git push、未 git commit、未跑 test、未删除文件、未启动新训练、未修改预注册判据；
  未运行 `night_report.py --summary`（链已完成，步骤 3 条件不成立）。
- P222 G0 进程未干预；GPU 当前被其占用（7736 MiB / 100%）。
- 本汇报落盘：桌面 `docs/experiments/NIGHT-MORNING-REPORT-2026-11-08.md`；
  WSL 主副本同步至 `docs/experiments/`。
