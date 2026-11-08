# 夜间无人值守实验链（2026-11-07 23:40 → 2026-11-08 09:00）

> 用户指令：明早 9 点前自动往下跑、自行决策、跑完继续找别的方向接着运行。
> 链脚本 `tools/blank_gov/night_chain.sh`｜状态文件 `artifacts/cache/night_state.txt`｜
> 主日志 `artifacts/logs_NIGHT_chain.log`｜硬停 09:00。
> **未跑 test、未 commit、未 push。**

## 1. 统一判据（23:30 预注册，所有实验共用，禁止事后调整）

项目此前 D1–D4 + E0 连续判负，暴露两类「假改善」：
D4 是**错误在类别间搬家**（uWER/低频桶改善，但高频桶与 micro 恶化）；
E0 是**删→插的替换**（uWER/低频桶只统计 ref token 的 sub+del，看不见插入爆炸）。
因此判据改为**主看尾部 + 双护栏**：

| 编号 | 规则 |
|---|---|
| R1 | uWER 改善 ≥ 2.0pp |
| R2 | 低频桶 WER 改善 ≥ 2.0pp |
| R3 | 高频桶恶化 ≤ 1.0pp |
| R4 | WER_official 退化 ≤ 1.0pp |
| R5 | 插入数相对基线增幅 ≤ 10%（E0 教训） |
| R6 | micro sub+del 恶化 ≤ 0.30pp（D4 教训） |
| R7 | 训练类：6 轮 best dev ≤ 34.80（零训练实验跳过） |

全部满足才 PASS → 才值得投 2 seed + 配对 bootstrap；否则 FAIL，不进 test。
另报配对 bootstrap 95% CI（句子级，20000 次）与非 blank 帧占比，不参与 PASS。

统一工具：`night_eval_full.py`（全指标+逐句编辑数）、`night_judge.py`（判定）、
`night_report.py`（自动阶段报告 + 桌面镜像 + 整夜汇总）。

## 2. 队列与依据等级

| 阶段 | 方向 | 依据 | 成本 |
|---|---|---|---|
| S0 | E1（CSGC 跨句 gloss 原型）收尾 + 统一口径复判 | 【有】AAAI 2024 CSGC；本地 D4 极端长尾、part3 §20 实词 AUC 0.93–1.0 | 已在训练 |
| F0 | 插入率感知解码扫描 τ×β（零训练） | 【有】Menon ICLR21；本地 E0 插入爆炸反证据；【无】联合网格属工程尝试 | ~1 min CPU |
| F1 | **训练期 logit adjustment（LA-CTC）** | 【有】Menon ICLR21 / Balanced Softmax；本地 E0 前提 `corr(log freq, logp)=0.8756` 成立 | 6 轮 ≈3h |
| F2 | **视觉-文本语义原型对齐（VTA）** | 【有】FGXM（PRL 2025，即 uWER 原论文）；【间接】CVT-SLR CVPR23 / ACMC25 | 6 轮 ≈3h |
| E3 | 手形聚类前提探针（零训练） | 【有】EMNLP 2023 / HandsOn FG2025 | ~1 min CPU |
| S5 | 训练期×解码期组合扫描（**探索性，不判 PASS/FAIL**） | 【无】纯诊断，为下一轮选题 | ~10 min |

**F1 的立论**：E0 只证伪了 **post-hoc** τ 调整（低频桶 −9.1pp 但插入爆炸、总量崩到 45.21），
明确留下「训练期让模型自行补偿」这一未验证口子；且 LA 是 **per-class logit 偏置**，
与 D4 的 per-sample loss 加权、E0 的解码后处理都是不同机制。

**F2 的立论**：48% 的类只有 1 个 train 样本，决策边界欠定；
`NormLinear` 列范数被强制相等（E0 测得 `corr(freq,‖w‖)=0.10`），类原型缺语义邻域结构。
用本地 Qwen2.5-1.5B 抽 3516 个 gloss 的文本嵌入 → PCA 到 1024 固定，
对类原型加低频加权 cosine 正则，无新参数、推理零开销。

## 3. 执行与失败处理

- **一次只跑一个训练**：链先等 E1 链退出，再依次 F1 → F2；GPU 全程独占。
- 每个训练：备份 p78 → 打默认关闭的开关 → `py_compile` + `--help` 门禁 → 冒烟 → 正式 6 轮
  （配方逐项沿用 P184：160 / 冻结 conv2d / lr 2e-5 / seed 42 / p184-init）→ 全指标评估 → 判定 → 报告。
- 训练启动需剩余 ≥ 3.5h，训练本身 `timeout = 剩余 − 30min`，超时用已落盘 best ckpt 继续评估。
- 补丁失败 / 冒烟失败 / 无 best ckpt → 该阶段跳过并记录，链继续下一阶段。
- 若时间富余，S5 做探索性组合扫描；09:00 前生成
  `docs/experiments/NIGHT-RUN-2026-11-08.md` 总汇总（WSL + 桌面）。
- 跨会话兜底：一次性自动化 **08:35** 检查链状态、必要时补汇总、出早间汇报；
  其指令严格禁止 push / commit / test / 删除 / 新增长训练 / 改判据。

## 4. 启动前已验证（非预计）

- 9 个脚本 `py_compile` + `bash -n` 全过；两个补丁在 p78 副本上**叠加试打**成功
  （LA 7 处含 4 个辅助 CTC 头，VTA 3 处），叠加后 `--help` 新参数可见。
- F0 实跑 25 组合、E3 实跑、judge/report 代码路径自比验证均通过。

**F0 预先结果**（链会重跑）：τ=0.3 / β=2.0 是护栏内 uWER 最低组合 →
WER **34.31**（基线 34.34）、低频桶 **68.06**（−2.46pp）、uWER 58.74（−1.54pp）、
插入 89（基线 90）。⇒ blank bonus 确实把 E0 的插入爆炸压回去了，尾部也有改善，
但 uWER 增益 <2pp，按 R1 判负。这同时是「训练期补偿值得试」的直接依据。

**E3 预先结果**（链会重跑）：1-NN top1 **27.44%**（随机 0.617%，44×），
但 KMeans AMI **0.0599** < 0.10 ⇒ 按预注册 AND 规则 STOP。
诚实记录：1-NN 是显著信号，说明样本级描述子含 gloss 身份信息；
要重开须新写预注册判据（改用近邻检索而非全局聚类），不得事后放宽。

## 5. 查看方式

```bash
wsl -e bash -lc 'cd ~/FYP/domain-bounded-cslr && \
  cat artifacts/cache/night_state.txt && tail -30 artifacts/logs_NIGHT_chain.log'
# 各阶段收据：artifacts/metrics/blank-gov/*-judge.json
# 阶段报告：docs/experiments/stage-p21*.md（桌面镜像同路径）
```

⚠️ 项目铁律：WorkBuddy 重启会杀掉 WSL 后台进程。若中途重启，08:35 的自动检查会发现链中断，
但**不会自动重启训练**，需要明早手动续跑。
