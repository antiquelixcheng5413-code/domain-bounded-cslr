# P171 — TFNet(resnet34MAM) @160+RGB 微调（从发布权重）

> 状态：冒烟通过（2026-10-27），正式微调已启动（30 ep，后台）
> 方向来源：用户选 **B（微调发布权重→160+RGB）**，推翻此前"接受 VAC@160 53.73% 天花板"的结论。

## 0. 背景与立论

- 本地发现官方发布权重 `桌面/中文手语识别/官方权重/TFNet-CE-CSL-CSLDaily-32.46.pth`
  （879.6MB，**完整 TFNet = resnet34MAM 骨干**，313 键 `moduleNet_state_dict`，与 `Net.moduleNet("TFNet")` strict 逐键吻合）。
- 该权重在 CE-CSL **dev 实测 33.92%**（README 标 test 32.46%，是 CE-CSL 真货，非 CSL-Daily）。
- 我们 from-scratch 只训过 VAC(resnet18) 最佳 53.73%，**从没完整训过 TFNet**（p158 死在 ep1）。
- ⇒ 此前"53.73 是天花板、8.6pp 是训练赤字"的结论因发现本地发布权重而**失效**；
  真实缺口是 **骨干差距 VAC→TFNet/MAM（20pp 量级）**，而非数据/架构瓶颈。

## 1. 实施路径（风险最低）

- **转换器** `tools/blank_gov/make_p171_init.py`：用与 p78 完全相同的 `Net.moduleNet("TFNet")` 构建，
  载入发布 `moduleNet_state_dict`（strict=True，missing=0/unexpected=0），
  新建 optimizer(Adam 1e-4/wd 1e-4) + `word_classifier`(随机，p78 自加多任务头，发布权重无此键)，
  打包成 p78 能直接 `--resume` 的初始化 ckpt（`p171-init-tfnet160.pt`，342MB，epoch=0）。
- p78 训练循环**一行不改** → 复用已验证的 loss/head/评估路径，风险最低。
- 输入 160+RGB（已修通道顺序 `--fix-rgb-order`），回收发布权重 224→160 的分辨率代价。

## 2. 冒烟结果（1 epoch，2026-10-27）

| 指标 | 值 |
|---|---|
| 主干 | TFNet / resnet34MAM，73.27M 参数 |
| 输入 | 160×160 RGB + bf16 |
| 单 epoch 实测耗时 | **~51 min**（含 496-batch dev 评估） |
| GPU 显存 | 4798 MiB / 8GB（稳装，3D 主干比 VAC 重 ~2×） |
| **ep001 dev WER** | **42.47%** |
| 对照：发布 224 基线 | 33.92% |
| 对照：VAC@160 from-scratch | 53.73% |

- **结论**：比 VAC@160 低 **11.3pp**（骨干效应坐实）；比 224 基线高 8.55pp（分辨率落差 + 仅 1 轮适配）。
- 无 load/shape/NaN 报错，全链路跑通。冒烟达成"验证可跑 + 测速"双目标，且给出强正向信号。

## 3. 正式微调方案（已启动）

- 命令（后台）：`p78 --tag p171-finetune-tfnet160 --module TFNet --img-size 160 --batch 2 --mem-budget 7e9
  --amp-bf16 --lr 1e-4 --weight-decay 1e-4 --hidden 1024 --seed 42 --official-pad
  --word-head 0.3 --word-boundary ctc_align --warmup-ep 3 --fix-rgb-order
  --epochs 30 --total-epochs 30 --resume artifacts/checkpoints/p171-smoke-tfnet160-last.pt`
- 续训起点：冒烟 last ckpt（epoch=1，WER 42.47），跑 ep2→30（29 轮）。
- 调度：沿用 init 的 MultistepLR 里程碑 [19,25]（按 30 总轮缩放，gamma 0.2）。
- 预计总耗时：29 × 51min ≈ **25h**（5060 8G，单机串行）。
- 日志：`artifacts/logs_P171_finetune.log`

## 4. 写死判定（收尾时核对）

- **成功（回收分辨率代价）**：微调后 dev WER 落在发布 224 基线 33.92 的 **~2pp 内（≤36%）**。
  理想：逼近/匹配 33.92（说明 160 适配几乎无损）。
- **部分成功**：落在 36%~42.47%（仍有分辨率代价但显著优于 VAC@160）。
- **负向（极不可能）**：> 53.73%（劣于 VAC@160）→ 说明骨干/适配出问题，需排查。
- **对照价值**：vs VAC@160 53.73，直接量化"骨干效应"（VAC→TFNet/MAM）这一 20pp 量级缺口。

## 5. 风险与备注

- 25h 长任务：用 Bash 工具 `run_in_background` 托管（会话回收不杀），勿用 nohup/& 自起。
- CUDA 999 瞬时故障：若崩，从 best ckpt 续训（发布权重完好，无数据丢失风险）。
- 手机小程序已连上电脑 WorkBuddy → 可远程看进度 / 停任务。
- 本实验不属"纯 from-scratch 复现"，是迁移微调；FYP 叙事上归为"基于官方 TFNet 权重在轻量 160 配置下的适配/蒸馏式部署"。

## 6. 收尾结论（2026-10-28，已停）

**判定：净负向（判死）。** 过夜训练 ep9→ep19 曲线已明确退化，用户授权"跑到起床"后于 ep19 主动停止。

| epoch | dev WER | 说明 |
|---|---|---|
| ep1 | 42.47% | 冒烟后 |
| ep2 | 40.43% | ↓ |
| **ep3** | **39.90%** | **best（全程最低）** |
| ep6 | 41.13% | 回升 |
| ep9 | 43.91% | 跳升 |
| ep12 | 44.30% | 平台 |
| ep14 | 46.02% | 平台高点 |
| ep19 | 45.60% | 停止点（刚过 lr 首次衰减，无回升）|

- **best 锁死 ep3（39.90%）**，之后 16 轮只退不进；ep7 起稳在 43–46% 平台（均值 ~44.4）。
- **过拟合签名干净**：ctc 训练损失从 9.0（ep2）单调降到 7.72（ep19），dev WER 反涨 ⇒ 微调在摧毁预训练泛化。
- **关键对照（P156 零样本，2026-10-28）**：TFNet@160 RGB **零样本 36.81%** 比本实验微调 best 39.90% 还低 3pp。
  ⇒ 在 160+word-head 管线里微调发布权重是**净负向**；发布权重的最优用法是**原封不动零样本用 160 输入**，而非加辅助头重训。
- **战略定位**：本项目没有"训出更好模型"。真贡献 = (a) 坐实并量化 20pp 骨干缺口（VAC 53.73 → TFNet 33.92）；(b) 量化 TFNet 在 160 下的温和分辨率代价（+2.89pp 零样本）；(c) 证伪"微调发布权重有用"。
- best ckpt（39.90%@ep3）与 last 完好，`p171-finetune-tfnet160-*.pt` 留作退化曲线证据。

## 7. 待办

- [x] 逐 epoch 退化曲线 + 与 33.92 / 53.73 对照（净负向，判死）
- [x] P156 零样本 160 RGB = 36.81%（证明零样本 > 微调）
- [ ] A 线（from-scratch TFNet@160 同配置对照）：价值已降级——既然零样本部署已 36.81%，A 线最多逼近 33.92，但 ~6h/轮不现实，低优先
- [ ] 清理 p78 死参数 `--w-kd`（低优先）
