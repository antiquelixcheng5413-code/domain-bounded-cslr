# P170 — Dropout 正则探针（VAC@160, dropout=0.3, RGB）

> 状态：启动中（2026-10-26）
> 对照基准：P150（`p150-rgbfix`，VAC@160-RGB，dev 53.73%）
> 唯一变量：**主干空间特征后加 `nn.Dropout(0.3)`**

## 1. 动机（为什么是 dropout）
P169（VAC@160 更长训练 + 余弦重置 lr）出现典型过拟合：ctc 训练损失从 7.16 崩到 2.12，
但 dev WER 全程（best 57.28%）无一低于 P150 的 53.73%。这直接指向**"缺正则 / 容量过配"**，
而非判别力本身。dropout 是最规范的单变量正则探针——若它能把 53.73% 再压低，即为真实增益；
若无效，则进一步坐实"瓶颈不在正则，而在判别力/数据规模"。

## 2. 实验配置（与 P150 的唯一差异 = dropout）
| 项 | P150 | P170 |
|---|---|---|
| 主干 | VAC(resnet18+TC+BiLSTM) | 同左 |
| dropout | **0（默认）** | **0.3（空间特征后）** |
| 分辨率 / 通道 | 160 / RGB(`--fix-rgb-order`) | 同左 |
| word-head | 0.3 | 同左 |
| 调度 | multistep[35,45] γ=0.2，55ep | 同左（**不重置 lr、不用余弦、不开 SWA**） |
| 训练 | 从零 | 从零（保持单变量干净） |
| 其他 | official-pad / ctc_align / warmup3 / seed42 / bf16 / batch2 / mem5.2e9 | 同左 |

- 标签：`p170-dropout03-rgb`
- 日志：`artifacts/logs_P170_vac160_dropout.log`

## 3. 验收判据（启动前写死）
- 评估口径：与 P150 完全相同的 `official_wer.py` 语料级 WER。
- 判定（照铁律）：与 P150 逐轮配对 + best WER 对比。
  - **Δ ≥ 2pp 且胜率 ≥ 75%** ⇒ 记为"dropout 有效增益"；
  - 否则 ⇒ 记为"噪声内 / 正则无效"，并把"dropout 探针"加入判死清单。
- 噪声基准沿用 1.556pp；报数必须 mean±std+胜率+SNR。
- **不做方向性判断直到跑完**（冒烟只验能否跑）。

## 4. 代码改动
- `external/TFNet/Net.py`：`moduleNet` 增 `dropout=0.0`；VAC 分支加 `self.drop=nn.Dropout(dropout)`，
  forward 在 `x=self.conv2d(x)` 之后应用。**默认 0 恒等，历史实验不受影响。**
- `tools/blank_gov/p78_train_official_tfnet.py`：增 `--dropout` 参数并透传。
- 编译 + `--help` 自检通过（`COMPILE_OK`，dropout 出现在参数列表）。

## 5. 预期与风险
- 预期：弱。若过拟合确为主因，可能收回 53.73% 甚至略降 1–2pp；若数据本身判别力不足，dropout 也填不满 8.6pp。
- 风险：dropout 可能拖慢收敛或略损训练损失，但 55ep 足够观察。
- 成本：~25min/轮 × 55 ≈ 23h @160（5060 8G）。
