# 阶段报告模板（CSLR / WER 主线）

> 用途：**每个实验运行结束后**必须产出一份，放在本目录，文件名 `stage-<tag>.md`。
> 本模板替换 `experiment-record-template.md` 用于 CSLR 主线——
> 旧模板的指标项（Top-1 / Macro-F1 / signer-independent / latency）是孤立词分类口径，不适用。
> 首次使用：P101b-ctrl（`stage-P101b-ctrl.md`）。

```markdown
# 阶段报告：<tag>

- **Experiment ID**:
- **Date**:
- **Git commit**: （`git rev-parse --short HEAD`，若有未提交改动需注明数量）
- **代码来源**: `external/TFNet`（是否直接 import 官方代码，有无重写）
- **词表**: 官方 Word2Id 3515 类（全收、无截断）
- **收据**: `artifacts/metrics/blank-gov/<tag>-official-tfnet.json`

---

## Research question

<本次实验要回答的唯一问题。必须单一名问题，不能是"看看效果"。>

---

## Verdict（判答）

<判答在前。只能是三者之一：判活 / 判死 / inconclusive。>
<理由不能基于 best WER 数字本身，必须引用下面的 mean±std 与信噪比。>

---

## 结果

| 指标 | 值 |
|---|---|
| best dev WER（官方口径） | xx.xx % @ ep N |
| 对照组 / 前次配置 | xx.xx % |
| 官方 SOTA（TFNet, dev） | 45.1 % |
| 距 SOTA | ±xx.x pp |
| 训练耗时 | xxx min |
| 参数量 | xx.xx M |
| NaN / 兜底步数 | x / x |

**与前次的配对差（必填，不得只报 best）：**

| 统计量 | 值 |
|---|---|
| mean | ±x.xxx pp |
| std | x.xxx pp |
| 胜率 | n/m |
| **信噪比**（|mean| ÷ 噪声基准） | x.xx |

> 噪声基准当前取值 **1.556 pp**（P101b-ctrl 实测）。累积更多重复组后须回来更新此值。
> 判读：信噪比 <1 ⇒ 不可判；1~2 ⇒ 强提示非已证实；>2 ⇒ 可判。

---

## 泛化诊断（必填）

| epoch | train CTC | dev WER |
|---|---|---|
| 1 | | |
| 中段 | | |
| 末段 | | |

- train loss 降幅 / 是否单调
- dev 是否已进平台（给出平台区间宽度）
- ⇒ 判断是**欠拟合**还是**泛化差距**

---

## 配置（完整，便于复现）

```python
<从收据 config 字段原样贴出>
```

## Caveats / 限制

1. **解码方式**：本实验为 greedy；官方 42.1 % 用 beam width 10 ⇒ 与官方严格不可比。
2. <本次特有的局限>

## Next decision

<下一步。判据必须在下一次实验**启动前**写定——禁止事后挑选阈值。>
```

---

## 硬性约束

1. **禁止只报 best WER**（P83 / P101 教训）。必须给 mean±std + 胜率 + 信噪比。
2. **判据先于实验**（P84 教训）。本文件的 "Next decision" 就是下一次判定「活 / 死」的依据。
3. **gold 口径**：一切 WER 用官方 token 级（corpus-level），详见 `docs/planning/OFFICIAL_EVAL_PROTOCOL.md`。
4. **不得自动推送**：报告写完后不 commit/push，等用户明确说「推送」。
