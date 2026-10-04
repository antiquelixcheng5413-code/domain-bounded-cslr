"""
BIO 三类帧级标签生成 + 加权交叉熵。

来源（多篇独立验证）：
  1. Wójcicka LREC 2026 Sec 4.2 —— BIO 替代 IO，**B 标签 ±k 时间膨胀**
  2. Wójcicka Table 2 类别权重 λ_Out=0.5, λ_Beg=5.0, λ_In=1.0
  3. EMNLP 2023 脚注 6 —— sign 任务类别比例 B:I:O = 1:5:18
  4. Hands-On FG2025 Sec III-D1 —— 帧级 CE 与 gloss 级 CTC 联合损失
  5. EMNLP 2023 Figure 4 —— 25fps 下 BIO 复现率 99.7%，IO 做不到

为什么这直接对治 97% blank：
  CTC 收敛到"全 blank"平凡解；BIO 版收敛到"全 Out"平凡解。
  两者同构 —— 稀有类给高权重、主导类压低。
  Wójcicka 原文：最高权重给 Begin（Out 的 10 倍）以强制高召回，
  同时降权 Out 以防退化为"每帧都预测 Out"的平凡解。

**不要加 E 类**（Wójcicka Sec 4.2.3 实证退化）：
  原文归因：sign 起始由锐利高速运动标记，结束往往是渐缓减速/保持/回到中性位。
  运动学不对称，E 类无法稳定预测。
"""

from __future__ import annotations

import numpy as np

OUT, IN, BEG = 0, 1, 2
NUM_CLASSES = 3
CLASS_NAMES = {OUT: "O", IN: "I", BEG: "B"}

# Wójcicka Table 2 的实测权重
DEFAULT_WEIGHTS = {OUT: 0.5, IN: 1.0, BEG: 5.0}


def labels_from_boundaries(
    n_frames: int,
    boundaries: np.ndarray,
    dilate_k: int = 1,
) -> np.ndarray:
    """从边界下标生成 BIO 三类标签。

    标签语义（对齐 Wójcicka Sec 4.2.1）：
      O = 段外/背景（含 rest position 与非语言学过渡动作）
      I = 手语词内部
      B = 手语词起始

    dilate_k: 时间膨胀半径。论文明确为缓解人工标注抖动导致的稀疏，
              "t ± k (resulting in a 3-frame active window)"。
              **这是本模块对 blank 稀疏最直接的一招** —— 把 one-hot
              边界尖峰摊成 3 帧宽的窗口，让边界附近帧都有监督。

    边界为词间间隔的**右边界**（即词 i 的最后一帧之后）。
    第一个边界之前标 O；两个边界之间标 I；每个边界所在帧及其
    后 dilate_k 帧标 B。
    """
    y = np.full(n_frames, OUT, dtype=np.int64)
    if n_frames == 0:
        return y

    b = np.asarray(sorted(set(int(x) for x in boundaries if 0 <= int(x) < n_frames)))
    if b.size:
        for start in b:
            # 论文原文：t ± k，"resulting in a 3-frame active window"
            # 即窗口以边界帧为中心，左右各 k 帧，共 2k+1 帧
            lo = max(start - dilate_k, 0)
            hi = min(start + dilate_k + 1, n_frames)
            y[lo:hi] = BEG
    # 每个 B 窗口之后到下一个 B 窗口之前为 I。
    # 注意 I 段的右界要避开下一个 B 窗口的左沿，否则会覆盖掉 B 标签。
    if b.size:
        for i, start in enumerate(b):
            seg_start = min(start + dilate_k + 1, n_frames)
            if i + 1 < b.size:
                seg_end = max(seg_start, b[i + 1] - dilate_k)
            else:
                seg_end = n_frames
            if seg_end > seg_start:
                y[seg_start:seg_end] = IN
    return y


def labels_from_peak_count(
    n_frames: int,
    n_segments: int,
    dilate_k: int = 1,
) -> np.ndarray:
    """无边界标注时的退化路径：按句级 gloss 数均分。

    只在完全没有 TFD 伪边界时用（例如 TFD 输出为空）。
    质量明显低于 TFD 路径，仅作兜底。
    """
    if n_segments <= 0:
        return np.full(n_frames, OUT, dtype=np.int64)
    edges = np.linspace(0, n_frames, n_segments + 1).astype(int)
    y = np.full(n_frames, OUT, dtype=np.int64)
    for i in range(n_segments):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            continue
        y[max(s - dilate_k, 0) : min(s + dilate_k + 1, e)] = BEG
        y[min(s + dilate_k + 1, e) : e] = IN
    return y


def class_weights_from_labels(
    labels_list: list[np.ndarray],
    base_weights: dict[int, float] | None = None,
    extra_beg_boost: float = 1.0,
) -> np.ndarray:
    """按实际类别比例算权重，与论文两处依据交叉校准。

    两处依据：
      - Wójcicka 实测 λ_Out=0.5, λ_Beg=5.0, λ_In=1.0
      - EMNLP 脚注 6 比例 sign B:I:O = 1:5:18
        -> 归一化权重约为 B:1 / I:5/18=0.28 / O:18/18=1.0
        -> 相对 B:O = 1:18，比 Wójcicka 的 1:10 更激进

    策略：以 Wójcicka 实测权重为基底（它是唯一在真实语料上跑出来的），
    再按你的实际比例做温和调整（几何平均，避免小数据下比例噪声主导）。
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for y in labels_list:
        for c in range(NUM_CLASSES):
            counts[c] += float((y == c).sum())
    total = counts.sum()
    if total <= 0:
        return np.array([DEFAULT_WEIGHTS[OUT], DEFAULT_WEIGHTS[IN], DEFAULT_WEIGHTS[BEG]])

    freq = np.maximum(counts, 1.0) / total
    # 频率反比（平移到 [0.5, 5.0] 区间，与 Wójcicka 的量级对齐）
    inv = 1.0 / freq
    inv = inv / inv[IN]  # 以 I 为基准
    inv = np.clip(inv, 0.5, 5.0)
    inv[BEG] *= extra_beg_boost

    if base_weights is None:
        base = np.array([DEFAULT_WEIGHTS[OUT], DEFAULT_WEIGHTS[IN], DEFAULT_WEIGHTS[BEG]])
    else:
        base = np.array([base_weights[OUT], base_weights[IN], base_weights[BEG]])

    # 几何平均：论文实测值 与 数据驱动值
    w = np.sqrt(base * inv)
    return (w / w[IN]).astype(np.float32)


def weighted_frame_ce(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """加权交叉熵（论文 Sec 4.3.3）。

    L = -Σ_t w_{y_t} log p_{y_t}(t)
    """
    p = _softmax(logits)
    n = labels.shape[0]
    if n == 0:
        return 0.0
    w = weights[labels]
    return float(-(w * np.log(p[np.arange(n), labels] + eps)).mean())


def combined_loss(
    frame_logits: np.ndarray,
    frame_labels: np.ndarray,
    weights: np.ndarray,
    ctc_loss: float,
    lambda_ctc: float = 0.1,
) -> float:
    """帧级加权 CE 与 gloss 级 CTC 的联合损失。

    形式来自 Hands-On FG2025 Sec III-D1：帧级 CE + gloss 级 CTC 同时优化。
    lambda_ctc 从 0.1 起调（论文未给具体值，需自己扫）。

    直觉：CTC 提供句子级对齐压力，帧级 CE 提供逐帧密集压力。
    后者正是 CTC 缺的（97% blank 下 CTC 的逐帧梯度几乎全在 blank 类）。
    """
    return weighted_frame_ce(frame_logits, frame_labels, weights) + lambda_ctc * ctc_loss


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def label_distribution(labels: np.ndarray) -> dict:
    """标签分布诊断。

    检查点：若 O 类占比 > 0.9，模型有极大概率退化成"全 O"平凡解，
    此时应提高 B 权重或检查 TFD 伪边界是否可信。
    EMNLP sign 任务参考比例 B:I:O = 1:5:18（O 占 75%）。
    """
    n = labels.shape[0]
    if n == 0:
        return {CLASS_NAMES[c]: 0.0 for c in range(NUM_CLASSES)}
    return {CLASS_NAMES[c]: float((labels == c).sum()) / n for c in range(NUM_CLASSES)}
