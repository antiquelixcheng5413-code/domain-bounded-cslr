"""
免训练时序特征差分（TFD）分词器。

来源：TS²-TFD, Neurocomputing 2027（见 参考论文/新参考论文/
Self-supervised-temporal-sign-segmentation-using-the-temporal-_2027_Neurocom.pdf）
  - 式(1)(2) TFD 距离定义
  - Algorithm 1 无参数边界检测

为什么用它：这是本项目唯一"零标注、零训练、O(N)、CPU 可跑"的帧级边界来源。
下游 BIO 标签生成、CSFormer spotting 监督全部依赖它产出的伪边界。
论文 Table 2：免训练版本 AmF1B 26.00，已超过全部 5 个无监督基线。

注意（论文 Table 1 证据）：余弦距离比 L2 只高 +0.41 AmF1B，差异在噪声量级。
本实现默认 L2（metric="l2"），需要复现论文表格时切到 "cosine"。

边界不可跨数据集迁移（论文 Sec 4.4.9）：把 NIASL2021 的 (T,M)=(22,17)
直接用到 PHOENIX2014，AmF1B 崩到 3.88。必须用 suggest_T_from_gloss_stats
在本数据集上重新估计。
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _reduce(feat: np.ndarray, t: int, m: int, metric: str) -> np.ndarray:
    """计算式(1)(2)：当前帧特征与过去/未来局部时序均值之间的距离。

    feat: (N, C) 帧级特征序列
    t:   过去/未来窗口大小
    m:   局部极大值半径（Algorithm 1 的 M）
    """
    n = feat.shape[0]
    # Algorithm 1 明确从 t+1 迭代到 n-t，两端不参与
    if n <= 2 * t:
        return np.zeros(n, dtype=np.float64)

    mu_past = np.zeros_like(feat)
    mu_future = np.zeros_like(feat)
    # 累积和实现 O(N) 滑窗均值，避免 O(N*t) 的循环
    cs = np.cumsum(np.vstack([np.zeros((1, feat.shape[1])), feat]), axis=0)
    idx = np.arange(n)
    past_lo = np.clip(idx - t, 0, n)
    past_hi = np.clip(idx, 0, n)
    fut_lo = np.clip(idx + 1, 0, n)
    fut_hi = np.clip(idx + 1 + t, 0, n)
    mu_past = (cs[past_hi] - cs[past_lo]) / np.maximum(past_hi - past_lo, 1)[:, None]
    mu_future = (cs[fut_hi] - cs[fut_lo]) / np.maximum(fut_hi - fut_lo, 1)[:, None]

    if metric == "cosine":
        num = np.sum(mu_past * mu_future, axis=1)
        den = np.maximum(np.linalg.norm(mu_past, axis=1), _EPS) * np.maximum(
            np.linalg.norm(mu_future, axis=1), _EPS
        )
        d = 1.0 - num / den
    elif metric == "l2":
        d = np.linalg.norm(mu_past - mu_future, axis=1)
    else:
        raise ValueError(f"unknown metric: {metric}")

    tfd = np.zeros(n, dtype=np.float64)
    tfd[t : n - t] = d[t : n - t]
    return tfd


def tfd_signal(feat: np.ndarray, t: int, metric: str = "l2") -> np.ndarray:
    """式(1)(2)：返回逐帧 TFD 信号，长度与输入相同，两端为 0。"""
    return _reduce(np.asarray(feat, dtype=np.float64), t=t, m=0, metric=metric)


def detect_boundaries(
    feat: np.ndarray,
    t: int,
    m: int,
    metric: str = "l2",
    as_peak: bool = True,
) -> np.ndarray:
    """Algorithm 1：TFD 信号的局部极大值即边界。

    as_peak=True  -> 返回峰值帧下标（Algorithm 1 原式）
    as_peak=False -> 返回 b_t = "该帧是否越过一个 TFD 峰值" 的 0/1 边界信号，
                      用于 SignShift 式逐帧二分类 + L_smooth 正则。

    返回下标严格递增，且已去掉距序列两端 t 帧内的候选。
    """
    feat = np.asarray(feat, dtype=np.float64)
    n = feat.shape[0]
    tfd = _reduce(feat, t=t, m=m, metric=metric)
    if n == 0:
        return np.zeros(0, dtype=bool) if not as_peak else np.zeros(0, dtype=np.int64)

    cand = np.zeros(n, dtype=bool)
    # Algorithm 1 第二个循环：i 从 m+1 到 n-m
    for i in range(max(m + 1, t + 1), max(n - m, t + 1)):
        lo = max(i - m, 0)
        hi = min(i + m + 1, n)
        if tfd[i] == tfd[lo:hi].max():
            cand[i] = True

    if as_peak:
        return np.flatnonzero(cand).astype(np.int64)

    b = np.zeros(n, dtype=bool)
    peaks = np.flatnonzero(cand)
    for p in peaks:
        for j in range(p, min(p + m + 1, n)):
            b[j] = True
    return b


def suggest_T_from_gloss_stats(
    n_frames: int,
    n_gloss: int,
    ratio_lo: float = 0.6,
    ratio_hi: float = 0.9,
) -> int:
    """论文 Sec 4.4.9 的 T 选择规则。

    T 取目标数据集"平均每 gloss 帧长"的 60%–90%。
    论文原话：只用二十条标注视频估计就足够。

    n_frames / n_gloss 即平均每 gloss 帧长。
    """
    if n_gloss <= 0:
        raise ValueError("n_gloss must be positive")
    avg = float(n_frames) / float(n_gloss)
    t = int(round(avg * (ratio_lo + ratio_hi) / 2.0))
    return max(2, min(t, max(2, n_frames // 4)))


def suggest_TM(
    n_frames: int,
    n_gloss: int,
    under_segment_bias: float = 1.5,
) -> tuple[int, int]:
    """给出 (T, M) 起点。

    论文做法：M 略小于 T。对比学习采样时用更大的 M 故意造成欠分割
    （PHOENIX2014 评测 M=4，采样 M=6）。这里给评测用的 M。

    消融依据（Table 7）：随机 negative 比边界 negative 差 0.69；
    过分割采样（M=2）比欠分割差 0.89。宁可欠分割。
    """
    t = suggest_T_from_gloss_stats(n_frames, n_gloss)
    m = max(2, int(round(t / under_segment_bias)))
    return t, min(m, max(2, t - 1))
