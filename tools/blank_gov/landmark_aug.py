"""
输入侧 landmark 增强：neck 归一化 + 肩距缩放 + 速度/加速度差分。

来源：Wójcicka et al., LREC-COLING 2026
（参考论文/新参考论文/03_Wojcicka_LREC2026.pdf）Sec 4.1、4.1.2

为什么做：手语起始是快速加速、结束是减速保持，运动学不对称。
只喂静态位置坐标，边界附近没有可分信号；加上 Δp、Δ²p 后
位置→速度→加速度三层动力学都进网络。

差分公式（论文脚注 2）：
    Δx_t  = x_t - x_{t-1}
    Δ²x_t = Δx_t - Δx_{t-1}

输出维度：368 -> 1104（3 × 368）。

本项目已有的 48×368 landmark 特征若已是归一化后的坐标，
用 augment() 即可；若坐标未归一化，先用 normalize_neck_shoulder()。
"""

from __future__ import annotations

import numpy as np

# MediaPipe landmark 分块（48 = 下标区间约定，按实际特征布局调整）
# 常见布局：pose 上半身 / 双手 / 脸，具体区间需用 inspect_feature_layout 核对
DEFAULT_BLOCKS = {
    "pose": (0, 33),
    "face": (33, 45),
    "hands": (45, 48),
}


def inspect_feature_layout(dim: int = 368, verbose: bool = True) -> dict:
    """368 维 landmark 的分块检查。

    论文 Wójcicka 明确：pose 丢弃下半身（髋/膝/踝/足），
    坐姿录制下半身无语言信息且常出画。face 不用 468 稠密网格，
    只取脸廓/眉/眼/唇的语义子集。

    EMNLP 2023 的负面结果：128 个纯面部轮廓点反而变差
    （sign IoU 0.66 -> 0.58），原文归因"太稠密，可能混淆模型"。
    Wójcicka 报告面部 +13.6pp。两篇差异在于是否语义筛选。
    """
    layout = dict(DEFAULT_BLOCKS)
    total = sum(b - a for a, b in layout.values())
    if verbose:
        print(f"dim={dim}, blocks={layout}, covered={total}")
        if total != dim:
            print(
                f"警告：分块覆盖 {total} != {dim}。"
                "请按实际特征布局调整 DEFAULT_BLOCKS，"
                "或用 select_landmark_subset 显式筛选。"
            )
    return layout


def normalize_neck_shoulder(pose: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """neck 原点 + 肩距缩放（论文 Sec 4.1.1）。

    pose: (T, P, C)  C 通常为 3 (x,y,z) 或 2 (x,y)
    返回同形状数组，坐标系的原点落在双肩中点，尺度为肩距。

    MediaPipe pose 的肩点下标：11 = 左肩, 12 = 右肩。
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 3:
        raise ValueError(f"expected (T,P,C), got {pose.shape}")
    l_sh = pose[:, 11, :2]
    r_sh = pose[:, 12, :2]
    neck = (l_sh + r_sh) / 2.0
    shoulder_width = np.linalg.norm(l_sh - r_sh, axis=-1, keepdims=True)  # (T,1)
    scale = np.maximum(shoulder_width, eps)
    out = (pose[:, :, :2] - neck[:, None, :]) / scale[:, :, None]
    if pose.shape[-1] == 3:
        # z 单独按同一 scale 缩放，但**不**减去 neck 的 z（原点在 z 上不可靠）
        out = np.concatenate([out, pose[:, :, 2:3] / scale[:, :, None]], axis=-1)
    return out


def select_landmark_subset(
    pose: np.ndarray,
    drop_lower_body: bool = True,
    face_points: np.ndarray | None = None,
) -> np.ndarray:
    """按论文做 landmark 语义筛选。

    drop_lower_body: 丢弃髋(23,24)/膝(25,26)/踝(27,28)/足(31,32)
                     只留肩(11,12)/肘(13,14)/腕(15,16) + 鼻(0)/眼(2,5) 作头部朝向参考
    face_points:     脸廓/眉/眼/唇的**语义子集**下标。
                     传 None 表示不接面部（保守起点）。
                     不要直接喂 128 稠密轮廓点——EMNLP 2023 实测变差。
    """
    pose = np.asarray(pose)
    keep = [0, 2, 5, 11, 12, 13, 14, 15, 16]
    if not drop_lower_body:
        keep = list(range(pose.shape[1]))
    elif face_points is not None:
        keep = keep + list(face_points)
    return pose[:, keep, :]


def temporal_derivative(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """一阶 + 二阶差分（论文 Sec 4.1.2 脚注 2）。

    feat: (T, D) -> (T, 3D)

    边界约定：Δx_t = x_t - x_{t-1}，故 Δx_0 无定义取 0；
    Δ²x_t = Δx_t - Δx_{t-1}，故 Δ²x_0 与 Δ²x_1 均取 0。
    """
    feat = np.asarray(feat, dtype=np.float64)
    t = feat.shape[0]
    d1 = np.zeros_like(feat)
    d2 = np.zeros_like(feat)
    if t >= 2:
        d1[1:] = feat[1:] - feat[:-1]
    if t >= 3:
        d2[2:] = d1[2:] - d1[1:-1]
    return np.concatenate([feat, d1, d2], axis=-1)


def augment(feat: np.ndarray, use_derivative: bool = True) -> np.ndarray:
    """主入口：landmark 特征 -> 增强后的特征。

    feat: (T, 368) 现有特征
    返回 (T, 1104) 若 use_derivative，否则 (T, 368)

    归一化请在调用前完成（若特征已归一化则跳过）。
    """
    feat = np.asarray(feat, dtype=np.float32)
    if not use_derivative:
        return feat
    return temporal_derivative(feat).astype(np.float32)


def derivative_stats(feat: np.ndarray) -> dict:
    """诊断用：看速度/加速度的量级是否合理。

    若 ||Δp|| 与 ||p|| 同量级，说明差分被噪声主导，
    应先做时间平滑或降采样再差分。
    """
    feat = np.asarray(feat, dtype=np.float64)
    d1 = np.linalg.norm(np.diff(feat, axis=0), axis=-1)
    d2 = np.linalg.norm(np.diff(np.diff(feat, axis=0), axis=0), axis=-1)
    p = np.linalg.norm(feat, axis=-1)
    return {
        "pos_mean": float(p.mean()),
        "d1_mean": float(d1.mean()) if d1.size else 0.0,
        "d2_mean": float(d2.mean()) if d2.size else 0.0,
        "ratio_d1_pos": float(d1.mean() / max(p.mean(), 1e-8)) if d1.size else 0.0,
        "ratio_d2_pos": float(d2.mean() / max(p.mean(), 1e-8)) if d2.size else 0.0,
    }
