# -*- coding: utf-8 -*-
"""P52 · 路线 1 决定性验证：旧 Solutions API 能否复现训练特征

## 目的
P51 已证明两种特征**判别力几乎相同**（macroAUC 0.7791 vs 0.7744），
所以 P48b 那 29 倍差距是**坐标系错位**，不是特征质量差异。
⇒ 正确修法是让线上产出与训练一致的特征（路线 1），而不是全量重提（路线 2）。

路线 1 的成败点只有一个：
**`mp.solutions.holistic` 能否复现出与 `artifacts/part3_features/*.npy`
一致的 368 维特征？**

若一致 → 零重训解决 skew，48 轮实验的判优价值全部保住
若不一致 → 旧 landmark 质量无法复现，必须转路线 2（5.3 小时重提）

## 本脚本在 legacy-mp 环境（python3.12 + mediapipe 0.10.21）里跑
必须独立于主 venv（3.14 + 0.10.35 无 solutions 命名空间）。

## 对齐要点（照抄训练 extractor 的常量，见 src/cslr/features/extractor.py）
| 区间 | 宽度 | 内容 |
|---|---|---|
| [0:126] | 126 | 双手 21 点 × 3（左手先右手后） |
| [126:158] | 32 | pose 8 点 × 4（xyz + visibility） |
| [158:182] | 24 | face 8 点 × 3 |
| [182:186] | 4 | presence |
| [186:368] | 182 | 前 182 维的一阶差分（在原始帧率上） |

归一化：origin = 双肩中点，scale = |左肩 - 右肩|（肩距归一化）
POSE_INDICES = (11,12,13,14,15,16,23,24)；FACE_INDICES = (10,33,61,133,152,263,291,362)
重采样：np.linspace(0, T-1, 48).round()
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
FACE_INDICES = (10, 33, 61, 133, 152, 263, 291, 362)
N_FRAMES = 48
BASE_SIZE = 182


def main() -> None:
    import cv2
    import mediapipe as mp

    REPO = Path("/home/su127/FYP/domain-bounded-cslr")
    FEAT = REPO / "artifacts/part3_features"
    VID = Path("/mnt/c/Users/su127/Desktop/csl视频")
    OUT = Path("/tmp/p52")

    print("python    :", sys.version.split()[0])
    print("mediapipe :", mp.__version__)
    from mediapipe.python.solutions import holistic as mp_holistic
    print("solutions : 可用 ✅", flush=True)

    def find_video(sid):
        for split in ("train", "dev"):
            root = VID / split
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                p = d / (sid + ".mp4")
                if p.exists():
                    return p
        return None

    def extract_legacy(video_path):
        """用旧 Solutions API 产出 (48, 368)，布局与训练 extractor 一致。"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("cannot open %s" % video_path)
        hol = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,          # 与训练侧默认值对齐
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        bases, masks = [], []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hol.process(rgb)
                pose = np.zeros((len(POSE_INDICES), 4), np.float32)
                # 🔴 旧 Solutions API 返回 protobuf 对象 `NormalizedLandmarkList`：
                #    既不支持下标（not subscriptable）也不支持迭代（not iterable）。
                #    真正的 landmark 列表在 `.landmark` 字段里。
                #    （实测 pose=33 / hand=21 / face=468 个点）
                p_raw = getattr(res, "pose_landmarks", None)
                p = list(p_raw.landmark) if (p_raw is not None
                                            and len(p_raw.landmark)) else None
                if p:
                    for k, i in enumerate(POSE_INDICES):
                        pose[k] = (p[i].x, p[i].y, p[i].z,
                                   getattr(p[i], "visibility", 0.0) or 0.0)
                hands = np.zeros((2, 21, 3), np.float32)
                hl = hr = False
                lh = getattr(res, "left_hand_landmarks", None)
                rh = getattr(res, "right_hand_landmarks", None)
                if lh is not None and len(lh.landmark):
                    hands[0] = np.array([[q.x, q.y, q.z] for q in lh.landmark],
                                        np.float32)
                    hl = True
                if rh is not None and len(rh.landmark):
                    hands[1] = np.array([[q.x, q.y, q.z] for q in rh.landmark],
                                        np.float32)
                    hr = True
                face = np.zeros((len(FACE_INDICES), 3), np.float32)
                hf = False
                fl = getattr(res, "face_landmarks", None)
                if fl is not None and len(fl.landmark) > max(FACE_INDICES):
                    for k, i in enumerate(FACE_INDICES):
                        q = fl.landmark[i]
                        face[k] = (q.x, q.y, q.z)
                    hf = True
                org = np.array([0.5, 0.5, 0.0], np.float32)
                sc = 1.0
                if p is not None and len(p) > 12:
                    ls = np.array([p[11].x, p[11].y, p[11].z], np.float32)
                    rs = np.array([p[12].x, p[12].y, p[12].z], np.float32)
                    org = (ls + rs) / 2.0
                    d = float(np.linalg.norm(ls - rs))
                    if d > 1e-6:
                        sc = d
                pf = np.concatenate([(pose[:, :3] - org) / sc, pose[:, 3:4]], 1)
                b = np.concatenate([
                    ((hands - org) / sc).reshape(-1),
                    pf.reshape(-1),
                    ((face - org) / sc).reshape(-1)]).astype(np.float32)
                assert b.shape == (BASE_SIZE,), b.shape
                m = np.array([1.0 if p is not None else 0.0,
                              1.0 if hl else 0.0,
                              1.0 if hr else 0.0,
                              1.0 if hf else 0.0], np.float32)
                bases.append(b)
                masks.append(m)
        finally:
            cap.release()
            try:
                hol.close()
            except Exception:                                    # noqa: BLE001
                pass
        if not bases:
            raise RuntimeError("no frames")
        base = np.stack(bases)
        mask = np.stack(masks)
        diff = np.zeros_like(base)
        if base.shape[0] > 1:
            diff[1:] = base[1:] - base[:-1]
        full = np.concatenate([base, mask, diff], 1)
        T = full.shape[0]
        idx = np.linspace(0, T - 1, N_FRAMES).round().astype(int)
        return full[idx].astype(np.float32)

    # ---- 逐个 train 视频对比 ----
    names = sorted(x.name.replace(".landmark.npy", "")
                   for x in (FEAT / "train").glob("*.landmark.npy"))[:6]
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for sid in names:
        p = FEAT / "train" / (sid + ".landmark.npy")
        vp = find_video(sid)
        if vp is None:
            print("  %s 视频缺失，跳过" % sid, flush=True)
            continue
        ref = np.load(p).astype(np.float32)
        t0 = time.time()
        try:
            new = extract_legacy(vp)
        except Exception as exc:                                 # noqa: BLE001
            print("  %s 提取失败: %s" % (sid, str(exc)[:80]), flush=True)
            continue
        el = time.time() - t0
        d = np.abs(ref - new)
        cos = float(np.mean(np.sum(ref * new, 1) /
                            (np.linalg.norm(ref, axis=1) *
                             np.linalg.norm(new, axis=1) + 1e-9)))
        rec = {"sid": sid, "seconds": round(el, 1),
               "identical": bool(np.array_equal(ref, new)),
               "max_abs_diff": round(float(d.max()), 5),
               "mean_abs_diff": round(float(d.mean()), 5),
               "cosine_mean": round(cos, 5),
               "ref_std": round(float(ref.std()), 5),
               "new_std": round(float(new.std()), 5)}
        recs.append(rec)
        print("  %s  %.1fs  逐位一致=%-5s  max|d|=%.4f  cos=%.4f  "
              "std %.4f vs %.4f"
              % (sid, el, rec["identical"], rec["max_abs_diff"],
                 rec["cosine_mean"], rec["ref_std"], rec["new_std"]), flush=True)

    if not recs:
        print("\n❌ 无成功样本")
        return
    n_id = sum(1 for r in recs if r["identical"])
    cos = float(np.mean([r["cosine_mean"] for r in recs]))
    mx = float(max(r["max_abs_diff"] for r in recs))
    print("\n" + "=" * 70)
    print("路线 1 可行性判定")
    print("=" * 70)
    print("  逐位一致 %d/%d，平均 cos=%.4f，最大 |diff|=%.4f"
          % (n_id, len(recs), cos, mx))
    if n_id == len(recs):
        print("  ✅ **完全一致** -> 路线 1 可行，零重训解决 skew")
    elif cos > 0.999:
        print("  ✅ **数值一致（浮点级差异）** -> 路线 1 可行")
    elif cos > 0.95:
        print("  ⚠️ 高度相关但非一致（cos=%.4f）" % cos)
        print("     需进一步判断：是参数不同还是实现细节差异")
    else:
        print("  ❌ **无法复现**（cos=%.4f）-> 路线 1 不可行，转路线 2" % cos)

    p = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/"
             "blank-gov/p52-legacy-repro.json")
    p.write_text(json.dumps({"per_sample": recs,
                             "n_identical": n_id, "n_total": len(recs),
                             "cosine_mean": cos, "max_abs_diff": mx},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
