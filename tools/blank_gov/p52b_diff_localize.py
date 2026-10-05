# -*- coding: utf-8 -*-
"""P52b · 旧 API 复现差异定位：差异集中在哪个块/哪些维度？

## P52 结果
cos 0.9685 但非逐位一致，max|d|=2.39。
「高度相关但不精确」有两种成因，**修法完全不同**：
  A. **参数不同**（如 model_complexity / refine_face_landmarks / 置信度阈值）
     -> 改参数即可对齐，路线 1 仍可行
  B. **实现细节不同**（如手别约定、坐标后处理、差分基准）
     -> 需要对齐代码，路线 1 成本上升

## 本脚本按块拆解差异
hands / pose / face / presence / deltas 各自的
  cosine、平均绝对差、最大绝对差、std 比
差异最大的块 = 优先排查对象。

同时对差异最大的样本逐维看是「少数维爆炸」还是「整体偏移」。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
FACE_INDICES = (10, 33, 61, 133, 152, 263, 291, 362)
N_FRAMES, BASE_SIZE = 48, 182
BLOCKS = {"hands[0:126]": (0, 126), "pose[126:158]": (126, 158),
          "face[158:182]": (158, 182), "presence[182:186]": (182, 186),
          "deltas[186:368]": (186, 368)}


def extract_legacy(video_path):
    from mediapipe.python.solutions import holistic as mp_holistic
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("cannot open")
    hol = mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                               refine_face_landmarks=False,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
    bases, masks = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose = np.zeros((len(POSE_INDICES), 4), np.float32)
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
                hands[0] = np.array([[q.x, q.y, q.z] for q in lh.landmark], np.float32)
                hl = True
            if rh is not None and len(rh.landmark):
                hands[1] = np.array([[q.x, q.y, q.z] for q in rh.landmark], np.float32)
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
            b = np.concatenate([((hands - org) / sc).reshape(-1),
                                pf.reshape(-1),
                                ((face - org) / sc).reshape(-1)]).astype(np.float32)
            bases.append(b)
            masks.append(np.array([1.0 if p is not None else 0.0,
                                   1.0 if hl else 0.0,
                                   1.0 if hr else 0.0,
                                   1.0 if hf else 0.0], np.float32))
    finally:
        cap.release()
        try:
            hol.close()
        except Exception:                                          # noqa: BLE001
            pass
    base, mask = np.stack(bases), np.stack(masks)
    diff = np.zeros_like(base)
    if base.shape[0] > 1:
        diff[1:] = base[1:] - base[:-1]
    full = np.concatenate([base, mask, diff], 1)
    idx = np.linspace(0, full.shape[0] - 1, N_FRAMES).round().astype(int)
    return full[idx].astype(np.float32)


def main() -> None:
    REPO = Path("/home/su127/FYP/domain-bounded-cslr")
    FEAT = REPO / "artifacts/part3_features"
    VID = Path("/mnt/c/Users/su127/Desktop/csl视频")

    def find_video(sid):
        for split in ("train", "dev"):
            root = VID / split
            if root.exists():
                for d in sorted(root.iterdir()):
                    p = d / (sid + ".mp4")
                    if p.exists():
                        return p
        return None

    sids = ["train-00006", "train-00001", "train-00002"]
    out = {"per_sample": {}}
    for sid in sids:
        p = FEAT / "train" / (sid + ".landmark.npy")
        vp = find_video(sid)
        if not p.exists() or vp is None:
            continue
        ref = np.load(p).astype(np.float32)
        new = extract_legacy(vp)
        print("\n" + "=" * 72)
        print("%s" % sid)
        print("=" * 72)
        print("%-18s %9s %11s %11s %9s" %
              ("block", "cos", "mean|d|", "max|d|", "std比"))
        rec = {}
        for nm, (s0, s1) in BLOCKS.items():
            # 特征是 (48, 368) 二维 —— 按帧切维度，不能用 3 维索引
            A, B = ref[:, s0:s1], new[:, s0:s1]
            a, b = A.reshape(-1), B.reshape(-1)
            c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            d = np.abs(A - B)
            sr = float(B.std() / (A.std() + 1e-9))
            rec[nm] = {"cos": round(c, 4), "mean_abs": round(float(d.mean()), 5),
                       "max_abs": round(float(d.max()), 4),
                       "std_ratio": round(sr, 3)}
            print("%-18s %9.4f %11.5f %11.4f %9.3f"
                  % (nm, c, d.mean(), d.max(), sr))

        # 差异最大的 5 个维度，看是否集中在少数维
        d = np.abs(ref - new)
        flat = d.mean(0)                            # (368,) 每维跨帧平均差
        top = np.argsort(-flat)[:5]
        print("\n  差异最大的 5 个维度（跨帧平均 |d|）：")
        for k in top:
            print("    dim %3d  mean|d|=%.4f  ref mean=%+.4f  new mean=%+.4f"
                  % (k, flat[k], ref[:, k].mean(), new[:, k].mean()))
        rec["top_diff_dims"] = [{"dim": int(k), "mean_abs": round(float(flat[k]), 4),
                                 "ref_mean": round(float(ref[:, k].mean()), 4),
                                 "new_mean": round(float(new[:, k].mean()), 4)}
                                for k in top]
        out["per_sample"][sid] = rec

    # 综合判断
    print("\n" + "=" * 72)
    print("综合判断")
    print("=" * 72)
    agg = {}
    for nm in BLOCKS:
        cs = [out["per_sample"][s][nm]["cos"] for s in out["per_sample"]
              if nm in out["per_sample"][s]]
        ms = [out["per_sample"][s][nm]["mean_abs"] for s in out["per_sample"]
              if nm in out["per_sample"][s]]
        agg[nm] = {"cos_mean": round(float(np.mean(cs)), 4),
                   "mean_abs_mean": round(float(np.mean(ms)), 5)}
        print("  %-18s 平均 cos=%.4f  平均 |d|=%.5f"
              % (nm, agg[nm]["cos_mean"], agg[nm]["mean_abs_mean"]))
    out["aggregate"] = agg
    worst = min(agg.items(), key=lambda kv: kv[1]["cos_mean"])
    print("\n  差异最大的块：%s（cos=%.4f）" % (worst[0], worst[1]["cos_mean"]))
    p = REPO / "artifacts/metrics/blank-gov/p52b-diff-localize.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> %s" % p)


if __name__ == "__main__":
    main()
