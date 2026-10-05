"""定位 deltas 段差异的根因：原始帧数不同 vs 差分公式不同

假设：deltas 段 std 比 2.6，说明差分的**时间步长**不同。
- 若训练特征的原始帧数 ≈ 我的帧数，差分应一致
- 若训练用了不同的帧率（如隔帧采样），差分幅度会按比例放大

验证方法：训练特征无法反推原始帧数，但可以用一个不变量 ——
**训练特征 base 段的帧数固定 48，无法反推。**
换个思路：检查差分的自相关性。
  若 delta[t] = base[t] - base[t-1] 在原始帧率上求，
  则 delta 序列相邻项高度相关（因为 base 平滑）。
  若在重采样后求，delta 会更稀疏。

更直接的办法：跑两种变体，看哪个 cos 高。
  变体 A：原始帧率算差分 -> 重采样（当前实现）
  变体 B：重采样到 48 -> 算差分
"""
import sys
from pathlib import Path

import cv2
import numpy as np

POSE_INDICES = (11, 12, 13, 14, 15, 16, 23, 24)
FACE_INDICES = (10, 33, 61, 133, 152, 263, 291, 362)
N_FRAMES = 48


def extract(video_path, delta_mode):
    from mediapipe.python.solutions import holistic as mp_holistic
    cap = cv2.VideoCapture(str(video_path))
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
    T_native = base.shape[0]

    def build(bt, mt):
        d = np.zeros_like(bt)
        if bt.shape[0] > 1:
            d[1:] = bt[1:] - bt[:-1]
        full = np.concatenate([bt, mt, d], 1)
        idx = np.linspace(0, full.shape[0] - 1, N_FRAMES).round().astype(int)
        return full[idx].astype(np.float32)

    if delta_mode == "native_then_resample":      # 变体 A
        return build(base, mask), T_native
    # 变体 B：先重采样 base/mask，再算差分
    idx = np.linspace(0, T_native - 1, N_FRAMES).round().astype(int)
    return build(base[idx], mask[idx]), T_native


def main():
    FEAT = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/part3_features")
    VID = Path("/mnt/c/Users/su127/Desktop/csl视频")

    def fv(sid):
        for split in ("train", "dev"):
            r = VID / split
            if r.exists():
                for d in sorted(r.iterdir()):
                    p = d / (sid + ".mp4")
                    if p.exists():
                        return p
        return None

    for sid in ("train-00001", "train-00002"):
        ref = np.load(FEAT / "train" / (sid + ".landmark.npy")).astype(np.float32)
        vp = fv(sid)
        print("\n%s" % sid)
        for mode in ("native_then_resample", "resample_then_delta"):
            new, T = extract(vp, mode)
            c_all = float(np.dot(ref.reshape(-1), new.reshape(-1)) /
                          (np.linalg.norm(ref) * np.linalg.norm(new) + 1e-9))
            a = ref[:, 186:368].reshape(-1)
            b = new[:, 186:368].reshape(-1)
            c_d = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            a2 = ref[:, :182].reshape(-1)
            b2 = new[:, :182].reshape(-1)
            c_b = float(np.dot(a2, b2) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-9))
            print("  %-22s 原始帧=%3d  base cos=%.4f  deltas cos=%.4f  整段 cos=%.4f"
                  % (mode, T, c_b, c_d, c_all))
        # 训练 deltas 的 std 作为参考
        print("  训练特征 deltas std = %.4f" % ref[:, 186:368].std())


if __name__ == "__main__":
    main()
