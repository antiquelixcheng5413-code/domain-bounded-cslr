"""P66b：为什么 face 检出率是0（全部配置）/ 脸到底占画面多少

P66 全部配置检出 0.000，但已提取的特征里 face=0.720 —— 矛盾！
必须查清：是不是 dev 视频与 train 视频差异？还是别的？
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "app" / "backend"))

import mediapipe as mp                                      # noqa: E402
from realtime_landmark import REPO as MP_REPO, FACE_INDICES  # noqa: E402

MODELS = MP_REPO / "models"
v = mp.tasks.vision

print("=" * 72)
print("1. 已提取特征里 face 的真实检出率（train vs dev）")
print("=" * 72)
for tag, root in (("train", "artifacts/part3_features_tasksapi/train"),
                  ("old train", "artifacts/part3_features/train"),
                  ("old dev", "artifacts/part3_features/validation")):
    fs = sorted(glob.glob(str(REPO / root / "*.landmark.npy")))[:150]
    if not fs:
        print("  %-10s 无文件" % tag)
        continue
    F = np.concatenate([np.load(f)[:, 182:186] for f in fs], 0)
    print("  %-10s n=%3d  pose=%.3f handL=%.3f handR=%.3f face=%.3f"
          % (tag, len(fs), F[:, 2].mean(), F[:, 0].mean(),
             F[:, 1].mean(), F[:, 3].mean()))

print("\n" + "=" * 72)
print("2. 直接测：train 视频 vs dev 视频的 face 检出")
print("=" * 72)
print("  （P66 我用的是 dev 视频，检出全 0；但特征里 face=0.720，矛盾）")

sets = [
    ("dev  (csl视频/dev/A)", sorted(Path(
        "/mnt/c/Users/su127/Desktop/csl视频/dev/A").glob("*.mp4"))[:3]),
    ("train(csl视频/train/A)", sorted(Path(
        "/mnt/c/Users/su127/Desktop/csl视频/train/A").glob("*.mp4"))[:3]),
]
B = lambda q: mp.tasks.BaseOptions(model_asset_path=str(MODELS / q))

for tag, vids in sets:
    if not vids:
        print("  %-26s 目录不存在" % tag)
        continue
    frames = []
    for p in vids:
        cap = cv2.VideoCapture(str(p))
        fs = []
        while len(fs) < 30:
            ok, f = cap.read()
            if not ok:
                break
            fs.append(f)
        cap.release()
        frames.extend(fs)
    lm = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
        base_options=B("face_landmarker.task"),
        running_mode=v.RunningMode.VIDEO, num_faces=1))
    pose = v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(
        base_options=B("pose_landmarker_lite.task"),
        running_mode=v.RunningMode.VIDEO, num_poses=1))
    hits = phits = 0
    ts = 0
    for f in frames:
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        r = lm.detect_for_video(img, ts)
        pr = pose.detect_for_video(img, ts)
        ts += 33
        fl = getattr(r, "face_landmarks", None) if r else None
        if fl and len(fl) > max(FACE_INDICES):
            hits += 1
        if pr.pose_landmarks:
            phits += 1
    print("  %-26s %d 帧  face检出 %.3f  pose检出 %.3f  分辨率 %dx%d"
          % (tag, len(frames), hits / len(frames), phits / len(frames),
             frames[0].shape[1], frames[0].shape[0]))
    lm.close()
    pose.close()

print("\n" + "=" * 72)
print("3. 🔴 关键：已提取特征里 face 到底是哪一段？")
print("=" * 72)
fs = sorted(glob.glob(str(REPO / "artifacts/part3_features_tasksapi/train"
                        / "*.landmark.npy")))[:5]
if fs:
    a = np.load(fs[0])
    print("  presence 位[182:186] 前 8 帧：")
    print("   ", np.round(a[:8, 182:186], 2))
    nz = (np.abs(a[:, 158:182]).sum(1) > 1e-6)
    print("  face 段[158:182] 非零帧占比 = %.3f" % nz.mean())
    print("  presence[3] 均值 = %.3f" % a[:, 185].mean())
    print("  ⇒ 若两者不一致，说明 presence 记录的不是 face 段是否为零")

out = {}
p = REPO / "artifacts/metrics/blank-gov/p66b-face-debug.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")