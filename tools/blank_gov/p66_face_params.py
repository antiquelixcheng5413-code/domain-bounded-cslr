"""P66：FaceLandmarker 检出率调参 —— 把 face 从 0.720 拉回接近旧版 0.998

实测发现（p65b）：新版 face 检出 0.720，旧版 holistic 0.998。
face 段 24 维在未检出时全零 = 28% 的帧丢失 6.5% 的维度。

本脚本系统测试各参数组合，找出能把检出率拉起来的配置。
测的是**检出率**，不测精度 —— 先确认参数有效，再重提。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def _find_repo() -> Path:
    for c in Path(__file__).resolve().parents:
        if (c / "src" / "cslr" / "recognition"
                / "gloss_sequence.py").exists():
            return c
    raise RuntimeError("repo root not found")


REPO = _find_repo()
sys.path.insert(0, str(REPO / "app" / "backend"))

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402
import mediapipe as mp                                      # noqa: E402
from realtime_landmark import REPO as MP_REPO, FACE_INDICES  # noqa: E402

MODELS = MP_REPO / "models"
VIDEOS = sorted(Path("/mnt/c/Users/su127/Desktop/csl视频/dev/A").glob("*.mp4"))[:4]
N_FRAMES = 40

print("=" * 72)
print("FaceLandmarker 检出率调参")
print("=" * 72)
print("  目标：把 face 检出率从 0.720 拉到接近旧版 0.998")
print("  旧 holistic 基准：0.998（基本恒检出）\n")

# 先解码所有帧
frames = []
for p in VIDEOS:
    cap = cv2.VideoCapture(str(p))
    fs = []
    while len(fs) < N_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f)
    cap.release()
    frames.extend(fs)
print("  样本：%d 个视频，共 %d 帧" % (len(VIDEOS), len(frames)))

v = mp.tasks.vision
B = lambda q: mp.tasks.BaseOptions(model_asset_path=str(MODELS / q))

CONFIGS = [
    # (标签, num_faces, min_detection_confidence, min_tracking_confidence)
    ("当前（默认）", 1, None, None),
    ("置信度 0.3", 1, 0.3, 0.3),
    ("置信度 0.1", 1, 0.1, 0.1),
    ("num_faces=2", 2, None, None),
    ("num=2 + 置信0.2", 2, 0.2, 0.2),
    ("num=3 + 置信0.1", 3, 0.1, 0.1),
    ("IMAGE模式+置信0.1", 1, 0.1, 0.1),   # 占位，运行时区分
]

results = []
for label, nf, mdc, mtc in CONFIGS:
    kw = {"base_options": B("face_landmarker.task"),
          "running_mode": v.RunningMode.VIDEO, "num_faces": nf}
    if mdc is not None:
        kw["min_face_detection_confidence"] = mdc
        kw["min_tracking_confidence"] = mtc
    try:
        lm = v.FaceLandmarker.create_from_options(
            v.FaceLandmarkerOptions(**kw))
    except Exception as exc:                                  # noqa: BLE001
        print("  %-20s 初始化失败 %s" % (label, str(exc)[:50]))
        continue

    t0 = time.time()
    hits = 0
    ts = 0
    for f in frames:
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        r = lm.detect_for_video(img, ts)
        ts += 33
        fl = getattr(r, "face_landmarks", None) if r else None
        if fl and len(fl) > max(FACE_INDICES):
            hits += 1
    el = time.time() - t0
    rate = hits / len(frames)
    lm.close()
    results.append({"config": label, "num_faces": nf,
                    "min_det_conf": mdc, "detect_rate": round(rate, 4),
                    "ms_per_frame": round(el / len(frames) * 1000, 1)})
    print("  %-20s num_faces=%d conf=%-5s 检出 %.3f  %.1f ms/帧"
          % (label, nf, mdc, rate, el / len(frames) * 1000))

print("\n" + "=" * 72)
print("对照旧基准与当前")
print("=" * 72)
print("  旧 holistic 基准      0.998")
best = max(results, key=lambda r: r["detect_rate"])
print("  最佳配置              %.3f（%s）"
      % (best["detect_rate"], best["config"]))
print("  当前配置              %.3f"
      % results[0]["detect_rate"])
gain = best["detect_rate"] - results[0]["detect_rate"]
print("  可提升                %+.3f" % gain)

print("\n" + "=" * 72)
print("还要测：输入是否被降采样影响（MediaPipe 内部固定 192x192）")
print("=" * 72)
print("""  说明：MediaPipe FaceLandmarker 内部固定 192x192，
  所以对1080p 直接跑不会更慢，但**脸在画面里占比小**时可能检不出。
  CE-CSL 是复杂背景 + 真实手语，人脸占比可能很小。""")

out = {"samples": {"n_videos": len(VIDEOS), "n_frames": len(frames)},
       "old_holistic_baseline": 0.998,
       "current": results[0] if results else None,
       "best": best,
       "gain": round(gain, 4),
       "all": results}
p = REPO / "artifacts/metrics/blank-gov/p66-face-params.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % p)