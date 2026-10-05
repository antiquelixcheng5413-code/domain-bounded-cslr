"""P67：face 检出 0.722 是「脸太小检不出」还是「参数不对」

判定方法：
  用**提取器完全相同的访问方式**（face_landmarks[0] 嵌套）重测，
  并测不同 num_faces /置信度 / 输出face_blendshapes 的影响。
另外测一个关键假设：**裁剪出上半身区域再送进去**，
  因为 MediaPipe 内部固定 192x192，1080p 里脸只占很小一块 →
  缩到 192 后脸几乎不可辨。这是 P43 之后一直没验证的假设。
"""
from __future__ import annotations

import glob
import json
import sys
import time
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
B = lambda q: mp.tasks.BaseOptions(model_asset_path=str(MODELS / q))
MAXIDX = max(FACE_INDICES)

p = sorted(Path("/mnt/c/Users/su127/Desktop/csl视频/train/A").glob("*.mp4"))[0]
cap = cv2.VideoCapture(str(p))
frames = []
while len(frames) < 50:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print("样本：%s，%d 帧，%dx%d"
      % (p.name, len(frames), frames[0].shape[1], frames[0].shape[0]))
print("FACE_INDICES=%s  max=%d" % (FACE_INDICES, MAXIDX))


def run(label, *, num_faces=1, conf=None, crop=None, upscale=1.0):
    kw = {"base_options": B("face_landmarker.task"),
          "running_mode": v.RunningMode.VIDEO, "num_faces": num_faces}
    if conf is not None:
        kw["min_face_detection_confidence"] = conf
        kw["min_tracking_confidence"] = conf
    try:
        lm = v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(**kw))
    except Exception as exc:                                  # noqa: BLE001
        print("  %-30s init fail %s" % (label, str(exc)[:40]))
        return None
    hits = 0
    ts = 0
    t0 = time.time()
    for f in frames:
        g = f
        if crop:                      # 裁上半身
            h, w = f.shape[:2]
            g = f[int(h * 0.05):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
        if upscale != 1.0:
            g = cv2.resize(g, None, fx=upscale, fy=upscale,
                           interpolation=cv2.INTER_CUBIC)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(g, cv2.COLOR_BGR2RGB))
        r = lm.detect_for_video(img, ts)
        ts += 33
        # ✅ 用提取器相同的访问方式：face_landmarks[0] 嵌套
        fl = r.face_landmarks if r else None
        if fl and len(fl) > 0:
            inner = fl[0]
            if len(inner) > MAXIDX:
                hits += 1
    el = time.time() - t0
    lm.close()
    rate = hits / len(frames)
    print("  %-30s 检出 %.3f  %.1f ms/帧"
          % (label, rate, el / len(frames) * 1000))
    return {"label": label, "rate": round(rate, 4),
            "ms": round(el / len(frames) * 1000, 1)}


print("\n" + "=" * 72)
print("A. 用正确访问方式重测（此前是我的脚本 bug）")
print("=" * 72)
res = []
res.append(run("默认 num_faces=1"))
res.append(run("num_faces=2", num_faces=2))
res.append(run("置信度 0.3", conf=0.3))
res.append(run("置信度 0.1", conf=0.1))

print("\n" + "=" * 72)
print("B. 🔴 关键假设：脸在 1080p 里占比太小，192x192 内部缩放后不可辨")
print("=" * 72)
print("  MediaPipe 内部固定 192x192。1080p 里人脸通常只占 ~100x100px，")
print("  等比缩到192x192 后脸只剩 ~18x18px —— 可能低于检出门槛。")
print("  验证：裁上半身区域再送入，让脸在裁剪框里占更大比例。")
res.append(run("裁上半身 0.05-0.75h/0.2-0.8w", crop=True))
res.append(run("裁上半身 + 放大1.5x", crop=True, upscale=1.5))
res.append(run("仅放大 2x（不裁）", upscale=2.0))

print("\n" + "=" * 72)
print("C. 对照")
print("=" * 72)
print("  旧 holistic 基准          1.000")
print("  已提特征 presence[3]      ",
      end="")
fs = sorted(glob.glob(str(REPO / "artifacts/part3_features_tasksapi/train"
                        / "*.landmark.npy")))[:200]
F = np.concatenate([np.load(f)[:, 182:186] for f in fs], 0)
print("%.3f" % F[:, 3].mean())
best = max((r for r in res if r), key=lambda r: r["rate"])
print("  本次最佳%-30s %.3f" % (best["label"], best["rate"]))

out = {"sample": p.name, "n_frames": len(frames),
       "face_indices": list(FACE_INDICES), "max_index": MAXIDX,
       "old_baseline": 1.0,
       "feature_presence_mean": round(float(F[:, 3].mean()), 4),
       "runs": res, "best": best}
q = REPO / "artifacts/metrics/blank-gov/p67-face-crop-test.json"
q.parent.mkdir(parents=True, exist_ok=True)
q.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n收据 -> %s" % q)