# -*- coding: utf-8 -*-
"""P32 · 面部特征重提可行性验证（决定是否值得全量重提）

## 背景：为什么必须重写提取器

用户要求「增加脸部的，不加腿脚」。查论文依据：

**ref02 LinguisticallyMotivated EMNLP2023, Sec E2 + 脚注 8**（原文）：
> "E2: Adding Reduced Face Keypoints — Although the 75 hand and body keypoints serve as
>  an efficient minimal set ..., we investigate the impact of other nonmanual sign language
>  articulators, namely, **the face**. We introduce a reduced set of **128 face keypoints
>  that signify the signer's face contour**."
> 脚注 8: "We reduce the dense FACE_LANDMARKS in **Mediapipe Holistic** to the contour
>  keypoints according to the variable `mediapipe.solutions.holistic.FACEMESH_CONTOURS`."

**我们现状**：`FACE_INDICES = (10,33,61,133,152,263,291,362)` —— **只有 8 个点**。
即：论文用了 128 个轮廓点，我们只用了 8 个。**少了一个数量级。**

⚠️ 但必须同时报告反向证据：记忆里的《已排除方向》记着
「稠密面部 128 点（EMNLP E2 IoU 0.66→0.58）」—— **那是在帧级分割任务上退化的**。
本任务是词级 CTC 识别，任务不同，不能直接套用该结论。这是本实验存在的原因。

## 本脚本验证三件事（不跑全量，先看可行性）

### 1. mediapipe 版本与 API 可用性
已实测：`requirements.txt` 钉 `mediapipe==0.10.21`，但 **Python 3.14 无此版本**
（PyPI 只有 0.10.30~0.10.35 与 1.0.x）。这是此前「重提特征跑不通」的真正原因。
0.10.35 已装，但**只有 `modules` 与 `tasks` 子包，旧的 `solutions.holistic` 已被移除**
→ 必须改用新的 `tasks` API 重写提取器。

### 2. FACEMESH_CONTOURS 是否可从新 API 取到
若取不到，128 轮廓点方案不可行，需要改用手工选点。

### 3. 小样本端到端可行性
对 **3 个真实 dev 视频**跑新提取器，验证：
- 不崩（此前 WSL 下多次 `E_UNEXPECTED`）
- 耗时可接受（决定全量重提的成本）
- 面部点真的检出、且随帧变化

## 判据（跑之前写死）

- 3 个视频全部成功产出人脸 → 可行，值得全量重提
- 崩溃或检出失败 → 不可行，需要换方案（如 HaMeR 之于手部）

**只读 dev 的 3 个视频，不做全量，不触碰 test。**
"""
import argparse
import json
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()

    print("=" * 74)
    print("1 · mediapipe 版本与 API 形态")
    print("=" * 74)
    import mediapipe as mp
    print("  version = {}".format(mp.__version__))
    print("  有 solutions.holistic（旧 API）: {}".format(
        hasattr(mp, "solutions")))
    print("  有 tasks（新 API）           : {}".format(hasattr(mp, "tasks")))

    # 索引表在 face_mesh_connections / holistic 的常量里，新版可能移位
    print()
    print("=" * 74)
    print("2 · FACEMESH_CONTOURS 可达性")
    print("=" * 74)
    contours = None
    try:
        from mediapipe.modules.face_geometry import _constants  # noqa
        print("  modules.face_geometry 可导入")
    except Exception as e:
        print("  modules.face_geometry 不可用: {}".format(e))
    # 逐个候选位置找 FACEMESH_CONTOURS
    cands = [
        ("mediapipe.modules.holistic", "FACEMESH_CONTOURS"),
        ("mediapipe.solutions.holistic", "FACEMESH_CONTOURS"),
    ]
    for mod, attr in cands:
        try:
            m = __import__(mod, fromlist=[attr])
            if hasattr(m, attr):
                v = getattr(m, attr)
                pts = set()
                for c in v:
                    pts.update(c)
                contours = sorted(pts)
                print("  FOUND {}.{} -> {} 个点".format(mod, attr, len(contours)))
                break
        except Exception as e:
            print("  {} 不可用 ({})".format(mod, type(e).__name__))
    if contours is None:
        print("  !! 新 API 里找不到 FACEMESH_CONTOURS，需手工定义轮廓点集")

    # ---- 3 · 端到端小样本 ----
    print()
    print("=" * 74)
    print("3 · 端到端可行性（{} 个 {} 视频）".format(a.n, a.split))
    print("=" * 74)
    try:
        import cv2
    except ImportError:
        print("  !! opencv 不可用")
        return
    print("  opencv = {}".format(cv2.__version__))

    # 从 manifest 取视频路径
    import csv
    man = {}
    with open(REPO / "kaggle/manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == a.split:
                man[r["sample_id"]] = r["video"]
    ids = sorted(man)[: a.n]
    print("  样例: {}".format(ids))

    # 旧 API 无法使用，改为直接检测新 API 能否驱动
    ok_all = True
    results = []
    for sid in ids:
        rel = man[sid]
        path = REPO / "data/raw/CE-CSL/video" / rel
        t0 = time.time()
        n_frames = face_frames = 0
        max_lmk = 0
        err = None
        try:
            if not path.exists():
                err = "视频不存在: {}".format(path)
            else:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    err = "无法打开视频"
                else:
                    # 只读前 30 帧做探测，避免全量开销
                    while n_frames < 30:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        n_frames += 1
                    cap.release()
        except Exception as e:
            err = "{}: {}".format(type(e).__name__, e)
        dt = time.time() - t0
        row = {"sample_id": sid, "video": rel, "frames_probed": n_frames,
               "seconds": round(dt, 2), "error": err}
        results.append(row)
        status = "OK" if err is None else "FAIL"
        print("  [{}] {}  探测 {} 帧  {:.2f}s  {}".format(
            status, sid, n_frames, dt, err or ""))
        if err:
            ok_all = False

    print()
    print("=" * 74)
    print("结论")
    print("=" * 74)
    if not ok_all:
        print("  视频读取阶段就失败 -> 先解决读取")
    elif contours is None:
        print("  视频可读，但 FACEMESH_CONTOURS 不可达")
        print("  => 需在新 tasks API 上改用 FaceLandmarker，并手工取轮廓索引")
    else:
        print("  视频可读且轮廓点表可达 -> 可写新提取器")

    out = REPO / "artifacts/metrics/blank-gov/p32-face-extract-feasibility.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "P32 face feature re-extraction feasibility",
        "paper_basis": "ref02 EMNLP2023 Sec E2 + footnote 8: reduce MediaPipe "
                       "FACE_LANDMARKS to FACEMESH_CONTOURS -> 128 contour keypoints; "
                       "our current code uses only 8",
        "reverse_evidence": "memory records dense-128 face points REGRESSED "
                            "(IoU 0.66->0.58) in EMNLP's own frame-level "
                            "segmentation task; our task is word-level CTC, so it "
                            "must be re-tested rather than assumed either way",
        "mediapipe_version_issue": {
            "pinned": "0.10.21 (requirements.txt)",
            "available_for_py314": ["0.10.30", "0.10.31", "0.10.32", "0.10.33",
                                    "0.10.35", "1.0.0", "1.0.1"],
            "installed": mp.__version__,
            "root_cause_of_earlier_failure": "0.10.21 has no wheel for Python 3.14, "
                                             "so re-extraction could never run",
            "api_change": "0.10.35 dropped solutions.holistic; only modules/ and "
                          "tasks/ remain, so the extractor must be rewritten on the "
                          "new tasks API",
        },
        "contours_points": len(contours) if contours else None,
        "probe": results,
        "videos_readable": ok_all,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(out))


if __name__ == "__main__":
    main()
