# -*- coding: utf-8 -*-
"""P33 · 面部 128 轮廓点重提 —— 小规模验证（50 个视频，先不投全量）

## 目标

ref02 EMNLP2023 Sec E2 + 脚注 8 的做法：
> "We introduce a reduced set of **128 face keypoints** that signify the signer's face
>  contour. We reduce the dense FACE_LANDMARKS in Mediapipe Holistic to the contour
>  keypoints according to the variable `FACEMESH_CONTOURS`."

我们现状只有 8 个点（`FACE_INDICES=(10,33,61,133,152,263,291,362)`），
**face 块仅占 368 维的 6.5%**。本脚本验证 128 点能带来多少增量。

## 为什么先小规模

全量 4973 个 train 视频按 77 fps 约需 3.5 小时。若 128 点本身无效，
这 3.5 小时白花。小规模验证只需几分钟，能先回答：
1. 128 点能否稳定检出（检出率）
2. 是否随帧变化（静态可分性的前提）
3. 相比现有 8 点，可分性有没有提升（**决定性判据**）

## 现有视频路径的坑（实测）

- `data/raw/CE-CSL/video/train/{A..L}/*.mp4` → 4973 个 ✅
- `data/raw/CE-CSL/video/validation/` → **0 个（空目录）**，但 514 个 landmark 特征还在
- dev 视频实际在 `/mnt/c/Users/su127/Desktop/csl视频/dev/{A..L}/`

## 判据（跑之前写死）

- `1NN(face128) > 1NN(face8) * 1.5` → **面部扩容有效**，值得投 3.5 小时全量
- 检出率 < 90% → 128 点不稳定，先解决检出
- 两者都不满足 → 面部不是瓶颈，回到数据增强方向

只读 train（不碰 dev/test 视频），不读 test split。
"""
import argparse
import collections
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

MODEL = REPO / "models" / "face_landmarker.task"
TRAIN_VIDEO = REPO / "data/raw/CE-CSL/video/train"
FACE8 = (10, 33, 61, 133, 152, 263, 291, 362)   # 现有 8 点


def contour_indices():
    """从 FaceLandmarksConnections 取轮廓点集合（等价旧 FACEMESH_CONTOURS）。"""
    import mediapipe as mp
    C = mp.tasks.vision.FaceLandmarksConnections
    pts = set()
    for name in ("FACE_LANDMARKS_TESSELATION",
                 "FACE_LANDMARKS_LIPS",
                 "FACE_LANDMARKS_LEFT_EYEBROW",
                 "FACE_LANDMARKS_RIGHT_EYEBROW",
                 "FACE_LANDMARKS_LEFT_EYE",
                 "FACE_LANDMARKS_RIGHT_EYE",
                 "FACE_LANDMARKS_NOSE"):
        for conn in getattr(C, name):
            pts.add(conn.start)
            pts.add(conn.end)
    return sorted(pts)


def pick_n(pts, n):
    """若并集超过 n 个点，等距抽稀到 n 个（保持覆盖全脸）。"""
    if len(pts) <= n:
        return list(pts)
    idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
    return [pts[i] for i in dict.fromkeys(idx.tolist())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=50)
    ap.add_argument("--n-points", type=int, default=128,
                    help="ref02 EMNLP 用 128 个轮廓点")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0=整段；调试时可设 40")
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p33-face128-pilot.json")
    a = ap.parse_args()

    import cv2
    import mediapipe as mp

    print("=" * 74)
    print("P33 · 面部 {} 点重提 —— 小规模验证（{} 视频）".format(a.n_points, a.n_videos))
    print("=" * 74)

    # ---- 点集 ----
    cont = contour_indices()
    print("轮廓连接集并集点数 = {}".format(len(cont)))
    sel = pick_n(cont, a.n_points)
    print("选取点数 = {}   {}".format(len(sel), sel[:12] + ["..."] if len(sel) > 12 else sel))
    sel_set = set(sel)
    face8_in = [i for i in FACE8 if i in sel_set]
    print("现有 8 点中被包含的: {}/8 {}".format(len(face8_in), face8_in))

    # ---- manifest ----
    man = {}
    with open(REPO / "kaggle/manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                man[r["sample_id"]] = r["video"]
    # 只保留 train 视频目录里真实存在的
    avail = [s for s in sorted(man)
             if (TRAIN_VIDEO / man[s]).exists()]
    print("train 视频可用 {} 个".format(len(avail)))
    picked = random.Random(20261004).sample(avail, min(a.n_videos, len(avail)))
    print("随机抽取（seed=20261004）{} 个: {}".format(len(picked), picked[:5]))

    # ---- 提特征 ----
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1)
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    out = {}
    t_all = time.time()
    # 现有 landmark 的归一化基准：双肩中点 + 肩距（extractor.py:_normalization_from_pose）。
    # FaceLandmarker 输出的是**归一化图像坐标**（约 0~1），与现有特征的尺度差 5.5 倍，
    # 必须施加同样的肩距归一化，否则 1-NN 的欧氏距离失去意义。
    import mediapipe as mp
    pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(REPO / "models/pose_landmarker_lite.task")),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    pose_lm = None
    pose_model = REPO / "models/pose_landmarker_lite.task"
    if pose_model.exists():
        pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt)
        print("pose_landmarker 已加载，用于计算肩距基准")
    else:
        print("⚠️ 无 pose_landmarker 模型，退化为按图像宽高归一化")

    IMAGE_W, IMAGE_H = 640.0, 360.0   # 退化路径的基准

    for k, sid in enumerate(picked, 1):
        path = TRAIN_VIDEO / man[sid]
        cap = cv2.VideoCapture(str(path))
        seq = []
        norms = []          # 每帧的 (origin_x, origin_y, scale)
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            r = lm.detect_for_video(img, k * 1000000 + n)
            # 肩距基准：优先用 pose 的 11/12 号点（与现有 extractor 一致）
            if pose_lm is not None:
                pr = pose_lm.detect_for_video(img, k * 1000000 + n)
                plm = pr.pose_landmarks[0] if pr.pose_landmarks else None
                if plm and len(plm) > 12:
                    lx, ly, lz = plm[11].x, plm[11].y, plm[11].z
                    rx, ry, rz = plm[12].x, plm[12].y, plm[12].z
                    ox, oy, oz = (lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2
                    sc = max(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5, 1e-6)
                else:
                    ox, oy, oz, sc = 0.5, 0.5, 0.0, 1.0
            else:
                ox, oy, oz, sc = 0.5, 0.5, 0.0, 1.0
            norms.append((ox, oy, oz, sc))
            if r.face_landmarks:
                pts = r.face_landmarks[0]
                seq.append([((p.x - ox) / sc, (p.y - oy) / sc, (p.z - oz) / sc)
                            for p in pts])
            n += 1
            if a.max_frames and n >= a.max_frames:
                break
        cap.release()
        out[sid] = seq
        if k <= 3 or k % 10 == 0:
            print("  [{}] {}  帧 {}  检出人脸 {} ({:.0f}%)".format(
                k, sid, n, len(seq), 100.0 * len(seq) / max(n, 1)))
    lm.close()
    if pose_lm is not None:
        pose_lm.close()
    dt = time.time() - t_all

    # ---- 统计 ----
    print()
    print("=" * 74)
    print("检出率与运动量")
    print("=" * 74)
    det = []
    for sid in picked:
        path = TRAIN_VIDEO / man[sid]
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if a.max_frames:
            n = min(n, a.max_frames)
        det.append((sid, n, len(out[sid])))
    all_n = sum(n for _, n, _ in det)
    all_hit = sum(h for _, _, h in det)
    print("  总体检出率 = {}/{} = {:.1%}".format(all_hit, all_n, all_hit / max(all_n, 1)))
    per = [h / max(n, 1) for _, n, h in det]
    print("  单视频检出率  mean {:.1%}  min {:.1%}  <50% 的视频 {} 个".format(
        float(np.mean(per)), float(np.min(per)),
        sum(1 for x in per if x < 0.5)))

    # 运动量：128 点序列的帧间位移
    disp = []
    for sid, seq in out.items():
        if len(seq) < 2:
            continue
        a0 = np.asarray(seq[:-1], dtype=np.float32)
        a1 = np.asarray(seq[1:], dtype=np.float32)
        idx = [i for i, p in enumerate(sel) if p < min(a0.shape[1], a1.shape[1])]
        if not idx:
            continue
        d = np.linalg.norm(a1[:, idx, :] - a0[:, idx, :], axis=-1)
        disp.append(float(d.mean()))
    if disp:
        da = np.array(disp)
        print("  128 点帧间平均位移  mean {:.5f}  median {:.5f}  p90 {:.5f}".format(
            da.mean(), np.median(da), np.percentile(da, 90)))
        print("  （对照：现有 hands 段位移 0.5447，pose 1.1780）")

    # ---- 可分性：128 点 vs 现有 8 点 ----
    print()
    print("=" * 74)
    print("静态可分性：128 点 vs 现有 8 点（1-NN，train 原型）")
    print("=" * 74)
    # 参考标签
    lab = {}
    with open(REPO / "data/raw/CE-CSL/label/train.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab[r["Number"]] = r["Gloss"]

    def mean_pool(sid, idxs, src=None):
        seq = out[sid]
        if len(seq) < 2:
            return None
        a = np.asarray(seq, dtype=np.float32)[:, idxs, :]   # [T,K,3]
        return a.mean(axis=0).ravel()

    from cslr.recognition.gloss_sequence import build_ordered_vocabulary
    voc, _ = build_ordered_vocabulary((g for g in lab.values()),
                                      min_frequency=2, max_tokens=300)

    def build_and_score(idxs, tag):
        """留一法 1-NN（LOO）。

        做法：每个样本与「同类的其它样本」比余弦相似度，取最高者为预测。
        这样每个查询的候选里天然不含自身，不需要额外的排除逻辑，
        也不会像普通原型 1-NN 那样出现「自己命中自己」的平凡正确。
        """
        by_class = collections.defaultdict(list)
        vecs, toks = [], []
        for i, sid in enumerate(picked):
            toks_s = [t.strip() for t in lab[sid].split("/") if t.strip()]
            if not toks_s:
                continue
            v = mean_pool(sid, idxs)
            if v is None:
                continue
            vecs.append(v)
            toks.append(toks_s[0])
        V = np.stack(vecs)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
        for i, gt in enumerate(toks):
            by_class[gt].append(i)
        # 词表内的词才算（OOV 无法评估）
        usable = {k: v for k, v in by_class.items() if k in voc and len(v) >= 2}
        c = n = 0
        for gt, idxs_ in usable.items():
            Q = V[idxs_]                       # [m, D]
            S = Q @ Q.T                        # 类内相似度矩阵
            for a_ in range(len(idxs_)):
                S[a_, a_] = -1.0               # 排除自身
                best_j = int(np.argmax(S[a_]))
                n += 1
                if toks[idxs_[best_j]] == gt:
                    c += 1
        acc = c / max(n, 1)
        print("  {:<14s} 可评估类 {:3d}  查询 {:3d}  LOO-1NN acc {:.4f}".format(
            tag, len(usable), n, acc))
        return acc

    acc128 = build_and_score(sel, "face128")
    acc8 = build_and_score([p for p in FACE8], "face8(现有)")
    acc478 = build_and_score(list(range(478)), "face478(全量)")

    verdict = None
    if acc128 is not None and acc8 is not None:
        ratio = acc128 / acc8 if acc8 > 1e-9 else float("inf")
        print()
        print("  128点 / 8点 = {:.2f}x".format(ratio))
        if acc128 >= acc8 * 1.5:
            verdict = ("面部扩容有效（{:.2f}x，判据 1.5x）-> 值得投全量重提"
                       "（4973 视频约 3.5 小时）".format(ratio))
        else:
            verdict = ("面部扩容收益不足（{:.2f}x < 1.5x）-> 128 点也解决不了，"
                       "应回到数据增强方向".format(ratio))
        print("判决: " + verdict)
        if all_n and all_hit / max(all_n, 1) < 0.9:
            print("⚠️ 但检出率 <90%，需先解决检出")

    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "experiment": "P33 face-128 pilot re-extraction",
        "paper_basis": "ref02 EMNLP2023 Sec E2 + footnote 8: reduce MediaPipe "
                       "FACE_LANDMARKS to FACEMESH_CONTOURS -> 128 contour keypoints",
        "why_pilot": "full extraction is ~3.5h; verify value first",
        "n_videos": len(picked),
        "sampling": "random.Random(20261004).sample(sorted(available train ids), 50)",
        "n_points_selected": len(sel),
        "contour_union_size": len(cont),
        "face8_covered_by_selection": len(face8_in),
        "detection": {"overall": round(all_hit / max(all_n, 1), 4),
                      "per_video_mean": round(float(np.mean(per)), 4),
                      "videos_below_50pct": sum(1 for x in per if x < 0.5)},
        "motion": {"mean_frame_displacement": round(float(np.mean(disp)), 5) if disp else None,
                   "reference": "existing hands 0.5447 / pose 1.1780"},
        "separability_1nn": {"face128": acc128, "face8": acc8, "face478": acc478},
        "ratio_128_over_8": round((acc128 / acc8) if (acc8 and acc8 > 1e-9) else -1, 3),
        "criterion": "acc128 >= acc8 * 1.5",
        "verdict": verdict,
        "seconds": round(dt, 1),
        "reads_test_split": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(outp))


if __name__ == "__main__":
    main()
