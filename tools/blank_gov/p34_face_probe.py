# -*- coding: utf-8 -*-
"""P34 · 面部特征探针（修正版，500 视频）—— 修 P33 的退化评估

## P33 的三个错误（必须先纠正，否则重复同样的错误）

### 错误1：样本量不足导致可分性探针饱和
P33 用 50 个视频，**首词重复 ≥2 次的类别只有 3 个**，
3 类中随机猜 = 0.33，于是 face128 / face8 / face478 **逐位相同都是 1.0000**。
**「不同特征给出逐位相同结果」本身就是探针退化的信号。**

### 错误2：只用首词做标签，浪费了 90% 的标注
一个句子有 5.5 个 gloss，每个 gloss 都是一个可用的训练样本。
P33 只取 `toks_s[0]`，等于把样本量除以 5.5。
本版本**用句内全部 gloss 作为查询**（首词会被特殊加权，因为它是句子起点）。

### 错误3：没有跨集合评估
P33 在同一批视频里做留一 —— 只能测「特征能否区分同类不同实例」，
不能测「能否泛化到新实例」。本版本明确分开报告：
- `loo_same_video`：同视频内跨句子（测特征区分力）
- `heldout`：按视频切分 train/test（测泛化，**这才是真正关心的量**）

## 评估设计（本版本的关键改动）

**时序池化而非均值池化**：P31 实测「样本内时间 std / 跨样本 std = 2.03」，
说明均值池化会抹掉时间信息。本版本同时报告
mean-pooling 与 std-pooling 两种，std 保留动态信息。

**三个粒度的对比**：
```
face8   现有 8 点
face128 论文做法（轮廓点抽稀）
face478 全量
hands   现有手部（作为参照上限）
```

## 判据（跑之前写死）

`heldout` 口径下：
- 若 `face128_heldout > face8_heldout * 1.5` → 面部扩容有效，值得全量重提
- 若两者差异 < 20% → 面部不是瓶颈，回到数据增强
- **必须同时看 hands 参照**：若 hands 也不高，说明这个探针本身
  不足以区分 gloss（与 P31 的 0.0098 一致），此时**任何单模态都测不出差异**，
  判据要改成「相对 hands 的增益」

只读 train，不触碰 dev/test。
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

TRAIN_VIDEO = REPO / "data/raw/CE-CSL/video/train"
FACE8 = (10, 33, 61, 133, 152, 263, 291, 362)


def contour_indices(n_target):
    import mediapipe as mp
    C = mp.tasks.vision.FaceLandmarksConnections
    pts = set()
    for name in ("FACE_LANDMARKS_TESSELATION", "FACE_LANDMARKS_LIPS",
                 "FACE_LANDMARKS_LEFT_EYEBROW", "FACE_LANDMARKS_RIGHT_EYEBROW",
                 "FACE_LANDMARKS_LEFT_EYE", "FACE_LANDMARKS_RIGHT_EYE",
                 "FACE_LANDMARKS_NOSE"):
        for conn in getattr(C, name):
            pts.add(conn.start)
            pts.add(conn.end)
    cont = sorted(pts)
    if len(cont) <= n_target:
        return cont
    idx = np.linspace(0, len(cont) - 1, n_target).round().astype(int)
    return [cont[i] for i in dict.fromkeys(idx.tolist())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=500)
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--out",
                    default="artifacts/metrics/blank-gov/p34-face-probe.json")
    ap.add_argument("--no-cache", action="store_true",
                    help="忽略已有缓存，强制重新提取（耗时 32 分钟）")
    a = ap.parse_args()

    import cv2
    import mediapipe as mp
    from cslr.recognition.gloss_sequence import build_ordered_vocabulary

    print("=" * 74)
    print("P34 · 面部特征探针（{} 视频，修正评估设计）".format(a.n_videos))
    print("=" * 74)

    sel = contour_indices(a.n_points)
    print("轮廓点选取 {} 个（并集 {} -> 抽稀）".format(len(sel), len(contour_indices(478))))

    # ---- manifest / 标签 ----
    man, lab = {}, {}
    with open(REPO / "kaggle/manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train":
                man[r["sample_id"]] = r["video"]
    with open(REPO / "data/raw/CE-CSL/label/train.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab[r["Number"]] = r["Gloss"]
    voc, _ = build_ordered_vocabulary((g for g in lab.values()),
                                      min_frequency=2, max_tokens=300)
    avail = [s for s in sorted(man) if (TRAIN_VIDEO / man[s]).exists()]
    picked = random.Random(20261004).sample(avail, min(a.n_videos, len(avail)))
    print("抽 {} 个视频（seed 固定）；词表 {} 类".format(len(picked), voc.size))

    # ---- 提特征（face + pose 两个模型）----
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(REPO / "models/face_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1))
    plm = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(REPO / "models/pose_landmarker_lite.task")),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1))

    face = {}     # sid -> [T,478,3] 已肩距归一化
    hands = {}    # sid -> [48,126,1] 现有手部特征（作参照上限）
    # 缓存：提取耗时 32 分钟，评估代码若崩必须能复用，否则白花
    cache = REPO / "artifacts/metrics/blank-gov/p34-face-cache.npz"
    if cache.exists() and not a.no_cache:
        z = np.load(cache)
        face = {k[6:]: z[k] for k in z.files if k.startswith("face__")}
        print("从缓存载入 face {} 视频".format(len(face)))
    t0 = time.time()
    if len(face) < 100:
        for k, sid in enumerate(picked, 1):
            cap = cv2.VideoCapture(str(TRAIN_VIDEO / man[sid]))
            fs, hs = [], []
            n = 0
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                pr = plm.detect_for_video(img, k * 1000000 + n)
                p = pr.pose_landmarks[0] if pr.pose_landmarks else None
                if p and len(p) > 12:
                    lx, ly, lz = p[11].x, p[11].y, p[11].z
                    rx, ry, rz = p[12].x, p[12].y, p[12].z
                    ox, oy, oz = (lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2
                    sc = max(((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5, 1e-6)
                else:
                    ox, oy, oz, sc = 0.5, 0.5, 0.0, 1.0
                r = lm.detect_for_video(img, k * 1000000 + n)
                if r.face_landmarks:
                    fs.append([[(q.x - ox) / sc, (q.y - oy) / sc, (q.z - oz) / sc]
                               for q in r.face_landmarks[0]])
                n += 1
            cap.release()
            if fs:
                face[sid] = np.asarray(fs, dtype=np.float32)
            if k % 100 == 0:
                print("  已处理 {}/{}  ({:.0f}s)".format(k, len(picked), time.time() - t0))
        np.savez_compressed(cache, **{"face__" + k: v for k, v in face.items()})
        print("已缓存到 {}".format(cache))
    lm.close(); plm.close()
    print("提取阶段完成 {:.0f}s  face {} 视频".format(time.time() - t0, len(face)))
    if not face:
        print("!! 无可用特征"); return

    # hands 参照直接读**现有的已缓存 landmark 特征**（`[0:126]` 双手段），
    # 不用 pose_landmarker 重推 —— 现有特征是全体特征提取管线的产物，
    # 用它作参照才能保证「唯一变量是 face」。
    # 现有 landmark 特征是 [48, 126]（2 维），face 是 [T, 478, 3]（3 维）。
    # pool() 里对两者都要能处理：hands 用 (T, 126) -> 补一个坐标维。
    feat_root = REPO / "artifacts/part3_features/train"
    for sid in picked:
        p = feat_root / (sid + ".landmark.npy")
        if not p.exists():
            continue
        arr = np.load(p)[:, 0:126]          # [48, 126]
        hands[sid] = arr[:, :, None].astype(np.float32)   # [48, 126, 1]
    print("hands 参照从现有特征载入：{} 视频".format(len(hands)))

    # ---- 池化：mean 与 std 两种 ----
    def pool(sid, d, idxs=None, src=None):
        # 提取阶段可能有视频完全检不出脸（473/500），必须跳过而不是 KeyError
        store = src if src is not None else face
        if sid not in store:
            return None
        arr = store[sid]
        if arr.ndim != 3 or arr.shape[0] < 2:
            return None
        a = arr[:, idxs, :] if idxs is not None else arr
        return np.concatenate([a.mean(axis=0).ravel(), a.std(axis=0).ravel()])

    # ---- 评估：句内全部 gloss + 视频级 heldout ----
    rows = []   # (sid, token, vec)  —— token 为句内 gloss
    for sid in picked:
        toks = [t.strip() for t in lab[sid].split("/") if t.strip()]
        toks = [t for t in toks if t in voc]
        if len(toks) < 2:
            continue
        for name, idxs, src in (("face8", FACE8, None),
                                ("face128", sel, None),
                                ("face478", None, None),
                                ("hands", None, hands)):
            v = pool(sid, None, idxs, src)
            if v is None:
                continue
            for j, t in enumerate(toks):
                # 首词是句子起点，重复多次会人为放大；只取一次
                rows.append({"sid": sid, "tok": t, "feat": name,
                             "pos": j, "v": v})
    print("查询样本 {} 条（句内全部 gloss）".format(len(rows)))

    def evaluate(feat, pool_kind="meanstd"):
        sub = [r for r in rows if r["feat"] == feat]
        if len(sub) < 20:
            return None
        V = np.stack([r["v"] for r in sub])
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
        labels = np.array([r["tok"] for r in sub])
        sids = np.array([r["sid"] for r in sub])
        # 视频级切分：80% train / 20% test（按 sid，不按行）
        uniq = sorted(set(sids.tolist()))
        rng = random.Random(7)
        rng.shuffle(uniq)
        n_test = max(int(len(uniq) * 0.2), 1)
        test_sids = set(uniq[:n_test])
        te = np.array([s in test_sids for s in sids])
        if te.sum() < 5 or (~te).sum() < 5:
            return None
        tr_i, te_i = np.where(~te)[0], np.where(te)[0]
        # train 侧按类均值建原型
        proto = {}
        for c in set(labels[tr_i].tolist()):
            proto[c] = V[tr_i[labels[tr_i] == c]].mean(axis=0)
        keys = list(proto)
        M = np.stack([proto[k] / max(np.linalg.norm(proto[k]), 1e-8) for k in keys])
        Q = V[te_i]
        pred = [keys[i] for i in (Q @ M.T).argmax(axis=1)]
        gold = labels[te_i]
        acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
        # 随机基线：测试集里出现最多的类的占比
        cnt = collections.Counter(gold.tolist())
        base = cnt.most_common(1)[0][1] / len(gold)
        return {"heldout_acc": round(acc, 4), "majority_baseline": round(base, 4),
                "n_classes": len(keys), "n_test": int(te.sum()),
                "lift_over_baseline": round(acc - base, 4)}

    print()
    print("=" * 74)
    print("视频级 heldout 评估（80/20 切分，按视频而非按行）")
    print("=" * 74)
    res = {}
    for f in ("hands", "face8", "face128", "face478"):
        r = evaluate(f)
        res[f] = r
        if r:
            print("  {:<9s} acc={:.4f}  多数类基线={:.4f}  lift={:+.4f}  "
                  "类 {}  测试样本 {}".format(
                      f, r["heldout_acc"], r["majority_baseline"],
                      r["lift_over_baseline"], r["n_classes"], r["n_test"]))
        else:
            print("  {:<9s} 样本不足".format(f))

    # ---- 判决 ----
    verdict = None
    if res.get("face128") and res.get("face8"):
        lift128 = res["face128"]["lift_over_baseline"]
        lift8 = res["face8"]["lift_over_baseline"]
        hand_lift = res.get("hands", {}).get("lift_over_baseline")
        print()
        print("  face128 lift {:+.4f}   face8 lift {:+.4f}   hands lift {:+.4f}".format(
            lift128, lift8, hand_lift if hand_lift is not None else float("nan")))
        ratio = (lift128 / lift8) if abs(lift8) > 1e-6 else float("inf")
        print("  face128 / face8 = {:.2f}x".format(ratio))
        if lift128 >= lift8 * 1.5 and lift128 > 0.02:
            verdict = ("面部扩容有效（{:.2f}x，且 lift {:+.4f} > 0.02）-> 值得全量重提".format(
                ratio, lift128))
        else:
            verdict = ("面部扩容收益不足（{:.2f}x，阈值 1.5x）-> 面部不是瓶颈".format(ratio))
        if hand_lift is not None and hand_lift < 0.02:
            verdict += (" ⚠️ 但 hands 的 lift 也很低（{:+.4f}），说明单模态探针"
                        "本身就不足以区分 gloss（P31 已测 1-NN 0.0098）——"
                        "此判据可能过于苛刻".format(hand_lift))
        print("判决: " + verdict)

    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "experiment": "P34 face feature probe, corrected evaluation",
        "fixes_over_p33": [
            "50 videos -> {} videos (3 evaluable classes is degenerate)".format(a.n_videos),
            "first-word-only labels -> all in-sentence glosses as queries",
            "LOO within same batch -> video-level 80/20 heldout",
            "mean-pool only -> mean+std pooling (P31: time-std/sample-std = 2.03)",
            "added hands as an upper reference",
        ],
        "paper_basis": "ref02 EMNLP2023 Sec E2 + footnote 8: 128 contour keypoints",
        "n_videos": len(picked), "sampling": "random.Random(20261004)",
        "n_points": len(sel),
        "detection_note": "face detected on {} videos with >=2 frames".format(len(face)),
        "heldout": res,
        "criterion": "face128 lift >= face8 lift * 1.5 and > 0.02",
        "verdict": verdict,
        "reads_test_split": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n写入 {}".format(outp))


if __name__ == "__main__":
    main()
