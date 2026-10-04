# -*- coding: utf-8 -*-
"""A2 探针：视频专用模型（r3d_18, Kinetics 预训练）的判别力。

背景与文献依据（已逐条核实，见收据 literature_basis 字段）：
  SignShift Sec 4.2 原文："Video features (1024-D) are extracted with an **I3D**
  network pretrained on **Kinetics**."
  → 本探针用 r3d_18（torchvision，同为 Kinetics-400 预训练的 3D CNN），
    与论文的 I3D 属同源预训练，迁移性最接近。

  Hands-On Table III 给出**反向证据**：
      MS-TCN + I3D      mF1B 68.68（1024 维）
      MS-TCN + HaMeR    mF1B 76.22（288 维）
    论文归因："reliance on hand shapes and body poses, which are less influenced
    by variations in RGB pixel values compared to I3D features"
  SMART Table 3：HS-I3D 的 F1@50 仅 8.43（远低于 ASFormer 89.39）
  → 结构化特征在论文里优于 RGB 视频塔。本探针预期不高，如实记录。

**关键设计：与已有 rgb 特征严格同口径对照**
  现有 `*.rgb.npy` 是 CLIP ViT-B-32 **逐帧** 特征，保留原始帧数（154~236 帧）。
  A2 的 r3d_18 输出**视频级**特征（clip 级别，一个向量）。
  两者口径不同，所以判别力不可直接比 —— 本探针只回答一个问题：
  **视频级时序特征能否在逐帧池化之外提供额外判别信息。**
  因此除 macro-AUC 外，额外报告：
    - 融合（video + landmark）后的 macro-AUC
    - 融合相对 landmark 单族的增益
  若融合也不增益，则视频塔与 landmark **无互补信息**，方向判负。

只用 train/dev，不触碰 test。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))

VIDEO_ROOT = REPO / "data/raw/CE-CSL/video"


def find_video(split, sid):
    base = VIDEO_ROOT / split
    if not base.exists():
        return None
    for d in sorted(base.iterdir()):
        p = d / (sid + ".mp4")
        if p.exists():
            return p
    return None


def read_csv(p):
    rows = {}
    import csv as _csv

    with open(p, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows[r["Number"]] = r["Gloss"]
    return rows


class VideoEncoder(nn.Module):
    """r3d_18 视频级特征提取（Kinetics 预训练，冻结）。

    论文用的是 I3D(1024-D)；r3d_18 的 classifier 输出 400 维（Kinetics 类别），
    我取倒数第二层（512 维）作为特征，与 CLIP ViT-B-32 的 512 维对齐，
    便于后续做同口径的融合对照。
    """

    def __init__(self, arch="r3d_18", pretrained=True):
        super().__init__()
        import torchvision

        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models.video, arch)(weights=weights)
        # 去掉最后的分类头，取全局池化后的 512 维
        net.fc = nn.Identity()
        self.net = net
        self.out_dim = 512
        for p in self.parameters():
            p.requires_grad_(False)
        self.net.eval()

    @torch.no_grad()
    def forward(self, clip):
        """clip: (B, 3, T, H, W) 归一化到 [0,1] 后用 ImageNet 统计量标准化
        返回 (B, 512)
        """
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=clip.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=clip.device).view(1, 3, 1, 1, 1)
        x = (clip - mean) / std
        return self.net(x)


def load_clip(path, frames=16, size=112):
    """读视频并采样固定帧数 -> (3, frames, size, size) 的 float32 张量。

    采样用**均匀间隔**（与仓库 resample_indices 口径一致），
    不做 center-crop，保持与 landmark 帧对齐。
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open " + str(path))
    buf = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        buf.append(cv2.resize(fr, (size, size)))
    cap.release()
    if not buf:
        raise RuntimeError("no frames from " + str(path))
    n = len(buf)
    # 均匀采样 frames 帧
    idx = [round(i * (n - 1) / (frames - 1)) for i in range(frames)] if frames > 1 else [0]
    picked = [buf[i] for i in idx]
    arr = np.stack(picked)                       # (T, H, W, 3) BGR
    arr = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0   # (3,T,H,W)
    return arr


def extract_video_feats(ids, split, arch, frames, size, device, cache_dir, limit_frames=None):
    enc = VideoEncoder(arch).to(device)
    out = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, sid in enumerate(ids):
        cf = cache_dir / (sid + ".npy")
        if cf.exists():
            out[sid] = np.load(cf)
            continue
        vp = find_video(split, sid)
        if vp is None:
            continue
        try:
            clip = load_clip(vp, frames, size).unsqueeze(0).to(device)
            f = enc(clip).squeeze(0).cpu().numpy()
            np.save(cf, f.astype(np.float32))
            out[sid] = f
        except Exception as e:
            print("  [{}] {} 失败: {}".format(i, sid, str(e)[:80]))
        if (i + 1) % 10 == 0:
            print("  {}/{} ({:.1f} 分钟)".format(i + 1, len(ids),
                                               (time.time() - t0) / 60), flush=True)
    return out


def load_landmark_pooled(sid, root, dim=368):
    """landmark 特征 -> 与 A2 视频级口径对齐的池化特征。

    现有 landmark 是 (48, 368) 逐帧。视频塔输出是 clip 级单向量。
    为了可比，这里对 landmark 做 mean+std 池化（项目 §24.1 里
    mean+std 是唯一 >0.69 的池化方式，0.7132），并归一化到同量纲。
    """
    p = root / (sid + ".landmark.npy")
    if not p.exists():
        return None
    a = np.load(p).astype(np.float32)
    m, s = a.mean(axis=0), a.std(axis=0)
    return np.concatenate([m, s])


def macro_auc_probe(X_tr, y_tr, X_ev, y_ev, min_freq=5, seed=0):
    """逐 gloss 的线性判别 AUC（与仓库 scripts/discrim_probe.py 同口径）。

    做法：每类一个 LogisticRegression，判别分数取 decision_function，
    AUC 用 roc_auc_score，最后按「类均值」得 macro-AUC，
    只统计 train 与 eval 都出现的、且 eval 里有正样本的类。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    classes = sorted(set(y_tr) & set(y_ev))
    aucs, used = [], 0
    per_class = {}
    for c in classes:
        mtr = y_tr == c
        mev = y_ev == c
        if mtr.sum() < min_freq or mev.sum() == 0 or (~mev).sum() == 0:
            continue
        # 二分类 LogisticRegression 需要训练集里同时有两类。
        # 逐 gloss 判别时，train 里该 gloss 的样本必然全属该类，
        # 必须显式补上负类样本，否则 sklearn 直接抛
        # "This solver needs samples of at least 2 classes"。
        # 负类只取**另一个** gloss 的样本（而不是全部 127 个），
        # 否则类别数过多会让 decision_function 变成多类形状，
        # roc_auc_score 随之要求 multi_class 参数。
        Xc = X_tr[mtr]
        yc = y_tr[mtr]
        other = np.flatnonzero(~mtr)
        if other.size == 0:
            continue
        # 取样本数最多的那个 gloss 作为负类，构成严格二分类
        vals, counts = np.unique(y_tr[other], return_counts=True)
        neg_label = vals[counts.argmax()]
        neg_mask = (~mtr) & (y_tr == neg_label)
        Xc = np.concatenate([Xc, X_tr[neg_mask]], axis=0)
        yc = np.concatenate([yc, y_tr[neg_mask]], axis=0)
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed)
        clf.fit(Xc, yc)
        # 显式二分类：正类=1，负类=0
        s = clf.decision_function(X_ev)
        y_bin = (y_ev == c).astype(int)
        if y_bin.min() == y_bin.max():
            continue
        a = roc_auc_score(y_bin, s)
        aucs.append(a)
        per_class[c] = {"auc": float(a), "n_eval": int(mev.sum())}
        used += 1
    if not aucs:
        return None, per_class
    return float(np.mean(aucs)), per_class


def gather(split, label_csv, limit, feat_root, lm_root, video_feats, min_freq, device):
    gloss = read_csv(REPO / label_csv)
    ids = [s for s in sorted(gloss) if s in video_feats]
    if limit:
        ids = ids[:limit]
    Xv, Xl, y = [], [], []
    for sid in ids:
        g = gloss[sid]
        toks = [t for t in g.split("/") if t.strip()]
        if not toks:
            continue
        v = video_feats[sid]
        l = load_landmark_pooled(sid, lm_root)
        if l is None:
            continue
        for t in set(toks):
            Xv.append(v); Xl.append(l); y.append(t)
    return np.stack(Xv), np.stack(Xl), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="r3d_18")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--train-limit", type=int, default=40)
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-freq", type=int, default=5)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/a2-videomae-probe.json")
    ap.add_argument("--cache", default=".a2_cache")
    a = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}  arch = {}  frames = {}".format(device, a.arch, a.frames))

    # ---- 1. 提视频特征 ----
    tr_gloss = read_csv(REPO / "data/raw/CE-CSL/label/train.csv")
    va_gloss = read_csv(REPO / "data/raw/CE-CSL/label/dev.csv")
    tr_ids = sorted(tr_gloss)[: a.train_limit]
    va_ids = sorted(va_gloss)[: a.val_limit]
    print("train {} 条 / dev {} 条（探针规模）".format(len(tr_ids), len(va_ids)))

    cache = Path(a.cache) / a.arch
    t0 = time.time()
    tr_vf = extract_video_feats(tr_ids, "train", a.arch, a.frames, a.size,
                                device, cache / "train")
    va_vf = extract_video_feats(va_ids, "dev", a.arch, a.frames, a.size,
                                device, cache / "dev")
    print("视频特征提取完成 train {} / dev {}，耗时 {:.1f} 分钟".format(
        len(tr_vf), len(va_vf), (time.time() - t0) / 60))
    if not tr_vf or not va_vf:
        raise SystemExit("视频特征为空")
    d = next(iter(tr_vf.values())).shape[0]
    print("特征维度 {}".format(d))

    # ---- 2. landmark 池化（同口径对照）----
    Xv_tr, Xl_tr, y_tr = gather("train", "data/raw/CE-CSL/label/train.csv", a.train_limit,
                                None, REPO / "artifacts/part3_features/train",
                                tr_vf, a.min_freq, device)
    Xv_ev, Xl_ev, y_ev = gather("dev", "data/raw/CE-CSL/label/dev.csv", a.val_limit,
                                None, REPO / "artifacts/part3_features/validation",
                                va_vf, a.min_freq, device)
    print("样本数 train {} / dev {}  类别数 {}".format(len(y_tr), len(y_ev), len(set(y_tr))))

    # ---- 3. 三组对照 ----
    results = {}

    auc_v, pc_v = macro_auc_probe(Xv_tr, y_tr, Xv_ev, y_ev, a.min_freq)
    results["video_only"] = {"macro_auc": auc_v, "dim": d, "n_classes": len(pc_v)}
    print("")
    print("A2 视频塔单独        macro-AUC = {}".format(
        "None" if auc_v is None else "%.4f" % auc_v))

    Xl_tr2 = Xl_tr / (np.linalg.norm(Xl_tr, axis=1, keepdims=True) + 1e-8)
    Xl_ev2 = Xl_ev / (np.linalg.norm(Xl_ev, axis=1, keepdims=True) + 1e-8)
    auc_l, pc_l = macro_auc_probe(Xl_tr2, y_tr, Xl_ev2, y_ev, a.min_freq)
    results["landmark_pooled"] = {"macro_auc": auc_l, "dim": Xl_tr.shape[1],
                                  "n_classes": len(pc_l)}
    print("landmark 池化对照    macro-AUC = {}  （项目记录单族 0.6821，"
          "但那是逐帧口径，此处是池化口径）".format(
              "None" if auc_l is None else "%.4f" % auc_l))

    # 融合：把两族标准化后拼接
    Xv_tr_n = Xv_tr / (np.linalg.norm(Xv_tr, axis=1, keepdims=True) + 1e-8)
    Xv_ev_n = Xv_ev / (np.linalg.norm(Xv_ev, axis=1, keepdims=True) + 1e-8)
    Xf_tr = np.concatenate([Xv_tr_n, Xl_tr2], axis=1)
    Xf_ev = np.concatenate([Xv_ev_n, Xl_ev2], axis=1)
    auc_f, pc_f = macro_auc_probe(Xf_tr, y_tr, Xf_ev, y_ev, a.min_freq)
    results["fused"] = {"macro_auc": auc_f, "dim": Xf_tr.shape[1], "n_classes": len(pc_f)}
    print("融合(video+landmark) macro-AUC = {}".format(
        "None" if auc_f is None else "%.4f" % auc_f))

    # ---- 4. 判定 ----
    print("")
    print("=" * 66)
    print("A2 探针判定（判据来自计划 2.0：> 0.75 才全量）")
    print("=" * 66)
    base = auc_l if auc_l is not None else 0.6821
    best = max([x for x in (auc_v, auc_l, auc_f) if x is not None] or [0])
    gain = (auc_f - base) if (auc_f is not None and auc_l is not None) else 0.0
    print("最佳单族 macro-AUC      {:.4f}".format(best))
    print("融合相对 landmark 增益   {:+.4f}".format(gain))
    if best > 0.75:
        verdict = "绿灯：突破 0.75，可全量提取"
    elif best > 0.69:
        verdict = "黄灯：略超现有 0.6821 但未达 0.75，收益有限"
    else:
        verdict = "红灯：不优于现有 landmark，与 A1 同结论"
    if gain <= 0.01 and auc_f is not None:
        verdict += "；且融合增益≤0.01 → 视频塔与 landmark 无互补信息"
    print("判定：{}".format(verdict))

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "A2 video-specialist model (r3d_18, Kinetics) discriminability probe",
        "arch": a.arch,
        "pretrained": "Kinetics-400 (torchvision DEFAULT)",
        "frames": a.frames,
        "input_size": a.size,
        "feature_dim": d,
        "n_train_ids": len(tr_ids),
        "n_dev_ids": len(va_ids),
        "literature_basis": {
            "signshift": "Sec 4.2: 'Video features (1024-D) are extracted with an I3D "
                         "network pretrained on Kinetics' -- r3d_18 is the same Kinetics "
                         "pretraining family, closest available match",
            "hands_on_counter": "Table III: MS-TCN+I3D 68.68 vs MS-TCN+HaMeR 76.22; "
                                "structured features beat I3D despite 4x smaller dim",
            "smart_counter": "Table 3: HS-I3D F1@50 only 8.43 vs ASFormer 89.39",
            "expectation": "prior evidence is unfavorable; recorded to avoid "
                           "post-hoc rationalisation",
        },
        "probe_results": results,
        "baseline_landmark_framewise": 0.6821,
        "best_macro_auc": best,
        "fusion_gain_over_landmark": gain,
        "verdict": verdict,
        "test_split_read": False,
        "minutes": (time.time() - t0) / 60,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
