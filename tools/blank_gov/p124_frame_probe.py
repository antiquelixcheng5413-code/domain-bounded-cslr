"""P124b：判别力探针 —— 判定「自监督预训练的主干」是否真的更好。

═══════════════════════════════════════════════════════════════
为什么先探针再训练（而不是直接跑 20 轮微调）
═══════════════════════════════════════════════════════════════
- 自监督预训练在 CSLR/CE-CSL 上**没有任何论文验证过**（依据【无】）；
  反向证据还有：官方 CorrNet（自监督一致性）47.2 比 VAC 45.1 差 2.1 pp。
- 一次 20 轮微调 ≈ 9 小时。探针只要 ≈ 10 分钟（纯前向 + 统计）。
⇒ 先用便宜的零号自检决定要不要花那 9 小时。

═══════════════════════════════════════════════════════════════
判据（预先写死，防作弊；且阈值必须在随机基线之上）
═══════════════════════════════════════════════════════════════
主判据：ΔCohen's d = d(自监督) − d(ImageNet)
   Cohen's d 的随机基线 = 0（同/异类距离无差异）
   判活：Δd ≥ +0.05 且 bootstrap 95% CI 下界 > 0
   判死：Δd ≤ +0.01 或 CI 上界 < +0.05
   灰区：其余 ⇒ 不下结论，按「判死」处理（不投 9 小时）
辅助：高频子集 macroAUC（多分类 OVR 的随机基线 = 0.5，
   ⚠️ P107B2 脚本里写的 0.10/0.25 阈值**低于随机基线**，是错的，本脚本修正）

═══════════════════════════════════════════════════════════════
口径
═══════════════════════════════════════════════════════════════
与 P107B2（probe_b2_fixed.py）完全一致，才能与 d=+0.4119 对照：
   片段 = 按 gloss 数等分时间轴；vec = concat(段内 mean, 段内 std)
   Cohen's d = (mean(异类距离) − mean(同类距离)) / std(混合距离)
⚠️ 只测**前端判别力**，不回答 WER 会不会降。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
RGB = REPO / "artifacts/official_rgb/train"
CSV = REPO / "external/TFNet/data/CE-CSL/train.csv"   # Number, Translator, ..., Gloss
MIN_FREQ = 20
PROJ_DIM = 256

# ---- 判据（预先写死）----
D_ALIVE = 0.05      # Δd 至少 +0.05（Cohen's d 的随机基线是 0）
D_DEAD = 0.01
N_BOOT = 200
BOOT_PAIRS = 5000
POINT_PAIRS = 20000


def log(*a):
    print("[P124b %s] %s" % (time.strftime("%H:%M:%S"),
                             " ".join(str(x) for x in a)), flush=True)


# ---------------------------------------------------------------- 提取
def extract(ckpt: str, out_dir: Path, size: int, max_videos: int,
            batch: int, workers_hint: int = 0):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if not ckpt else None)
    net.fc = torch.nn.Identity()
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        res = net.load_state_dict(sd, strict=False)
        log("载入 %s（missing=%d unexpected=%d）"
            % (ckpt, len(res.missing_keys), len(res.unexpected_keys)))
        if len(res.missing_keys) > 4:
            log("⚠️ missing 过多，主干可能没换成功：%s" % res.missing_keys[:6])
    net = net.to(dev).eval()

    # ⚠️ Number 形如 "train-00001"（不是纯数字），首行是表头
    rows = list(csv.reader(open(CSV, newline="", encoding="utf-8")))
    rows = [r for r in rows if r and r[0] and r[0] != "Number"]
    if max_videos:
        rows = rows[:max_videos]
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(rows):
            sid, tr = r[0], r[1]
            d = RGB / tr / sid
            if not d.is_dir():
                continue
            fs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
            if not fs:
                continue
            imgs = np.stack([cv2.resize(
                cv2.cvtColor(cv2.imread(str(d / f)), cv2.COLOR_BGR2RGB),
                (size, size)) for f in fs]).astype(np.float32)
            imgs = imgs / 127.5 - 1.0                       # 官方归一化
            feats = []
            for s in range(0, len(imgs), batch):
                x = torch.from_numpy(imgs[s:s + batch]).permute(0, 3, 1, 2).to(dev)
                with torch.amp.autocast("cuda", enabled=True,
                                        dtype=torch.bfloat16):
                    f = net(x)
                feats.append(f.float().cpu().numpy())
            np.save(out_dir / ("%s.npy" % sid), np.concatenate(feats, 0))
            n_ok += 1
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                log("  %d/%d 视频（%.0fs, ETA %.0fs）"
                    % (i + 1, len(rows), el, el / (i + 1) * (len(rows) - i - 1)))
    log("提取完成：%d 视频 → %s（%.1f min）"
        % (n_ok, out_dir, (time.time() - t0) / 60))


# ---------------------------------------------------------------- 片段
def build_segments(feat_dir: Path, rows, min_frames: int = 1):
    X, y, grp = [], [], []
    for r in rows:
        sid = r[0]
        f = feat_dir / ("%s.npy" % sid)
        if not f.exists():
            continue
        a = np.load(f).astype(np.float32)
        if a.ndim != 2 or a.shape[0] < min_frames:
            continue
        g = r[3].split("/") if len(r) > 3 else r[-1].split("/")
        T = a.shape[0]
        if len(g) == 0 or T < len(g):
            continue
        for k, w in enumerate(g):
            s = int(k * T / len(g))
            e = max(s + 1, int((k + 1) * T / len(g)))
            seg = a[s:e]
            X.append(np.concatenate([seg.mean(0), seg.std(0)]))
            y.append(w)
            grp.append(sid)
    return (np.stack(X).astype(np.float32), np.asarray(y), np.asarray(grp))


def cohens_d(X, y, n_pairs, seed=0):
    """与 probe_b2_fixed.py 完全同口径的 Cohen's d。"""
    gmap = collections.defaultdict(list)
    for xx, yy in zip(X, y):
        gmap[yy].append(xx)
    usable = {k: v for k, v in gmap.items() if len(v) >= 2}
    ks = list(usable)
    if len(ks) < 2:
        return float("nan"), 0
    rs = np.random.RandomState(seed)
    ii = rs.choice(len(ks), n_pairs)
    jj = rs.choice(len(ks), n_pairs)
    m = ii != jj
    ii, jj = ii[m], jj[m]
    same = np.linalg.norm(np.stack([usable[ks[i]][0] for i in ii])
                          - np.stack([usable[ks[i]][1] for i in ii]), axis=1)
    diff = np.linalg.norm(np.stack([usable[ks[i]][0] for i in ii])
                          - np.stack([usable[ks[j]][0] for j in jj]), axis=1)
    allv = np.concatenate([same, diff])
    d = (diff.mean() - same.mean()) / (allv.std() + 1e-9)
    return float(d), len(ks)


def subset_auc(X, y, grp, seed=0):
    """高频子集 macroAUC；用**固定**随机投影降维（两组共用 ⇒ 可比）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    cnt = collections.Counter(y)
    keep = np.array([cnt[w] >= MIN_FREQ for w in y])
    if keep.sum() < 100:
        return float("nan"), 0, 0
    Xk, yk, gk = X[keep], y[keep], grp[keep]
    rs = np.random.RandomState(seed)
    P = rs.normal(size=(Xk.shape[1], PROJ_DIM)).astype(np.float32) / np.sqrt(PROJ_DIM)
    Xk = Xk @ P
    gv = sorted(set(gk))
    half = set(gv[: len(gv) // 2])
    tr = np.array([g in half for g in gk])
    te = ~tr
    cnt_tr = collections.Counter(yk[tr])
    cnt_te = collections.Counter(yk[te])
    valid = set(c for c in cnt_te if cnt_te[c] >= 2 and cnt_tr.get(c, 0) >= 1)
    m = np.array([c in valid for c in yk])
    Xk, yk, tr, te = Xk[m], yk[m], tr[m], te[m]
    if len(set(yk)) < 2:
        return float("nan"), 0, 0
    lb = LabelBinarizer().fit(yk)
    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    clf.fit(Xk[tr], yk[tr])
    Pp = clf.predict_proba(Xk[te])
    Yt = lb.transform(yk[te])
    if Yt.shape[1] == 1:
        # 二分类退化：roc_auc_score 要求 y_score 为 1d（正类概率）
        # （150 条视频的冒烟测试就会走到这里；全量 2000 条不会）
        Yt = Yt.ravel()
        Pp = Pp[:, 1] if Pp.shape[1] == 2 else Pp.ravel()
        auc = roc_auc_score(Yt, Pp)
    else:
        auc = roc_auc_score(Yt, Pp, average="macro", multi_class="ovr")
    return float(auc), int(len(set(yk))), int(len(yk))


# ---------------------------------------------------------------- compare
def compare(dir_a: Path, dir_b: Path, name_a: str, name_b: str, out: Path):
    rows = [r for r in csv.reader(open(CSV, newline="", encoding="utf-8"))
            if r and r[0] and r[0] != "Number"]
    Xa, ya, ga = build_segments(dir_a, rows)
    Xb, yb, gb = build_segments(dir_b, rows)
    # 对齐：两边视频集合可能不同（取交集，按 sid 顺序对齐 y/grp）
    log("片段 A=%d  B=%d" % (len(Xa), len(Xb)))
    common = [s for s in dict.fromkeys(ga) if s in set(gb)]
    ma = np.isin(ga, common)
    mb = np.isin(gb, common)
    Xa, ya, ga = Xa[ma], ya[ma], ga[ma]
    Xb, yb, gb = Xb[mb], yb[mb], gb[mb]
    # 保证两段顺序一致（都按 sid 排序）
    oa = np.argsort(ga, kind="stable")
    ob = np.argsort(gb, kind="stable")
    Xa, ya, ga = Xa[oa], ya[oa], ga[oa]
    Xb, yb, gb = Xb[ob], yb[ob], gb[ob]
    assert list(ga) == list(gb), "两边片段顺序不一致"
    assert list(ya) == list(yb), "两边标签不一致"
    log("对齐后片段=%d  维度 A=%d B=%d" % (len(Xa), Xa.shape[1], Xb.shape[1]))

    da, ka = cohens_d(Xa, ya, POINT_PAIRS)
    db, kb = cohens_d(Xb, yb, POINT_PAIRS)
    log("Cohen's d: %s=%+.4f (类%d)   %s=%+.4f (类%d)"
        % (name_a, da, ka, name_b, db, kb))

    # ---- bootstrap（按视频聚类重抽样）----
    vids = np.asarray(sorted(set(ga)))
    vid2idx = collections.defaultdict(list)
    for i, v in enumerate(ga):
        vid2idx[v].append(i)
    rs = np.random.RandomState(0)
    ds = []
    t0 = time.time()
    for b in range(N_BOOT):
        pick = rs.choice(len(vids), len(vids), replace=True)
        idx = np.concatenate([vid2idx[vids[p]] for p in pick])
        d_a, _ = cohens_d(Xa[idx], ya[idx], BOOT_PAIRS, seed=b)
        d_b, _ = cohens_d(Xb[idx], yb[idx], BOOT_PAIRS, seed=b)
        ds.append(d_b - d_a)
        if (b + 1) % 50 == 0:
            log("  bootstrap %d/%d（%.0fs）" % (b + 1, N_BOOT, time.time() - t0))
    ds = np.asarray(ds)
    delta = db - da
    lo, hi = np.percentile(ds, [2.5, 97.5])

    auca, na_ca, na_n = subset_auc(Xa, ya, ga)
    aucb, nb_cb, nb_n = subset_auc(Xb, yb, gb)
    log("高频子集 macroAUC: %s=%.4f (%d类/%d片段)  %s=%.4f (%d类/%d片段)"
        % (name_a, auca, na_ca, na_n, name_b, aucb, nb_cb, nb_n))

    if delta >= D_ALIVE and lo > 0:
        verdict = "判活"
    elif delta <= D_DEAD or hi < D_ALIVE:
        verdict = "判死"
    else:
        verdict = "灰区"
    log("Δd = %+.4f  95%%CI [%+.4f, %+.4f]  ⇒ %s" % (delta, lo, hi, verdict))

    rep = {
        "experiment": "P124b",
        "date": time.strftime("%Y-%m-%d"),
        "probe": "自监督主干 vs ImageNet 主干的帧级判别力",
        "criteria_preregistered": {
            "metric": "ΔCohen's d（主）+ 高频子集 macroAUC（辅助）",
            "random_baseline": "Cohen's d = 0；多分类 OVR macroAUC = 0.5",
            "alive": "Δd >= %.2f 且 bootstrap95%%CI 下界 > 0" % D_ALIVE,
            "dead": "Δd <= %.2f 或 CI 上界 < %.2f" % (D_DEAD, D_ALIVE),
            "note": "P107B2 脚本的 0.10/0.25 阈值低于随机基线 0.5，本脚本已修正",
        },
        "n_segments": int(len(Xa)),
        "n_videos": int(len(vids)),
        "A": {"name": name_a, "dir": str(dir_a), "cohens_d": round(da, 4),
              "usable_classes": ka, "subset_macro_auc": round(auca, 4),
              "subset_classes": na_ca, "subset_segments": na_n},
        "B": {"name": name_b, "dir": str(dir_b), "cohens_d": round(db, 4),
              "usable_classes": kb, "subset_macro_auc": round(aucb, 4),
              "subset_classes": nb_cb, "subset_segments": nb_n},
        "delta_d": round(float(delta), 4),
        "bootstrap": {"n": N_BOOT, "pairs_per_rep": BOOT_PAIRS,
                      "ci95": [round(float(lo), 4), round(float(hi), 4)],
                      "by_video_cluster": True},
        "verdict": verdict,
        "caveat": "Cohen's d 只测前端判别力，不等于 WER 会降；"
                  "判活也只是「值得花 9 小时做微调验证」，不是已证实",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log("收据 -> %s" % out)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--ckpt", default="", help="空 = ImageNet 基线")
    e.add_argument("--out", required=True)
    e.add_argument("--size", type=int, default=160)
    e.add_argument("--max-videos", type=int, default=0)
    e.add_argument("--batch", type=int, default=64)

    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--name-a", default="ImageNet")
    c.add_argument("--name-b", default="P124a-SimCLR")
    c.add_argument("--out", required=True)

    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.ckpt, Path(a.out), a.size, a.max_videos, a.batch)
    else:
        v = compare(Path(a.a), Path(a.b), a.name_a, a.name_b, Path(a.out))
        sys.exit(0 if v == "判活" else 1)


if __name__ == "__main__":
    main()
