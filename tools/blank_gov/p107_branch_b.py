"""P107：岔路 B（帧级特征 → 大模型组句）自动衔接 —— 等 P101 出结果后启动。

🔴 为什么分两档：
   用户指示「等结果出来自动试试岔路 b」，但 P101 的结果决定**档位**：
   - P101 判死（词级辅助无效）
       ⇒ 说明瓶颈不在监督信号 ⇒ 换后端（岔路 B）有意义
       ⇒ 跑 B1 + B2 全流程
   - P101 判活（词级辅助有效）
       ⇒ 说明监督信号有用 ⇒ 应该继续在 CTC 路线上加投入
       ⇒ 只跑 B1（特征导出，可复用给任何后端），B2 降级为可选

   ⇒ B1 无论如何都跑（0.9 GB，1 小时，零风险，且是所有后端的地基）

⚠️ B2 的判据必须预先写死（防作弊，MEMORY 0.12）：
   判死条件：macroAUC ≤ 0.10（即与 P36 的 full368 = 0.041 同量级）
   判活条件：macroAUC ≥ 0.25（明显高于随机 0.5 的一半，有实用价值）
   灰区 0.10 < macroAUC < 0.25 ⇒ 报告数字，不下结论

坐标约定（与 p78 一致）：
   artifacts/official_rgb/{train,dev}/<字母>/<视频>/*.jpg  256×256
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = REPO / "venv/bin/python"
VENV = REPO / ".venv_mp/bin/python"
RGB = REPO / "artifacts/official_rgb"
OUT = REPO / "artifacts/frame_feats"
MET = REPO / "artifacts/metrics/blank-gov"

P101_TAGS = ["p101b-ctrl", "p101a-lowlr"]
# B2 判据（预先写死，见模块 docstring）
AUC_DEAD = 0.10
AUC_ALIVE = 0.25


def log(*a):
    print("[P107 %s] %s" % (time.strftime("%H:%M:%S"), " ".join(str(x) for x in a)),
          flush=True)


# ---------- 阶段 0：等 P101 ----------
def p101_done():
    """两组收据是否都存在。收据在 metrics/blank-gov/p101*.json。"""
    for t in P101_TAGS:
        cands = list(MET.glob(t + "*.json"))
        if not cands:
            return False, "缺 %s 收据" % t
    return True, "两组收据齐全"


def read_p101():
    """读 P101 结果，返回 {tag: best_wer}"""
    out = {}
    for t in P101_TAGS:
        for f in MET.glob(t + "*.json"):
            try:
                d = json.loads(f.read_text())
                out[t] = {"best_wer": d.get("best_wer_official"),
                          "history_len": len(d.get("history") or [])}
            except Exception as e:
                out[t] = {"error": str(e)}
    return out


# ---------- 阶段 1：B1 帧级特征导出 ----------
def export_frame_feats(split, max_videos=None):
    """用官方 resnet34MAM（fc=Identity）提取 512 维帧级特征。

    复用 p78 训练脚本里的模型构造，不重新实现（MEMORY 0.3 铁律）。
    """
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO / "external/TFNet"))
    import Module as TFModule  # noqa

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = TFModule.resnet34MAM()
    net.fc = TFModule.Identity()
    net.eval().to(dev)

    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((160, 160)), T.ToTensor(),
                    T.Normalize([.485, .456, .406], [.229, .224, .225])])

    split_dir = RGB / split
    vids = []
    for letter in sorted(os.listdir(split_dir)):
        for v in sorted(os.listdir(split_dir / letter)):
            vids.append((letter, v))
    if max_videos:
        vids = vids[:max_videos]
    log("导出 %s：%d 个视频" % (split, len(vids)))

    OUT.mkdir(parents=True, exist_ok=True)
    for i, (letter, v) in enumerate(vids):
        dst = OUT / split / (v + ".npy")
        if dst.exists():
            continue
        frames = sorted((split_dir / letter / v).glob("*.jpg"))
        if not frames:
            continue
        batch = []
        for fp in frames:
            try:
                batch.append(tf(Image.open(fp).convert("RGB")))
            except Exception:
                continue
        if not batch:
            continue
        with torch.no_grad():
            feats = []
            for s in range(0, len(batch), 16):
                b = torch.stack(batch[s:s + 16]).to(dev)
                # 逐帧推理：每帧当 T=1 的批
                # 输入布局 = [B, C, T, H, W]（Module.py:325 conv1 kernel=(1,7,7)）
                # MotorAttention kernel=(k,1,1) ⇒ 卷积核第 0 位是**时间**维，
                #   故单帧须给 T=1，batch 维放帧数 ⇒ [b, 3, 1, H, W]
                out = net(b.unsqueeze(2))            # [b,3,1,H,W]
                f = out[0] if isinstance(out, (list, tuple)) else out
                feats.append(f.float().cpu())        # [b, 512]
            f = torch.cat(feats).numpy().astype(np.float16)   # [T, 512]
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, f)
        if (i + 1) % 100 == 0:
            log("  %d/%d" % (i + 1, len(vids)))
    log("%s 导出完成 → %s" % (split, OUT / split))


# ---------- 阶段 2：B2 判别力探针 ----------
def probe(run_tag):
    """用导出的帧特征做「按 gloss 数等分片段池化 → 分类」测 macroAUC。

    🔴 必须用片段池化而非整句池化（MEMORY 0.6e，P19/P31/P34 三轮教训）。
    零号自检 = Cohen's d（不依赖分类器）。
    """
    import numpy as np

    rows = list(csv.DictReader(
        open(REPO / "data/raw/CE-CSL/label/train.csv", encoding="utf-8")))
    feats_dir = OUT / "train"
    if not feats_dir.exists():
        log("B2 跳过：%s 不存在" % feats_dir)
        return

    X, y, grp = [], [], []
    for r in rows:
        f = feats_dir / (r["Number"] + ".npy")
        if not f.exists():
            continue
        a = np.load(f).astype(np.float32)          # [T, 512]
        g = r["Gloss"].split("/")
        T_ = a.shape[0]
        for i, w in enumerate(g):                  # 按 gloss 数等分
            s = int(i * T_ / len(g))
            e = max(s + 1, int((i + 1) * T_ / len(g)))
            seg = a[s:e]
            X.append(np.concatenate([seg.mean(0), seg.std(0)]))
            y.append(w)
            grp.append(r["Number"])
    if not X:
        log("B2 跳过：无可用片段")
        return

    X = np.stack(X)
    log("片段池化完成：%d 片段 × %d 维" % X.shape)

    # 交 50% 验证（按视频分组，防泄漏）
    gv = sorted(set(grp))
    half = set(gv[:len(gv) // 2])
    tr = np.array([g in half for g in grp])
    te = ~tr

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer().fit(y)
    Y = lb.transform(y)                       # one-hot，仅用于 roc_auc_score
    y_arr = np.asarray(y)                     # 1d label，用于分类器 fit
    if Y.shape[1] < 2:
        log("B2 类别不足，跳过")
        return
    clf = LogisticRegression(max_iter=300, n_jobs=-1)
    clf.fit(X[tr], y_arr[tr])                 # sklearn 要求 y 为 1d 类标
    P = clf.predict_proba(X[te])
    auc = roc_auc_score(Y[te], P, average="macro", multi_class="ovr")

    # 零号自检：Cohen's d（同词 vs 异词）
    gmap = {}
    for i, (xx, yy) in enumerate(zip(X[te], y_arr[te])):
        gmap.setdefault(yy, []).append(xx)
    same, diff = [], []
    ks = list(gmap)
    rs = np.random.RandomState(0)
    for _ in range(20000):
        a, bq = rs.choice(len(ks), 2, replace=False)
        if ks[a] == ks[bq]:
            continue
        va = gmap[ks[a]]
        if len(va) < 2:
            continue
        same.append(np.linalg.norm(va[0] - va[1]))
        diff.append(np.linalg.norm(va[0] - gmap[ks[bq]][0]))
    d = (np.mean(diff) - np.mean(same)) / (np.std(same + diff) + 1e-9)

    verdict = ("判死" if auc <= AUC_DEAD else
               "判活" if auc >= AUC_ALIVE else "灰区")
    rep = {"n_segments": int(len(X)), "dim": int(X[1].shape[0]),
           "n_classes": int(Y.shape[1]),
           "macroAUC": round(float(auc), 4),
           "cohens_d": round(float(d), 4),
           "thresholds": {"dead": AUC_DEAD, "alive": AUC_ALIVE},
           "verdict": verdict,
           "compare": "P36 full368 macroAUC=0.0410 / upper pose+face=0.3041"}
    (MET / (run_tag + ".json")).write_text(
        json.dumps(rep, ensure_ascii=False, indent=2))
    log("B2 结果：macroAUC=%.4f  d=%+.4f  ⇒ %s" % (auc, d, verdict))


def main():
    log("=" * 70)
    log("P107：等 P101 → 自动启动岔路 B")
    log("=" * 70)

    # 阶段 0
    while True:
        done, why = p101_done()
        if done:
            break
        log("等 P101：%s" % why)
        time.sleep(600)

    r = read_p101()
    log("P101 结果：%s" % json.dumps(r, ensure_ascii=False))
    b, a = r.get("p101b-ctrl", {}), r.get("p101a-lowlr", {})
    alive = (isinstance(a.get("best_wer"), (int, float))
             and isinstance(b.get("best_wer"), (int, float))
             and a["best_wer"] < b["best_wer"])

    # 阶段 1：B1 无论如何都跑
    log("=" * 70)
    log("阶段 B1：导出帧级特征（全量，与 P101 结论无关）")
    log("=" * 70)
    try:
        export_frame_feats("dev")
        export_frame_feats("train")
    except Exception as e:
        log("❌ B1 失败：%s" % e)
        return 1

    # 阶段 2：B2
    log("=" * 70)
    log("阶段 B2：判别力探针（P101 %s）" % ("判活⇒降级为可选" if alive else "判死⇒全流程"))
    log("=" * 70)
    probe("p107b2-framefeat-probe")

    log("P107 全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
