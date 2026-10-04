# -*- coding: utf-8 -*-
"""P4 探针：在**原生帧率**上重提 landmark，比较 TFD 边界质量 vs 48 帧版本。

动机（本项目三次实验 P2-a/P2-b/P3 之后的结论）：
  blank 率与帧级监督都卡在「伪边界质量」上。现有特征全部是
  `frame_sampling: "resample-48"`（5487 条无一例外），
  而原视频是 30fps、87~333 帧（中位约 170）。
  **TFD 从未在原生帧率上跑过。** 帧率高 4.9 倍意味着 TFD 的
  局部时序窗口能看到更细的运动变化，边界定位可能显著更好。

本脚本只做一件事：验证「帧率是不是边界质量的瓶颈」。
- 若原生帧的 TFD 边界明显更密/更准 → 帧率是瓶颈，值得继续投入
- 若无差异 → 立刻停止帧级这条线，转特征侧

严格约束：
- 复用仓库既有 `MediaPipeHolisticExtractor`（只改 sequence_length），
  **不改仓库任何代码**，保证除帧率外口径完全一致
- 用 `/tmp/mp_venv`（Python 3.12 + mediapipe==0.10.21，与 requirements 一致）
- 只抽 dev 的一小批（约 40 条），不跑全量
- **不触碰 test**
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import tfd as TFD


def video_split_dir(split):
    """视频目录用 dev/train/test，特征目录用 validation/train。

    仓库里两套命名不一致：data/raw/CE-CSL/video/{train,dev,test}
    vs artifacts/part3_features/{train,validation}。
    这里做一次映射，避免调用方传错。
    """
    return {"validation": "dev", "dev": "dev", "train": "train", "test": "test"}[split]


def find_video(split, sid):
    base = REPO / "data/raw/CE-CSL/video" / video_split_dir(split)
    for d in sorted(base.iterdir()):
        p = d / (sid + ".mp4")
        if p.exists():
            return p
    return None


def read_gloss(split):
    csv_path = REPO / "data/raw/CE-CSL/label" / ("dev.csv" if split == "validation" else "train.csv")
    out = {}
    import csv as _csv

    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            out[r["Number"]] = r["Gloss"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p4-native-fps-probe.json")
    ap.add_argument("--cache", default="/tmp/p4_native_feats")
    a = ap.parse_args()

    from cslr.features.extractor import MediaPipeHolisticExtractor

    gloss = read_gloss(a.split)
    cache = Path(a.cache) / video_split_dir(a.split)
    cache.mkdir(parents=True, exist_ok=True)

    # 只取前 limit 条有视频的 dev 样本
    sids = []
    base = REPO / "data/raw/CE-CSL/video" / video_split_dir(a.split)
    for d in sorted(base.iterdir()):
        for v in sorted(d.iterdir()):
            sid = v.stem
            if sid in gloss and sid not in sids:
                sids.append(sid)
            if len(sids) >= a.limit:
                break
        if len(sids) >= a.limit:
            break
    print("探针样本 {} 条（split={}）".format(len(sids), a.split))

    rows = []
    t0 = time.time()
    todo = [s for s in sids if not (cache / (s + ".npy")).exists()]
    print("待提取 {} 条（已缓存 {} 条）".format(len(todo), len(sids) - len(todo)))

    for i, sid in enumerate(todo):
        vid = find_video(a.split, sid)
        if vid is None:
            continue
        n_gloss = len([t for t in gloss[sid].split("/") if t.strip()])
        if n_gloss == 0:
            continue
        # 关键：sequence_length=None 时不重采样，保留原生帧序列。
        # extractor._resample 在 sequence_length 为 None 时直接返回原序列，
        # 所以这里先看它对 None 的处理；若不支持则用极大值代替。
        try:
            ext = MediaPipeHolisticExtractor(sequence_length=10**9)
            res = ext.extract(vid)
            native = res.features
        except Exception as e:
            print("  [{}] {} 提取失败: {}".format(i, sid, e))
            continue
        if native.shape[0] == 0:
            print("  [{}] {} 空序列".format(i, sid))
            continue
        np.save(cache / (sid + ".npy"), native.astype(np.float32))
        rows.append({"sid": sid, "native_frames": int(native.shape[0]),
                     "n_gloss": n_gloss, "source_frames": int(res.source_frames)})
        if (i + 1) % 10 == 0:
            print("  {}/{} ({:.1f} 分钟)".format(i + 1, len(todo), (time.time() - t0) / 60),
                  flush=True)

    print("原生帧率提取完成：{} 条，耗时 {:.1f} 分钟".format(
        len(rows), (time.time() - t0) / 60))

    # ---------- 对比：48 帧 vs 原生帧的 TFD 边界 ----------
    per = []
    for r in rows:
        sid = r["sid"]
        f_native = np.load(cache / (sid + ".npy")).astype(np.float64)
        f48_path = REPO / "artifacts/part3_features" / a.split / (sid + ".landmark.npy")
        if not f48_path.exists():
            continue
        f48 = np.load(f48_path).astype(np.float64)
        ng = r["n_gloss"]

        # T 参数按各自序列长度重估（论文 Sec 4.4.9：T ≈ 平均每 gloss 帧长 60-90%）
        t48, m48 = TFD.suggest_TM(f48.shape[0], ng)
        tn, mn = TFD.suggest_TM(f_native.shape[0], ng)
        pk48 = TFD.detect_boundaries(f48, t=t48, m=m48, metric="l2")
        pkn = TFD.detect_boundaries(f_native, t=tn, m=mn, metric="l2")

        # 边界信号强度：越锐利越可信
        sig48 = TFD.tfd_signal(f48, t=t48)
        sign = TFD.tfd_signal(f_native, t=tn)

        per.append({
            "sid": sid,
            "n_gloss": ng,
            "T48": f48.shape[0], "Tn": int(f_native.shape[0]),
            "ratio": round(f_native.shape[0] / f48.shape[0], 2),
            "t48": t48, "tn": tn,
            "pk48": int(pk48.size), "pkn": int(pkn.size),
            "ratio48": round(pk48.size / ng, 3),
            "ration": round(pkn.size / ng, 3),
            # 边界锐度：TFD 峰值的 z-score，越大越有信息量
            "sharp48": round(float(sig48.max() / (sig48.std() + 1e-9)), 2),
            "sharpn": round(float(sign.max() / (sign.std() + 1e-9)), 2),
        })

    if not per:
        print("无可对比样本")
        return

    r48 = np.array([x["ratio48"] for x in per])
    rn = np.array([x["ration"] for x in per])
    s48 = np.array([x["sharp48"] for x in per])
    sn = np.array([x["sharpn"] for x in per])

    print("")
    print("=" * 68)
    print("P4 探针：原生帧率 vs 48 帧 的 TFD 边界质量")
    print("=" * 68)
    print("样本 {} 条，原生帧数中位 {}（约为 48 帧的 {:.1f} 倍）".format(
        len(per), int(np.median([x["Tn"] for x in per])),
        float(np.median([x["ratio"] for x in per]))))
    print("")
    print("峰数/gloss 比   48帧 {:.2f}   原生 {:.2f}   (论文欠分割偏置目标 <1)".format(
        np.median(r48), np.median(rn)))
    print("TFD 峰锐度     48帧 {:.2f}   原生 {:.2f}   (z-score，越大越锐)".format(
        np.median(s48), np.median(sn)))
    print("")
    # 关键判据：原生帧的峰数是否更接近 gloss 数
    better = int((rn > r48).sum())
    print("原生帧峰数更多的样本  {}/{}".format(better, len(per)))
    # 锐度提升
    print("原生帧锐度更高的样本  {}/{}".format(int((sn > s48).sum()), len(per)))
    print("")
    # 与真实 gloss 数的差距（欠分割程度，越接近 1 越好）
    print("峰数/gloss 接近 1 的样本占比   48帧 {:.1%}   原生 {:.1%}".format(
        float((np.abs(r48 - 1) < 0.3).mean()),
        float((np.abs(rn - 1) < 0.3).mean())))

    print("")
    print("=== 判据 ===")
    gain_n = np.median(rn) - np.median(r48)
    gain_s = np.median(sn) / max(np.median(s48), 1e-9)
    if gain_n >= 0.2 and gain_s >= 1.2:
        verdict = "帧率是瓶颈：原生帧边界明显更好，值得继续投入帧级路线"
    elif gain_n >= 0.1 or gain_s >= 1.1:
        verdict = "原生帧略好但改善有限，收益不足以支撑继续投入"
    else:
        verdict = "帧率不是瓶颈：原生帧无明显改善，应停止帧级路线，转特征侧"
    print(verdict)

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "P4 TFD boundary quality: native fps vs resample-48",
        "extractor": "MediaPipeHolisticExtractor(sequence_length=1e9) — same code path as repo",
        "mediapipe": "0.10.21 (matches requirements.txt)",
        "n_samples": len(per),
        "peak_per_gloss_median_48": float(np.median(r48)),
        "peak_per_gloss_median_native": float(np.median(rn)),
        "tfd_sharpness_median_48": float(np.median(s48)),
        "tfd_sharpness_median_native": float(np.median(sn)),
        "native_frames_median": int(np.median([x["Tn"] for x in per])),
        "verdict": verdict,
        "per_sample": per,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
