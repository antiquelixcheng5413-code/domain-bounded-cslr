"""官方 TFNet 的 RGB 帧提取（严格照搬官方 CE-CSLDataPreProcess.py 的口径）。

官方原始逻辑（external/TFNet/CE-CSLDataPreProcess.py）：
    vid = imageio.get_reader(videoPath)
    nframes = vid.count_frames()
    for i in range(nframes):
        image = vid.get_data(i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        存成 00000.jpg / 00001.jpg ...
官方目录结构期望：{dataRoot}/{split}/{translator}/{sample_id}/{帧号}.jpg
    （由 MyDataset.__getitem__ 的 os.listdir(fn) 直接读帧目录确认）

与官方的两点差异（必须显式记录在收据里）：
  1. 官方用 imageio.v2，我们用 cv2.VideoCapture（逐帧解码）。
     帧数可能差 1~2 帧（编码器差异）。
  2. 官方 jpg 质量未指定（cv2.imencode 默认 95），我们显式沿用默认。

只处理 train + dev，**绝不碰 test**（test 冻结）。

用法：
    python tools/blank_gov/p77_extract_rgb.py --split dev --workers 6
    python tools/blank_gov/p77_extract_rgb.py --split train --workers 6
    python tools/blank_gov/p77_extract_rgb.py --split dev --limit 20# 小规模验证
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

_REPO = None
for _c in (Path(__file__).resolve().parents[2],):
    if (_c / "data" / "raw" / "CE-CSL").exists():
        _REPO = _c
        break
if _REPO is None:                       # WSL 下的兜底
    _REPO = Path("/home/su127/FYP/domain-bounded-cslr")
REPO = _REPO
sys.path.insert(0, str(REPO / "src"))

OUT_ROOT = REPO / "artifacts/official_rgb"
CSV_DIR = REPO / "data/raw/CE-CSL/label"
# dev 视频不在仓库里，实际位置（见 MEMORY 0.14）
DEV_VIDEO_FALLBACK = Path("/mnt/c/Users/su127/Desktop/csl视频/dev")

assert "test" not in str(OUT_ROOT), "test split 必须冻结"


def read_labels(split: str) -> dict[str, str]:
    p = CSV_DIR / ("%s.csv" % split)
    out = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["Number"]] = row["Translator"]
    return out


def find_video(split: str, sid: str, translator: str) -> Path | None:
    """定位 mp4。train 在仓库内；dev 要去桌面副本找。"""
    cands = []
    if split == "dev":
        cands.append(DEV_VIDEO_FALLBACK / translator / ("%s.mp4" % sid))
    else:
        cands.append(REPO / "data/raw/CE-CSL/video" / split / translator / ("%s.mp4" % sid))
    for c in cands:
        if c.exists():
            return c
    return None


def extract_one(job: tuple[str, str, str, str]) -> dict:
    split, sid, translator, out_root = job
    out_dir = Path(out_root) / split / translator / sid
    marker = out_dir / ".done"
    if marker.exists():
        return {"sid": sid, "status": "skip"}
    v = find_video(split, sid, translator)
    if v is None:
        return {"sid": sid, "status": "missing_video"}
    tmp = out_dir.with_name(out_dir.name + ".tmp%d" % os.getpid())
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            cap.release()
            return {"sid": sid, "status": "open_fail"}
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (256, 256))
            # 官方命名：5 位零填充（00000.jpg）
            name = "%05d.jpg" % n
            cv2.imencode(".jpg", img)[1].tofile(str(tmp / name))
            n += 1
        cap.release()
        if n == 0:
            return {"sid": sid, "status": "zero_frames"}
        # 原子替换：先写完再改名，避免半成品被下游读到
        if out_dir.exists():
            for f in out_dir.iterdir():
                f.unlink()
            out_dir.rmdir()
        tmp.rename(out_dir)
        marker = out_dir / ".done"
        marker.write_text("%d\n" % n, encoding="utf-8")
        return {"sid": sid, "status": "ok", "frames": n}
    except Exception as exc:                        # noqa: BLE001
        return {"sid": sid, "status": "error", "err": repr(exc)[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量")
    a = ap.parse_args()

    assert a.split != "test", "test split 冻结，不得提取"
    labels = read_labels(a.split)
    items = sorted(labels.items())
    if a.limit:
        items = items[:a.limit]
    out_root = str(OUT_ROOT)
    jobs = [(a.split, sid, tr, out_root) for sid, tr in items]
    print("待提取 %d 条（split=%s，**不含 test**）" % (len(jobs), a.split))
    print("输出根%s" % OUT_ROOT)

    stats = {"ok": 0, "skip": 0, "missing_video": 0, "zero_frames": 0,
             "open_fail": 0, "error": 0}
    frames_total = 0
    errs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(extract_one, j): j[1] for j in jobs}
        done = 0
        for fu in as_completed(futs):
            r = fu.result()
            st = r["status"]
            stats[st] = stats.get(st, 0) + 1
            if st == "ok":
                frames_total += r.get("frames", 0)
            elif st == "error":
                errs.append({"sid": r["sid"], "err": r.get("err")})
            done += 1
            if done % 200 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                print("  [%d/%d] ok=%d skip=%d  %.1f 条/s  已用 %.1f 分 ETA %.1f 分"
                      % (done, len(jobs), stats["ok"], stats["skip"],
                         rate, el / 60, eta / 60), flush=True)

    print()
    print("=" * 68)
    print("完成：%s" % stats)
    print("总帧数 = %d" % frames_total)
    rec = {
        "experiment": "P77",
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "官方 TFNet 的 RGB 帧提取（256x256 jpg，照官方 CE-CSLDataPreProcess.py）",
        "split": a.split,
        "test_split_excluded": True,
        "official_source": "external/TFNet/CE-CSLDataPreProcess.py",
        "deviations": [
            "官方用 imageio.v2 逐帧读，我们用 cv2.VideoCapture（帧数可能差 1~2）",
            "官方 jpg 质量未显式指定，我们沿用 cv2.imencode 默认",
        ],
        "total": len(jobs),
        "stat": stats,
        "frames_total": frames_total,
        "elapsed_s": round(time.time() - t0, 1),
        "errors": errs[:50],
    }
# ⚠️括号必须加：Path / str 的优先级高于 %，否则重演 P72 的 TypeError
    dst = REPO / ("artifacts/metrics/blank-gov/p77-extract-rgb-%s.json" % a.split)
    dst.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("收据 -> %s" % dst)


if __name__ == "__main__":
    main()