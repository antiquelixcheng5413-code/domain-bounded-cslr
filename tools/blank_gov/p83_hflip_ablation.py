"""P83 自动串联：等 P82 两组跑完，自动出对比收据。

要回答的问题：**中文手语的左右镜像手势语义是否相反？**
官方默认 RandomHorizontalFlip(0.5)，但对中文手语可能是有害的。
P82 已在跑 no-flip 组，本脚本等它结束后自动跑 hflip 组。

⚠️ 串联用「轮询 +进程名匹配」，不用 nohup（`wsl -e` 下 nohup 会随shell 退出被杀，
   这个坑我踩过 2 次，见 MEMORY 0.12）。
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = REPO / "venv/bin/python"
SCRIPT = REPO / "tools/blank_gov/p78_train_official_tfnet.py"
METRICS = REPO / "artifacts/metrics/blank-gov"

JOBS = [
    # (tag, extra_args, 中文标签)
    ("p82-vac-noflip", ["--no-hflip"], "关闭镜像翻转（对照组）"),
    ("p82-vac-hflip", [], "开启镜像翻转（官方默认 p=0.5）"),
]
EPOCHS = 6
MAX_TRAIN = 600
MODULE = "VAC"


def is_running() -> bool:
    r = subprocess.run(["bash", "-lc",
                        "ps -eo cmd | grep -c '[p]78_train_official'"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip()) > 0
    except ValueError:
        return False


def wait_for(tag: str, timeout_s: int) -> bool:
    """等某个 tag 的收据出现。"""
    rec = METRICS / ("%s-official-tfnet.json" % tag)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if rec.exists():
            return True
        time.sleep(30)
    return False


def run_one(tag: str, extra: list[str]) -> dict:
    rec = METRICS / ("%s-official-tfnet.json" % tag)
    if rec.exists():
        print("[skip] %s 已有收据" % tag, flush=True)
        return json.loads(rec.read_text(encoding="utf-8"))
    cmd = [str(PY), "-u", str(SCRIPT), "--module", MODULE,
           "--epochs", str(EPOCHS), "--max-train", str(MAX_TRAIN),
           "--tag", tag] + extra
    print("[run ] %s" % " ".join(cmd[2:]), flush=True)
    log = "/tmp/%s.log" % tag
    with open(log, "w") as fh:
        r = subprocess.run(cmd, cwd=str(REPO), stdout=fh,
                           stderr=subprocess.STDOUT, timeout=6 * 3600)
    if rec.exists():
        return json.loads(rec.read_text(encoding="utf-8"))
    return {"error": "no receipt, exit=%s, see %s" % (r.returncode, log)}


def main() -> None:
    print("=" * 70)
    print("P83 自动串联：hflip 开关对比（VAC, %d 条, %d epoch）"
          % (MAX_TRAIN, EPOCHS))
    print("=" * 70)

    # 1) 等当前在跑的 no-flip 组
    if is_running():
        print("检测到 P82 no-flip 组在跑，等待其结束…", flush=True)
        wait_for("p82-vac-noflip", 4 * 3600)
        # 等进程真正退出
        t0 = time.time()
        while is_running() and time.time() - t0 < 600:
            time.sleep(10)

    results = {}
    for tag, extra, label in JOBS:
        print()
        print("→ %s （%s）" % (tag, label), flush=True)
        results[tag] = run_one(tag, extra)
        w = results[tag].get("best_wer_official")
        if w is not None:
            print("   %s: WER_official = %.2f%%" % (tag, w), flush=True)

    # 2) 出对比收据
    a = results.get("p82-vac-noflip", {})
    b = results.get("p82-vac-hflip", {})
    wa, wb = a.get("best_wer_official"), b.get("best_wer_official")

    # 🔴🔴 判据铁律（P83 首版栽过）：**绝不只比 best 单点**。
    # 必须做逐 epoch 配对分析：① 每 epoch 差 ② 均值±std ③ 胜率
    #     ④ 符号是否翻转 —— 任一不稳定即判「噪声范围内，不可下结论」。
    # 理由：P39 教训 + 本次实测反例（首版按 best 判「hflip 更差」，
    #     逐 epoch 分析却显示 hflip 略好，符号在 ep1 翻转）。
    import statistics
    diffs, pairs = [], []
    if a.get("history") and b.get("history"):
        ha = {h["epoch"]: h["dev_wer_official"] for h in a["history"]}
        hb = {h["epoch"]: h["dev_wer_official"] for h in b["history"]}
        common = sorted(set(ha) & set(hb))
        # diff = noflip - hflip（正 = noflip 更好 = hflip 有害）
        diffs = [ha[e] - hb[e] for e in common]
        pairs = [(e, ha[e], hb[e], ha[e] - hb[e]) for e in common]

    verdict = "数据不足"
    stats = {}
    if diffs:
        m = statistics.fmean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        win = sum(1 for d in diffs if d > 0)
        sign_flip = (any(d > 0 for d in diffs) and any(d < 0 for d in diffs))
        stats = {"mean_diff_pp": round(m, 2), "std_pp": round(sd, 2),
                 "win_rate": "%d/%d" % (win, len(diffs)),
                 "sign_flip": sign_flip,
                 "per_epoch": [{"epoch": e, "noflip": x, "hflip": y,
                                "diff": round(z, 2)} for e, x, y, z in pairs]}
        if sign_flip or abs(m) < 2 * sd:
            verdict = ("差异 %.2f ± %.2f pp，符号%s ⇒ **噪声范围内，"
                       "判定「hflip 开关在本实验无显著影响」**，"
                       "不足以判定中文手语镜像语义是否相反"
                       % (m, sd, "不稳定" if sign_flip else "稳定"))
        elif m > 0:
            verdict = ("noflip 持续更优 %.2f ± %.2f pp ⇒ "
                       "**hflip 有害，中文手语镜像语义可能相反，建议关闭**"
                       % (m, sd))
        else:
            verdict = ("hflip 持续更优 %.2f ± %.2f pp ⇒ "
                       "镜像翻转对本数据有帮助，官方默认正确" % (-m, sd))

    out = {
        "experiment": "P83",
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "RandomHorizontalFlip 开关对比 —— 验证中文手语镜像语义是否相反",
        "module": MODULE,
        "official_benchmark_dev": a.get("official_benchmark_dev"),
        "config": {"epochs": EPOCHS, "max_train": MAX_TRAIN,
                   "note": "两组唯一差异 = RandomHorizontalFlip"},
        "noflip_wer": wa,
        "hflip_wer": wb,
        "delta_best_pp": (None if wa is None or wb is None
                          else round(wb - wa, 2)),
        "paired_analysis": stats,
        "verdict": verdict,
        "_caveat": "⚠️ 600 条 / 6 epoch 规模。仅 best 单点比较不可靠，"
                   "本判据已改为逐 epoch 配对分析（P39/P83 教训）。",
        "full": results,
    }
    dst = METRICS / "p83-hflip-ablation.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print()
    print("=" * 70)
    for e, x, y, z in pairs:
        print("  ep%03d  noflip %6.2f%%  hflip %6.2f%%  diff %+6.2f pp"
              % (e, x, y, z))
    if stats:
        print("  平均 %+.2f ± %.2f pp  胜率 %s  符号翻转 %s"
              % (stats["mean_diff_pp"], stats["std_pp"],
                 stats["win_rate"], stats["sign_flip"]))
    print("=" * 70)
    print("结论：%s" % verdict)
    print("收据 -> %s" % dst)


if __name__ == "__main__":
    main()