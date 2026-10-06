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
    verdict = "数据不足"
    if wa is not None and wb is not None:
        d = wb - wa
        if abs(d) < 0.5:
            verdict = "两组基本无差异，镜像翻转在本任务上中性"
        elif wb > wa:
            verdict = ("开启 hflip 后 WER 恶化 %.2f pp⇒ "
                       "**中文手语镜像语义确实相反，应关闭**" % d)
        else:
            verdict = ("开启 hflip 后 WER 改善 %.2f pp ⇒ "
                       "镜像翻转对本数据有帮助，官方默认正确" % -d)

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
        "delta_pp": (None if wa is None or wb is None else round(wb - wa, 2)),
        "verdict": verdict,
        "full": results,
    }
    dst = METRICS / "p83-hflip-ablation.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print()
    print("=" * 70)
    print("结论：%s" % verdict)
    print("收据 -> %s" % dst)


if __name__ == "__main__":
    main()