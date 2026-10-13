"""P127 崩溃自愈守候（2026-10-13）。

为什么需要：P127 是单独启动的后台任务，**没有队列守护**。
而用户会更新 WorkBuddy / 办公占用 GPU ⇒ 都会导致进程被回收或 sticky error
（MEMORY 18 / 30）。没有自愈的话，11 h 的训练可能白跑。

机制（与 p124_queue.ensure_train 同思路，但只守一个任务）：
  - 每 60 s 轮询：收据出了 ⇒ 正常结束，退出
  - 进程没了但收据没出 ⇒ 等 90 s 让 GPU 释放 ⇒ `--resume last.pt` 重启
  - 最多 12 次（用户办公时约 1.5 h 可能崩一次，11 h 需 ~8 次余量）
  - 🔴 全程只可能有 **一个** 训练进程（守候本身不训练）

用法（后台任务模式启动，MEMORY 18）：
    ./venv/bin/python -u tools/blank_gov/p127_watch.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
TAG = "p127-res224"
RECEIPT = REPO / "artifacts/metrics/blank-gov" / ("%s-official-tfnet.json" % TAG)
LAST = REPO / "artifacts/checkpoints" / ("%s-last.pt" % TAG)
TRAIN = REPO / "tools/blank_gov/p78_train_official_tfnet.py"
PY = str(REPO / "venv/bin/python")
LOG = REPO / ("artifacts/logs_p127_res224.log")
WATCH_LOG = REPO / "artifacts/logs_p127_watch.log"

MAX_RESTART = 12
WAIT_GPU = 90


def log(*a):
    msg = "[WATCH %s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                             " ".join(str(x) for x in a))
    print(msg, flush=True)
    with open(WATCH_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def pids() -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args", "--no-headers"],
                         capture_output=True, text=True).stdout
    r = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        if "p78_train_official_tfnet.py" in parts[1] and "--tag %s" % TAG in parts[1]:
            try:
                r.append(int(parts[0]))
            except ValueError:
                pass
    return r


def alive() -> bool:
    return len(pids()) > 0


def cmd(resume: bool) -> list[str]:
    c = [PY, "-u", str(TRAIN), "--tag", TAG, "--module", "VAC",
         "--img-size", "224", "--amp-bf16",
         "--epochs", "20", "--total-epochs", "20",
         "--word-head", "0.0", "--mem-budget", "5.2e9", "--seed", "42"]
    if resume:
        c += ["--resume", str(LAST)]
    return c


def launch(resume: bool) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write("\n[WATCH %s] 重启（resume=%s）\n"
                 % (time.strftime("%m-%d %H:%M:%S"), resume))
        fh.flush()
        subprocess.Popen(cmd(resume), cwd=str(REPO), stdout=fh, stderr=fh)


def main() -> int:
    log("=== P127 守候启动（MAX_RESTART=%d）===" % MAX_RESTART)
    n = 0
    while True:
        if RECEIPT.exists():
            log("收据已出 ⇒ 正常结束")
            return 0
        if not alive():
            if n >= MAX_RESTART:
                log("⚠️ 已达重启上限 %d 次，停止守候" % MAX_RESTART)
                return 1
            n += 1
            has_ck = LAST.exists()
            log("进程不在（第 %d/%d 次）⇒ 等 %ds 让 GPU 释放"
                % (n, MAX_RESTART, WAIT_GPU))
            time.sleep(WAIT_GPU)
            if RECEIPT.exists():
                log("等待期间收据已出 ⇒ 结束")
                return 0
            if has_ck and LAST.stat().st_size < 1_000_000:
                # sticky error 写出的残file（MEMORY 22）
                LAST.unlink()
                log("⚠️ last.pt 过小，判定为损坏残file，已删除")
                has_ck = False
            launch(resume=has_ck)
            log("已重启（resume=%s）" % has_ck)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
