"""P136 串行链条：weight decay 调优（抗过拟合）。

🔴 为什么现在才做这个方向（重要反思）：
我们一直用 `--lr 1e-4 --weight-decay 1e-4`（官方值），**从未调过**。
而这两个值是官方为 **224 res + 55 轮**调的；我们跑的是 160 res + 20~48 轮。
分辨率与轮数都变了，最优超参很可能会变 —— 这是本项目的**系统性空白**。

同时我们已确认模型**过拟合**（P127：224 的 train CTC 更低但 dev 更差；
P123：train CTC 降到 ~5.5 而 dev 仍 55.84）⇒ 抗过拟合是最对症的方向。

本实验：weight decay 1e-4 → **1e-3**（10×），与 P124e 唯一变量。
P124e 配置：160 / bf16 / wh0 / 20 轮 / 无补帧 / seed 42 / wd 默认(1e-4)

判据（预先写死，ep13-20 与 P124e 配对）：
  mean ≤ −1.556 pp 且 SNR ≥ 2 且胜率 ≥ 6/8 ⇒ wd 增大有效
  |mean| < 1.556 pp                          ⇒ 不可判（换下一个超参）
  mean ≥ +1.556 pp                           ⇒ wd 增大有害（试更小 wd 或转 lr）

后续候选（本链条结束后按结果决定）：
  - lr 调优（5e-5 / 2e-4）—— 同样是零实现成本的空白
  - label smoothing（ref17 SALS、ref16 Sign2GPT 用 0.1）—— 需实现
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = str(REPO / "venv/bin/python")
METRICS = REPO / "artifacts/metrics/blank-gov"
CK = REPO / "artifacts/checkpoints"
CHAIN_LOG = REPO / "artifacts/logs_p136_chain.log"

MAX_RESTART = 12
WAIT_GPU = 90
TAG = "p136-wd1e3"
PAIR_BASE = "p124e-wh0-bf16"


def log(*a):
    msg = "[CHAIN %s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                             " ".join(str(x) for x in a))
    print(msg, flush=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def receipt(tag: str) -> Path:
    return METRICS / ("%s-official-tfnet.json" % tag)


def pids_for(tag: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args", "--no-headers"],
                         capture_output=True, text=True).stdout
    r = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        if "p78_train_official_tfnet.py" in parts[1] and ("--tag %s" % tag) in parts[1]:
            try:
                r.append(int(parts[0]))
            except ValueError:
                pass
    return r


def ensure_train() -> bool:
    base = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
            "--tag", TAG, "--module", "VAC", "--img-size", "160",
            "--amp-bf16", "--epochs", "20", "--total-epochs", "20",
            "--word-head", "0.0", "--weight-decay", "1e-3",
            "--mem-budget", "5.2e9", "--seed", "42"]
    last = CK / ("%s-last.pt" % TAG)
    n = 0
    while True:
        if receipt(TAG).exists():
            log("%s 收据已出 ⇒ 完成" % TAG)
            return True
        if not pids_for(TAG):
            if n >= MAX_RESTART:
                log("⚠️ 已达重启上限 %d，放弃" % MAX_RESTART)
                return False
            n += 1
            has_ck = last.exists()
            if has_ck and last.stat().st_size < 1_000_000:
                last.unlink()          # sticky error 残file（MEMORY 22）
                log("⚠️ last.pt 过小，判定损坏已删除")
                has_ck = False
            log("%s 进程不在（第 %d/%d 次）⇒ 等 %ds 后 resume=%s 重启"
                % (TAG, n, MAX_RESTART, WAIT_GPU, has_ck))
            time.sleep(WAIT_GPU)
            if receipt(TAG).exists():
                return True
            argv = base + (["--resume", str(last)] if has_ck else [])
            pf = REPO / "artifacts/logs_p136_wd1e3.log"
            with open(pf, "a", encoding="utf-8") as fh:
                fh.write("\n[CHAIN %s] 重启 resume=%s\n"
                         % (time.strftime("%m-%d %H:%M:%S"), has_ck))
                fh.flush()
                subprocess.Popen(argv, cwd=str(REPO), stdout=fh, stderr=fh)
        time.sleep(60)


def main() -> int:
    log("=== P136 串行链条启动（weight decay 调优，一次只跑一个）===")
    ok = ensure_train()
    if not ok:
        log("❌ %s 未拿到收据 ⇒ 链条中止" % TAG)
        return 1
    subprocess.call(
        [PY, "tools/blank_gov/pair_analyze.py",
         "--a", TAG, "--b", PAIR_BASE, "--lo", "13", "--hi", "20",
         "--note", "weight decay 1e-4 → 1e-3 的净效应（抗过拟合）"],
        cwd=str(REPO))
    log("=== 链条全部结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
