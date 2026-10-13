"""P130 串行链条（用户要求「跑完自动跑下一个」）。

    P129 裁切位置探针（~12 min）
        │
        ├─ 判活（最优 shift 改善 ≥ 1.0 pp）⇒ P130：训练采纳该测试裁切位置
        └─ 不判活                        ⇒ P131：去掉 TemporalRescale(0.2)
                                              （官方未启用，我们长期多带）
        │
        └─ 完成后自动 epoch 级配对（与 P124e，ep13-20）

🔴 全程**一次只跑一个**（用户明确要求，MEMORY 11.-1）。
🔴 每个训练任务带崩溃自愈：进程没了但收据没出 ⇒ 等 90s 让 GPU 释放 ⇒ --resume 重启。

判据全部预先写死，不在看到数字后临时改口。
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
CHAIN_LOG = REPO / "artifacts/logs_p130_chain.log"

MAX_RESTART = 12
WAIT_GPU = 90
PAIR_BASE = "p124e-wh0-bf16"      # 对照：160 / bf16 / wh0 / 20轮 / 无补帧 / seed42


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


def run_block(name: str, argv: list[str], logfile: str) -> int:
    p = REPO / logfile
    log("启动 %s（日志 %s）" % (name, logfile))
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("\n[CHAIN %s] %s\n" % (time.strftime("%m-%d %H:%M:%S"), name))
        fh.flush()
        rc = subprocess.call(argv, cwd=str(REPO), stdout=fh, stderr=fh)
    log("%s 结束 rc=%d" % (name, rc))
    return rc


def ensure_train(tag: str, extra: list[str], logfile: str) -> bool:
    """等收据；崩溃 ⇒ resume 重启（最多 MAX_RESTART 次）。返回是否拿到收据。"""
    base = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
            "--tag", tag, "--module", "VAC", "--img-size", "160",
            "--amp-bf16", "--epochs", "20", "--total-epochs", "20",
            "--word-head", "0.0", "--mem-budget", "5.2e9", "--seed", "42"]
    last = CK / ("%s-last.pt" % tag)
    n = 0
    while True:
        if receipt(tag).exists():
            log("%s 收据已出 ⇒ 完成" % tag)
            return True
        if not pids_for(tag):
            if n >= MAX_RESTART:
                log("⚠️ %s 已达重启上限 %d，放弃" % (tag, MAX_RESTART))
                return False
            n += 1
            has_ck = last.exists()
            if has_ck and last.stat().st_size < 1_000_000:
                last.unlink()          # sticky error 残file（MEMORY 22）
                log("⚠️ %s 过小，判定损坏已删除" % last.name)
                has_ck = False
            log("%s 进程不在（第 %d/%d 次）⇒ 等 %ds 后 resume=%s 重启"
                % (tag, n, MAX_RESTART, WAIT_GPU, has_ck))
            time.sleep(WAIT_GPU)
            if receipt(tag).exists():
                return True
            argv = base + (["--resume", str(last)] if has_ck else []) + extra
            pf = REPO / logfile
            with open(pf, "a", encoding="utf-8") as fh:
                fh.write("\n[CHAIN %s] 重启 resume=%s\n"
                         % (time.strftime("%m-%d %H:%M:%S"), has_ck))
                fh.flush()
                subprocess.Popen(argv, cwd=str(REPO), stdout=fh, stderr=fh)
        time.sleep(60)


def main() -> int:
    log("=== P130 串行链条启动（一次只跑一个）===")

    # ---------- 1. P129 裁切位置探针 ----------
    p129 = METRICS / "p129-cropshift-probe.json"
    if p129.exists():
        log("P129 收据已存在 ⇒ 跳过")
    else:
        run_block("P129", [PY, "-u", "tools/blank_gov/p129_cropshift_probe.py", "515"],
                  "logs_p129_cropshift.log")

    if not p129.exists():
        log("❌ P129 无收据 ⇒ 链条中止")
        return 1
    d = json.load(open(p129))
    best_s = int(d["best_shift"])
    gain = float(d["gain_pp"])
    log("P129 结论：最优 shift=%+d，改善 %.2f pp" % (best_s, gain))
    log("  %s" % d["verdict"])
    log("  已知答案校验（shift=0 应=55.84）：%s"
        % ("通过" if d.get("known_answer_check_passed") else "❌ 未通过"))

    # ---------- 2. 按预先写死的判据选下一个训练 ----------
    if gain >= 1.0:
        tag, extra = "p130-cropshift%+d" % best_s, ["--test-crop-shift", str(best_s)]
        log("⇒ 判活：启动 P130（测试裁切 y_shift=%+d），与 %s 配对" % (best_s, PAIR_BASE))
    else:
        tag, extra = "p131-norescale", ["--no-temporal-rescale"]
        log("⇒ 不判活：启动 P131（去掉 TemporalRescale），与 %s 配对" % PAIR_BASE)

    ok = ensure_train(tag, extra, "logs_%s.log" % tag.replace("-", "_"))
    if not ok:
        log("❌ %s 未拿到收据 ⇒ 链条中止" % tag)
        return 1

    # ---------- 3. 自动配对 ----------
    note = ("裁切位置 y_shift=%+d 净效应" % best_s) if tag.startswith("p130") \
        else "关掉 TemporalRescale(0.2) 的净效应"
    run_block("配对 %s vs %s" % (tag, PAIR_BASE),
              [PY, "tools/blank_gov/pair_analyze.py",
               "--a", tag, "--b", PAIR_BASE, "--lo", "13", "--hi", "20",
               "--note", note],
              "logs_pair_%s.log" % tag.replace("-", "_"))

    log("=== 链条全部结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
