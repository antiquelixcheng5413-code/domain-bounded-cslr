#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""判别力来源的自动串行队列（一次只跑一个 GPU 任务）

队列（P150 之前两度被中断/静默死亡 ⇒ 并入队列第一步，并用 -u 让日志实时落盘）
  0. P150  VAC + 160 + RGB（通道修正的唯一变量实验）      ~24h
  A. P156  官方权重消融：量 分辨率/通道 各值多少 pp（零训练） ~16min
  B. P157  VAC + 224 + RGB（通道修正基础上测分辨率）        ~24h
  C. P158  TFNet + 224 + RGB（测容量）                    ~41h

铁律
  - 一次只跑一个
  - 进度写 artifacts/metrics/blank-gov/p156-chain-progress.json，崩溃后可续跑
  - 每步用 python -u（无缓冲）⇒ 进程被杀也能看到它跑到哪
  - 若日志含 out of memory ⇒ 自动降 batch=1 重试一次

用法
  ./venv/bin/python -u tools/blank_gov/p156_chain.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = str(REPO / "venv/bin/python")
PROG = REPO / "artifacts/metrics/blank-gov/p156-chain-progress.json"

P78 = "tools/blank_gov/p78_train_official_tfnet.py"
BASE_ARGS = ["--amp-bf16", "--epochs", "55", "--total-epochs", "55",
             "--lr", "1e-4", "--weight-decay", "1e-4", "--hidden", "1024",
             "--seed", "42", "--official-pad", "--word-head", "0.3",
             "--word-boundary", "ctc_align", "--warmup-ep", "3",
             "--workers", "4", "--fix-rgb-order"]

STEPS = [
    {"name": "P150_vac160_rgb",
     "cmd": [PY, "-u", P78, "--tag", "p150-rgbfix", "--module", "VAC",
             "--img-size", "160", "--batch", "2", "--mem-budget", "5.2e9"]
            + BASE_ARGS},
    {"name": "A_official_ablation",
     "cmd": [PY, "-u", "tools/blank_gov/p156_official_ablation.py", "0", "fast"]},
    {"name": "B_vac224_rgb",
     "cmd": [PY, "-u", P78, "--tag", "p157-vac224-rgb", "--module", "VAC",
             "--img-size", "224", "--batch", "2", "--mem-budget", "5.2e9"]
            + BASE_ARGS},
    {"name": "C_tfnet224_rgb",
     "cmd": [PY, "-u", P78, "--tag", "p158-tfnet224-rgb", "--module", "TFNet",
             "--img-size", "224", "--batch", "1", "--mem-budget", "6.5e9"]
            + BASE_ARGS},
]


def log(*a):
    print("[chain %s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                             " ".join(str(x) for x in a)), flush=True)


def alive(tag):
    r = subprocess.run(["ps", "-eo", "args", "--no-headers"],
                       capture_output=True, text=True)
    return any(tag in ln and "python" in ln for ln in r.stdout.splitlines())


def load_progress():
    if PROG.exists():
        try:
            return json.loads(PROG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"done": []}


def save_progress(p):
    PROG.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")


def run_step(step):
    log("──── 开始 %s ────" % step["name"])
    logfile = REPO / ("artifacts/logs_%s.log" % step["name"])
    with open(logfile, "w", encoding="utf-8") as f:
        rc = subprocess.call(step["cmd"], cwd=str(REPO),
                             stdout=f, stderr=subprocess.STDOUT)
    if rc == 0:
        log("✅ %s 完成（rc=0）" % step["name"])
        return True
    txt = logfile.read_text(encoding="utf-8", errors="ignore")
    if "out of memory" in txt.lower():
        log("⚠️ %s 疑似 OOM ⇒ 降 batch=1 重试" % step["name"])
        cmd = list(step["cmd"])
        if "--batch" in cmd:
            cmd[cmd.index("--batch") + 1] = "1"
            with open(logfile, "a", encoding="utf-8") as f:
                f.write("\n\n===== 降 batch=1 重试 =====\n")
                rc2 = subprocess.call(cmd, cwd=str(REPO),
                                      stdout=f, stderr=subprocess.STDOUT)
            if rc2 == 0:
                log("✅ %s 重试完成" % step["name"])
                return True
    log("❌ %s 失败 rc=%s，日志 %s" % (step["name"], rc, logfile.name))
    return False


def main() -> int:
    log("链条启动：%d 步" % len(STEPS))
    if alive("p150-rgbfix") and "P150_vac160_rgb" not in load_progress().get("done", []):
        log("检测到已有 p150-rgbfix 在跑 ⇒ 等它结束再继续")
        while alive("p150-rgbfix"):
            time.sleep(300)
    prog = load_progress()
    for st in STEPS:
        if st["name"] in prog.get("done", []):
            log("跳过（已完成）%s" % st["name"])
            continue
        if run_step(st):
            prog.setdefault("done", []).append(st["name"])
            save_progress(prog)
        else:
            prog["failed_at"] = st["name"]
            save_progress(prog)
            log("链条在 %s 处停止" % st["name"])
            return 1
    log("🎉 队列全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
