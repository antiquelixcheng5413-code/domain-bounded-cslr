#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P144：换 seed 重跑主配置（回答 H-a / H-b 悬案）

目的（**不是**为了降 WER，而是为了测量方差）
  P140/P141/P142 三个不同改动都给出 +1.27~+2.03 pp 的相近负向结果。
  两种解释无法用现有数据分开：
    H-a：这三个改动真的都有害
    H-b：P123 的 55.84 含运气成分，任何扰动都会打破它
  ⇒ 换 seed 重跑**同配置**，看 55.84 能复现到什么程度：
     差 < 1.556 pp（噪声）⇒ 支持 H-a（基线稳定，改动确实有害）
     差 > 1.556 pp        ⇒ 支持 H-b（单次 run 方差大，此前结论都要打折）

🔴 本实验**禁用早停止损**
  若 seed 1234 更差，那正是要测的信号，提前杀掉就拿不到完整曲线。
  只保留崩溃自愈（resume 重启）+ 残file 校验。

配置 = P123 全部（lr 1e-4 + 里程碑[35,45] + 官方补帧 + wh0.3 + warmup3
       + hflip + bf16 + 160res），唯一变量 = seed 42 → 1234
评估 bias=0（与 P123 严格配对）；bias 收益事后用 P134 扫（零成本）
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = str(REPO / "venv/bin/python")
METRICS = REPO / "artifacts/metrics/blank-gov"
CK = REPO / "artifacts/checkpoints"
CHAIN_LOG = REPO / "artifacts/logs_p144_chain.log"   # 显式写死

MAX_RESTART = 12
WAIT_GPU = 90
POLL = 60

TAG = "p144-seed1234"
BASE = "p123-official55bf16"
PLANNED = 40
SEED = "1234"


def log(*a):
    msg = "[CHAIN %s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                             " ".join(str(x) for x in a))
    print(msg, flush=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def receipt(tag):
    return METRICS / ("%s-official-tfnet.json" % tag)


def pids_for(tag):
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


def main():
    log("=== P144 启动：换 seed 重跑主配置（seed 42 → %s）===" % SEED)
    log("🔴 本实验禁用早停止损（更差正是要测的信号），只保留崩溃自愈")
    last = CK / ("%s-last.pt" % TAG)
    n = 0
    base_argv = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                 "--tag", TAG, "--module", "VAC",
                 "--img-size", "160", "--amp-bf16",
                 "--epochs", "40", "--total-epochs", "55",
                 "--word-head", "0.3", "--word-boundary", "ctc_align",
                 "--warmup-ep", "3", "--official-pad",
                 "--lr", "1e-4",
                 "--blank-bias", "0.0",      # 与 P123 严格配对
                 "--mem-budget", "5.2e9", "--seed", SEED]
    logfile = REPO / "artifacts/logs_p144_seed1234.log"
    while True:
        if receipt(TAG).exists():
            log("%s 收据已出 ⇒ 完成" % TAG)
            break
        if pids_for(TAG):
            pass                      # 在跑就等着，不做任何止损判断
        else:
            if n >= MAX_RESTART:
                log("⚠️ %s 已达重启上限 %d" % (TAG, MAX_RESTART))
                return 1
            n += 1
            has_ck = last.exists()
            if has_ck and last.stat().st_size < 1_000_000:
                last.unlink()
                log("⚠️ last.pt 残file 已删除")
                has_ck = False
            log("%s 进程不在（第 %d/%d 次）⇒ 等 %ds 后 resume=%s 重启"
                % (TAG, n, MAX_RESTART, WAIT_GPU, has_ck))
            time.sleep(WAIT_GPU)
            if receipt(TAG).exists():
                break
            argv = base_argv + (["--resume", str(last)] if has_ck else [])
            with open(logfile, "a", encoding="utf-8") as fh:
                fh.write("\n[CHAIN %s] 启动/重启 resume=%s\n"
                         % (time.strftime("%m-%d %H:%M:%S"), has_ck))
                fh.flush()
                subprocess.Popen(argv, cwd=str(REPO), stdout=fh, stderr=fh)
        time.sleep(POLL)

    if receipt(TAG).exists():
        # 判据区间仍是 ep35-40，但**解读不同**：这里看的是方差，不是改进
        subprocess.call([PY, "tools/blank_gov/pair_analyze.py",
                         "--a", TAG, "--b", BASE, "--lo", "35", "--hi", "40",
                         "--note", "换 seed（42 → %s）的方差测量，非改进实验" % SEED],
                        cwd=str(REPO))
        # 额外报一个 best 对比（这才是回答 H-a/H-b 的关键数）
        try:
            a = json.load(open(receipt(TAG)))
            b = json.load(open(receipt(BASE)))
            log("★ 关键点：best %s = %.2f  vs  best %s = %.2f  ⇒ 差 %+.2f pp"
                % (TAG, a["best_wer_official"], BASE, b["best_wer_official"],
                   a["best_wer_official"] - b["best_wer_official"]))
            log("  判读：|差| < 1.556 ⇒ 支持 H-a（基线稳定）；"
                "> 1.556 ⇒ 支持 H-b（单次 run 方差大）")
        except Exception as e:
            log("best 对比失败：%s" % e)
    log("=== P144 链条结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
