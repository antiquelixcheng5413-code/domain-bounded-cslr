#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P141：主配置下试探**更大**的 lr（A 步）

动机（P140 的洞察，依据【有】）：
  lr 的最优值依赖调度长度 —— 短调度配小 lr，长调度配大 lr。
    20 轮调度 : lr 5e-5 = 58.30 ✅ 优于 1e-4 (59.61)
    55 轮调度 : lr 5e-5 = 57.14 ❌ 劣于 1e-4 (55.84)
  ⇒ 主配置跑 55 轮调度，**1e-4 未必是上限**，往上试探。

🔴 必须**直接在主配置上做**：P140 已证伪「小配置搜索 → 回代主配置」的策略。

配置 = P123 全部（55轮调度里程碑[35,45] + 官方补帧 + wh0.3 + warmup3 + hflip + bf16 + 160res）
唯一变量 = lr 1e-4 → 1.5e-4
🔴 --epochs 40 --total-epochs 55 故意不等：里程碑与 P123 一致才能 epoch 级配对，
   40 轮覆盖 P123 best 所在 ep39，省 7 轮 ≈3h。
评估用 bias=0（与 P123 严格配对）；加 bias 的收益事后用 P134 扫描（零成本 8 min）。
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = str(REPO / "venv/bin/python")
METRICS = REPO / "artifacts/metrics/blank-gov"
CK = REPO / "artifacts/checkpoints"
CHAIN_LOG = REPO / "artifacts/logs_p141_chain.log"   # 显式写死

MAX_RESTART = 12
WAIT_GPU = 90
POLL = 60

MIN_EP = 20          # 里程碑在 35，需跑够才谈得上比较
BAD_MARGIN = 3.0
RECENT = 3

TAG = "p141-lr15e4-main"
BASE = "p123-official55bf16"
PLANNED = 40


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


def should_stop(tag, base, min_ep):
    last = CK / ("%s-last.pt" % tag)
    if not last.exists() or not receipt(base).exists():
        return False, ""
    try:
        import torch
        hist = torch.load(last, map_location="cpu",
                          weights_only=False)["history"]
    except Exception as e:
        return False, "读 ckpt 失败：%s" % e
    if len(hist) < min_ep:
        return False, ""
    h = {r["epoch"]: r["dev_wer_official"] for r in hist}
    b = {r["epoch"]: r["dev_wer_official"]
         for r in json.load(open(receipt(base)))["history"]}
    # ⚠️ P123 的 ep1-31 是坏解码截断的虚高值 ⇒ 只比 ep32+
    common = sorted(e for e in h if e in b and e >= 32)
    if len(common) < 3:
        return False, ""
    deltas = [(e, h[e] - b[e]) for e in common]
    if len(deltas) < RECENT + 2:
        return False, ""
    recent = deltas[-RECENT:]
    if not all(d > BAD_MARGIN for _, d in recent):
        return False, ""
    if recent[-1][1] < deltas[-(RECENT + 2)][1]:
        return False, ""
    return True, ("最近 %d 轮 ΔWER 全 > %.1f pp 且发散：%s"
                  % (RECENT, BAD_MARGIN,
                     ", ".join("ep%02d %+.2f" % (e, d) for e, d in recent)))


def write_receipt(tag, reason, planned):
    import torch
    ck = torch.load(CK / ("%s-last.pt" % tag), map_location="cpu",
                    weights_only=False)
    hist = ck["history"]
    h = {r["epoch"]: r["dev_wer_official"] for r in hist}
    be = min(h, key=h.get)
    receipt(tag).write_text(json.dumps({
        "tag": tag,
        "config": {k: ck.get("config", {}).get(k) for k in
                   ["img_size", "amp_bf16", "word_head", "epochs",
                    "total_epochs", "seed", "official_pad", "batch",
                    "hidden", "lr", "weight_decay", "blank_bias"]},
        "history": hist,
        "best_wer_official": h[be], "best_epoch": be,
        "epochs_completed": len(hist),
        "_stopped_early": {"by": "p141_chain 早停止损", "reason": reason,
                           "epochs_run": len(hist), "epochs_planned": planned},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据已重建（%s best %.2f @ ep%d）" % (tag, h[be], be))


def main():
    log("=== P141 启动：主配置 lr 1e-4 → 1.5e-4（A 步）===")
    last = CK / ("%s-last.pt" % TAG)
    n = 0
    base_argv = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                 "--tag", TAG, "--module", "VAC",
                 "--img-size", "160", "--amp-bf16",
                 "--epochs", "40", "--total-epochs", "55",
                 "--word-head", "0.3", "--word-boundary", "ctc_align",
                 "--warmup-ep", "3", "--official-pad",
                 "--lr", "1.5e-4",
                 "--blank-bias", "0.0",      # 与 P123 严格配对
                 "--mem-budget", "5.2e9", "--seed", "42"]
    logfile = REPO / "artifacts/logs_p141_lr15e4_main.log"
    while True:
        if receipt(TAG).exists():
            log("%s 收据已出 ⇒ 完成" % TAG)
            break
        running = pids_for(TAG)
        if running:
            stop, why = should_stop(TAG, BASE, MIN_EP)
            if stop:
                log("⏹ %s 触发早停：%s" % (TAG, why))
                for pid in running:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except Exception:
                        pass
                time.sleep(30)
                write_receipt(TAG, why, PLANNED)
                break
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
        subprocess.call([PY, "tools/blank_gov/pair_analyze.py",
                         "--a", TAG, "--b", BASE, "--lo", "35", "--hi", "40",
                         "--note", "主配置 lr 1e-4 -> 1.5e-4 的净效应"],
                        cwd=str(REPO))
    log("=== P141 链条结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
