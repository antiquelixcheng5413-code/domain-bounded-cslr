#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P140 双阶段串行链条（一次只跑一个训练）

阶段 1  P139 续训：lr 2.5e-5（P138 已证明 5e-5 有效，继续往下探是否还能更好）
阶段 2  P140 回代：把已验证的 lr=5e-5 套回**最佳主配置**（P123 的全部设置）

🔴 为什么要写新链条而不是修 p139_chain.py：
   上一版是用 sed 批量替换生成的，**小写路径没被替换** ⇒
   CHAIN_LOG 仍写 logs_p138_chain.log、训练日志仍写 logs_p138_lr5e5.log
   ⇒ P139 的输出追加进了 P138 的日志文件（污染）。本文件所有路径显式写死。

🔴 P140 的 --epochs 40 --total-epochs 55 是**故意不等**：
   --total-epochs 决定 lr 里程碑（55 ⇒ [35,45]，与 P123 完全一致，才能 epoch 级配对）
   --epochs 决定实际跑多少轮（40，覆盖 P123 的 best 所在 ep39，省 7 轮 ≈3h）
   脚本会打印"实际轮数由 --epochs 决定"的警告，属预期行为。
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
CHAIN_LOG = REPO / "artifacts/logs_p140_chain.log"   # 🔴 显式写死，别再用变量拼

MAX_RESTART = 12
WAIT_GPU = 90
POLL = 60

# 早停判据（三条同时满足才停）
MIN_EP = 10          # 至少跑够这么多轮才考虑停
BAD_MARGIN = 3.0     # 最近 RECENT 轮 ΔWER 全部 > 这个值
RECENT = 3           # 且必须发散（最近一轮 ≥ 3 轮前）


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
    """三条同时满足才停：跑够轮数 · 最近 3 轮 Δ 全 > 3pp · 发散"""
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
    common = sorted(e for e in h if e in b)
    if len(common) < min_ep:
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
    """被 SIGTERM 杀掉时 p78 不写收据 ⇒ 用 last.pt 的 history 重建。
    不重建的话链条会误判"未完成"而反复重启。"""
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
                    "hidden", "lr", "weight_decay"]},
        "history": hist,
        "best_wer_official": h[be], "best_epoch": be,
        "epochs_completed": len(hist),
        "_stopped_early": {"by": "p140_chain 早停止损", "reason": reason,
                           "epochs_run": len(hist), "epochs_planned": planned},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据已重建（%s best %.2f @ ep%d）" % (tag, h[be], be))


def ensure_train(tag, base_argv, base, min_ep, planned, logname):
    """跑一个训练：崩溃自动 resume 重启 + 早停止损。返回是否拿到收据。"""
    logfile = REPO / logname          # 🔴 显式传入，不再拼错
    last = CK / ("%s-last.pt" % tag)
    n = 0
    while True:
        if receipt(tag).exists():
            log("%s 收据已出 ⇒ 完成" % tag)
            return True
        running = pids_for(tag)
        if running:
            stop, why = should_stop(tag, base, min_ep)
            if stop:
                log("⏹ %s 触发早停：%s" % (tag, why))
                for pid in running:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except Exception:
                        pass
                time.sleep(30)        # 等 last.pt 落盘
                write_receipt(tag, why, planned)
                return True
        else:
            if n >= MAX_RESTART:
                log("⚠️ %s 已达重启上限 %d" % (tag, MAX_RESTART))
                return False
            n += 1
            has_ck = last.exists()
            if has_ck and last.stat().st_size < 1_000_000:
                # sticky error 会写出 18KB 残file，按"存在"判断会误用坏权重
                last.unlink()
                log("⚠️ %s last.pt 残file 已删除" % tag)
                has_ck = False
            log("%s 进程不在（第 %d/%d 次）⇒ 等 %ds 后 resume=%s 重启"
                % (tag, n, MAX_RESTART, WAIT_GPU, has_ck))
            time.sleep(WAIT_GPU)
            if receipt(tag).exists():
                return True
            argv = base_argv + (["--resume", str(last)] if has_ck else [])
            with open(logfile, "a", encoding="utf-8") as fh:
                fh.write("\n[CHAIN %s] 启动/重启 resume=%s\n"
                         % (time.strftime("%m-%d %H:%M:%S"), has_ck))
                fh.flush()
                subprocess.Popen(argv, cwd=str(REPO), stdout=fh, stderr=fh)
        time.sleep(POLL)


def main():
    log("=== P140 双阶段串行链条启动（一次只跑一个）===")

    # ---------- 阶段 1：P139 续训（lr 2.5e-5，与 P124e 唯一变量）----------
    ok = ensure_train(
        tag="p139-lr25e5",
        base_argv=[PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                   "--tag", "p139-lr25e5", "--module", "VAC",
                   "--img-size", "160", "--amp-bf16",
                   "--epochs", "20", "--total-epochs", "20",
                   "--word-head", "0.0", "--lr", "2.5e-5",
                   "--mem-budget", "5.2e9", "--seed", "42"],
        base="p124e-wh0-bf16", min_ep=10, planned=20,
        logname="artifacts/logs_p139_lr25e5.log")
    if ok:
        subprocess.call([PY, "tools/blank_gov/pair_analyze.py",
                         "--a", "p139-lr25e5", "--b", "p124e-wh0-bf16",
                         "--lo", "13", "--hi", "20",
                         "--note", "lr 1e-4 -> 2.5e-5 的净效应"],
                        cwd=str(REPO))
    else:
        log("⚠️ P139 未拿到收据，继续下一阶段")

    # ---------- 阶段 2：P140 回代（P123 主配置 + lr 5e-5）----------
    # P123 = 55轮调度 + 官方补帧 + wh0.3 + warmup3 + hflip + bf16 + 160res
    # 唯一变量 = lr 1e-4 → 5e-5
    # 🔴 --epochs 40 控制实际轮数；--total-epochs 55 控制里程碑 [35,45]
    #    ⇒ 与 P123 里程碑一致才能 epoch 级配对，同时省掉 ep41-55（P123 best 在 ep39）
    ok = ensure_train(
        tag="p140-lr5e5-main",
        base_argv=[PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                   "--tag", "p140-lr5e5-main", "--module", "VAC",
                   "--img-size", "160", "--amp-bf16",
                   "--epochs", "40", "--total-epochs", "55",
                   "--word-head", "0.3", "--word-boundary", "ctc_align",
                   "--warmup-ep", "3", "--official-pad",
                   "--lr", "5e-5",
                   "--mem-budget", "5.2e9", "--seed", "42"],
        base="p123-official55bf16", min_ep=20, planned=40,
        logname="artifacts/logs_p140_lr5e5_main.log")
    if ok:
        subprocess.call([PY, "tools/blank_gov/pair_analyze.py",
                         "--a", "p140-lr5e5-main", "--b", "p123-official55bf16",
                         "--lo", "35", "--hi", "40",
                         "--note", "lr 5e-5 回代主配置的净效应（vs P123）"],
                        cwd=str(REPO))

    log("=== P140 链条全部结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
