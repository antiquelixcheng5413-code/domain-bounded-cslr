#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P142：label smoothing（B 步）—— 等 P141 结束后自动开跑

依据
  【有】ref17 Gloss2Text 提出 SALS（语义感知 label smoothing，手语专用），
        传统 LS：y_ls = (1-β)·onehot + β·uniform
  【有】ref16 Sign2GPT 用 label smoothing 0.1
  【间接】LS 抑制过拟合是通用原理（Szegedy 2016 / Müller 2019）
  ⚠️【无】CTC-based CSLR 上 LS 的**直接**论文证据 ⇒ 本项目首次尝试

实现（数学）
  KL(y_ls‖p) = (1-β)·(-log p[true]) + β·[-(1/N)·Σ_c log p_c]
                 └─ 对 CTC 即 CTC_loss ─┘   └─ ls_penalty() ─┘
  第二项只依赖 log_probs，与 CTCLoss 完全兼容（不需要改前向后向）。
  ⚠️ β 的有效强度在 CTC 上被放大 ~9 倍（路径求和压低了 per-step 有效 CE）⇒
     **用 β=0.01**（≈ 标准 CE 上的 label_smoothing=0.1，与 ref16 强度相当）

配置 = P123 全部（lr 1e-4 + 55轮里程碑[35,45] + 官方补帧 + wh0.3 + warmup3
       + hflip + bf16 + 160res）
唯一变量 = --label-smoothing 0 → 0.01
🔴 直接在主配置上做（P140 已证伪「小配置搜索 → 回代」策略）
评估 bias=0（与 P123 严格配对）
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
CHAIN_LOG = REPO / "artifacts/logs_p142_chain.log"   # 显式写死

MAX_RESTART = 12
WAIT_GPU = 90
POLL = 60

MIN_EP = 20
BAD_MARGIN = 3.0
RECENT = 3

TAG = "p142-ls001-main"
PREV = "p141-lr15e4-main"
BASE = "p123-official55bf16"
PLANNED = 40
LS = "0.01"


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


def wait_prev():
    """等前一个任务**真正结束**（保证一次只跑一个）。

    🔴 必须同时满足「收据已出」+「进程归零」两个条件才开跑：
       - 只看进程：P141 链条可能正在重启它（进程短暂消失）⇒ 会抢跑成并行
       - 只看收据：收据由链条写，写完时训练已停，两条一起看最稳
    """
    log("等待 %s 结束（需收据已出 且 进程归零）…" % PREV)
    waited = 0
    while waited < 36 * 3600:
        if receipt(PREV).exists() and not pids_for(PREV):
            log("%s 收据已出且进程归零 ⇒ 开始" % PREV)
            return
        time.sleep(300)
        waited += 300
    log("⚠️ 等待超时（36h），仍未满足两个条件 ⇒ 放弃，不启动")


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
                    "hidden", "lr", "weight_decay", "label_smoothing",
                    "blank_bias"]},
        "history": hist,
        "best_wer_official": h[be], "best_epoch": be,
        "epochs_completed": len(hist),
        "_stopped_early": {"by": "p142_chain 早停止损", "reason": reason,
                           "epochs_run": len(hist), "epochs_planned": planned},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据已重建（%s best %.2f @ ep%d）" % (tag, h[be], be))


def main():
    log("=== P142 启动：label smoothing β=%s（B 步）===" % LS)
    wait_prev()
    last = CK / ("%s-last.pt" % TAG)
    n = 0
    base_argv = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                 "--tag", TAG, "--module", "VAC",
                 "--img-size", "160", "--amp-bf16",
                 "--epochs", "40", "--total-epochs", "55",
                 "--word-head", "0.3", "--word-boundary", "ctc_align",
                 "--warmup-ep", "3", "--official-pad",
                 "--lr", "1e-4",
                 "--label-smoothing", LS,
                 "--blank-bias", "0.0",      # 与 P123 严格配对
                 "--mem-budget", "5.2e9", "--seed", "42"]
    logfile = REPO / "artifacts/logs_p142_ls001_main.log"
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
                         "--note", "label smoothing β=%s 的净效应（vs P123）" % LS],
                        cwd=str(REPO))
    log("=== P142 链条结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
