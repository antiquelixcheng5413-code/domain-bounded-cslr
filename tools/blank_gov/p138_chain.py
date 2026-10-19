"""P138：learning rate 减小方向（1e-4 → 5e-5），排在 P137 之后自动接上。

背景（两个"增大"全部失败）
  P136 wd 1e-4 → 1e-3 : mean +4.351 / 胜 0/8，Δtrain_ctc 单调扩大(+0.64→+5.91)
                        ⇒ **欠拟合**（正则过强把模型压死）
  P137 lr 1e-4 → 2e-4 : mean +2.843 / 胜 1/8，Δtrain_ctc 单调扩大(-0.50→+3.04)
                        ⇒ **训练不稳定**（lr 过大，ep1 略好之后持续恶化）
  ⇒ 两者共同模式：Δtrain_ctc 单调扩大 ⇒ 训练侧受损
  ⇒ 提示：官方的 lr=1e-4 / wd=1e-4 **可能已经在合理区间**，增大都有害

本实验：lr 1e-4 → **5e-5**（减半），与 P124e 唯一变量。

判据（预先写死，ep13-20 与 P124e 配对）
  mean ≤ −1.556 pp 且 SNR ≥ 2 且胜率 ≥ 6/8 ⇒ lr 减小有效
  |mean| < 1.556 pp                          ⇒ 不可判 ⇒ 超参方向基本排除
  mean ≥ +1.556 pp                           ⇒ 也有害 ⇒ lr=1e-4 确认为局部最优

🔴 若本次仍无收益，结论就是：**lr / wd 都不是缺口来源**，
   应转向需实现的抗过拟合手段（label smoothing / 更强增强 / dropout）。

启动时会先等 P137 结束（收据出现或训练进程消失）⇒ 保证一次只跑一个。
沿用 P137 的早停止损（ep≥10 且最近 3 轮 ΔWER 全 >3pp 且发散）。
"""
from __future__ import annotations

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
CHAIN_LOG = REPO / "artifacts/logs_p138_chain.log"

MAX_RESTART = 12
WAIT_GPU = 90
TAG = "p138-lr5e5"
PREV = "p137-lr2e4"
BASE = "p124e-wh0-bf16"

MIN_EP, BAD_MARGIN, RECENT = 10, 3.0, 3


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


def wait_prev() -> None:
    log("等待 %s 结束（保证一次只跑一个）…" % PREV)
    waited = 0
    while waited < 36 * 3600:            # 最多等 36 h
        if receipt(PREV).exists():
            log("%s 收据已出 ⇒ 开始" % PREV)
            return
        if not pids_for(PREV):
            # 进程没了但没收据：可能崩溃重启中，再等一会
            time.sleep(300)
            waited += 300
            if not pids_for(PREV):
                log("%s 进程已消失且无收据 ⇒ 不再等待，开始" % PREV)
                return
            continue
        time.sleep(300)
        waited += 300
    log("⚠️ 等待超时，直接开始")


def should_stop() -> tuple[bool, str]:
    last = CK / ("%s-last.pt" % TAG)
    if not last.exists():
        return False, ""
    try:
        import torch
        hist = torch.load(last, map_location="cpu",
                          weights_only=False)["history"]
    except Exception as e:
        return False, "读 ckpt 失败：%s" % e
    if len(hist) < MIN_EP:
        return False, ""
    h = {r["epoch"]: r["dev_wer_official"] for r in hist}
    b = {r["epoch"]: r["dev_wer_official"]
         for r in json.load(open(receipt(BASE)))["history"]}
    common = sorted(e for e in h if e in b)
    if len(common) < MIN_EP:
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


def write_receipt(reason: str) -> None:
    import torch
    ck = torch.load(CK / ("%s-last.pt" % TAG), map_location="cpu",
                    weights_only=False)
    hist = ck["history"]
    h = {r["epoch"]: r["dev_wer_official"] for r in hist}
    be = min(h, key=h.get)
    receipt(TAG).write_text(json.dumps({
        "tag": TAG,
        "config": {k: ck.get("config", {}).get(k) for k in
                   ["img_size", "amp_bf16", "word_head", "epochs", "seed",
                    "official_pad", "batch", "hidden", "lr", "weight_decay"]},
        "history": hist,
        "best_wer_official": h[be], "best_epoch": be,
        "epochs_completed": len(hist),
        "_stopped_early": {"by": "p138_chain 早停止损", "reason": reason,
                           "epochs_run": len(hist), "epochs_planned": 20},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据已重建（best %.2f @ ep%d）" % (h[be], be))


def ensure_train() -> bool:
    base = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
            "--tag", TAG, "--module", "VAC", "--img-size", "160",
            "--amp-bf16", "--epochs", "20", "--total-epochs", "20",
            "--word-head", "0.0", "--lr", "5e-5",
            "--mem-budget", "5.2e9", "--seed", "42"]
    last = CK / ("%s-last.pt" % TAG)
    n = 0
    while True:
        if receipt(TAG).exists():
            log("%s 收据已出 ⇒ 完成" % TAG)
            return True
        if pids_for(TAG):
            stop, why = should_stop()
            if stop:
                log("⏹ 触发早停：%s" % why)
                for pid in pids_for(TAG):
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except Exception:
                        pass
                time.sleep(30)
                write_receipt(why)
                return True
        if not pids_for(TAG):
            if n >= MAX_RESTART:
                log("⚠️ 已达重启上限 %d" % MAX_RESTART)
                return False
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
                return True
            argv = base + (["--resume", str(last)] if has_ck else [])
            pf = REPO / "artifacts/logs_p138_lr5e5.log"
            with open(pf, "a", encoding="utf-8") as fh:
                fh.write("\n[CHAIN %s] 重启 resume=%s\n"
                         % (time.strftime("%m-%d %H:%M:%S"), has_ck))
                fh.flush()
                subprocess.Popen(argv, cwd=str(REPO), stdout=fh, stderr=fh)
        time.sleep(60)


def main() -> int:
    log("=== P138 串行链条启动（lr 1e-4 → 5e-5）===")
    wait_prev()
    ok = ensure_train()
    if not ok:
        log("❌ %s 未拿到收据" % TAG)
        return 1
    subprocess.call(
        [PY, "tools/blank_gov/pair_analyze.py",
         "--a", TAG, "--b", BASE, "--lo", "13", "--hi", "20",
         "--note", "learning rate 1e-4 → 5e-5 的净效应"],
        cwd=str(REPO))
    log("=== 链条结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
