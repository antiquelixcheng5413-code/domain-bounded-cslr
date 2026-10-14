"""P137 串行链条：learning rate 调优（lr 1e-4 → 2e-4）+ 早停止损。

背景
- P136（wd 1e-4 → 1e-3）已**止损**：8 轮全程更差，且 Δtrain_ctc 单调扩大
  （+0.64 → +5.91）⇒ **欠拟合**（wd 太大把模型压死），不是过拟合
  ⇒ 增大 weight decay 这条路不通
- lr 是完全未探索的维度（从未调过），且零实现成本 ⇒ 下一个就测它

本次：lr 1e-4 → **2e-4**（2 倍），与 P124e 唯一变量。
P124e = 160 / bf16 / wh0 / 20 轮 / 无补帧 / seed 42 / lr 1e-4

判据（预先写死，ep13-20 与 P124e 配对）
  mean ≤ −1.556 pp 且 SNR ≥ 2 且胜率 ≥ 6/8 ⇒ lr 增大有效
  |mean| < 1.556 pp                          ⇒ 不可判（试 5e-5）
  mean ≥ +1.556 pp                           ⇒ 有害（试 5e-5）

🔴 早停止损（P127 / P136 两次手动止损的经验固化）
  同时满足才停：
    1. 已完成 ep ≥ 10
    2. 最近 3 轮的 ΔWER（P137 − P124e）**全部 > 3.0 pp**（明显更差）
    3. Δ 在扩大（最近一轮的 Δ ≥ 3 轮前的 Δ）⇒ 发散，不是收敛
  ⇒ 止损后立即写收据（p78 被杀不写收据，需重建）

后续：若 2e-4 有效则细化（3e-4 / 1.5e-4）；无效或有害则试 5e-5。
"""
from __future__ import annotations

import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = str(REPO / "venv/bin/python")
METRICS = REPO / "artifacts/metrics/blank-gov"
CK = REPO / "artifacts/checkpoints"
CHAIN_LOG = REPO / "artifacts/logs_p137_chain.log"

MAX_RESTART = 12
WAIT_GPU = 90
TAG = "p137-lr2e4"
BASE = "p124e-wh0-bf16"

# 早停参数
MIN_EP = 10
BAD_MARGIN = 3.0        # 明显更差的阈值（pp）
RECENT = 3


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


def base_curve() -> dict[int, float]:
    d = json.load(open(receipt(BASE)))
    return {r["epoch"]: r["dev_wer_official"] for r in d["history"]}


def should_stop() -> tuple[bool, str]:
    """读 last.pt 的 history，与基线比较，判断是否止损。"""
    last = CK / ("%s-last.pt" % TAG)
    if not last.exists():
        return False, ""
    try:
        import torch
        hist = torch.load(last, map_location="cpu",
                          weights_only=False)["history"]
    except Exception as e:                       # 存盘过程中可能是半文件
        return False, "读 ckpt 失败：%s" % e
    if len(hist) < MIN_EP:
        return False, ""
    h = {r["epoch"]: r["dev_wer_official"] for r in hist}
    b = base_curve()
    common = sorted(e for e in h if e in b)
    if len(common) < MIN_EP:
        return False, ""
    deltas = [(e, h[e] - b[e]) for e in common]
    if len(deltas) < RECENT:
        return False, ""
    recent = deltas[-RECENT:]
    if not all(d > BAD_MARGIN for _, d in recent):
        return False, ""
    # 发散判定：最近一轮的 Δ 不小于 3 轮前的 Δ
    if len(deltas) >= RECENT + 2:
        earlier = deltas[-(RECENT + 2)][1]
        if recent[-1][1] < earlier:
            return False, ""
    msg = ("最近 %d 轮 ΔWER 全部 > %.1f pp 且发散：%s"
           % (RECENT, BAD_MARGIN,
              ", ".join("ep%02d %+.2f" % (e, d) for e, d in recent)))
    return True, msg


def write_receipt(reason: str) -> None:
    """被止损时用 last.pt 的 history 重建收据（p78 被杀不写收据）。"""
    import torch
    last = CK / ("%s-last.pt" % TAG)
    ck = torch.load(last, map_location="cpu", weights_only=False)
    hist = ck["history"]
    h = {r["epoch"]: r["dev_wer_official"] for r in hist}
    be = min(h, key=h.get)
    out = receipt(TAG)
    out.write_text(json.dumps({
        "tag": TAG,
        "config": {k: ck.get("config", {}).get(k) for k in
                   ["img_size", "amp_bf16", "word_head", "epochs", "seed",
                    "official_pad", "batch", "hidden", "lr", "weight_decay"]},
        "history": hist,
        "best_wer_official": h[be], "best_epoch": be,
        "epochs_completed": len(hist),
        "_stopped_early": {"by": "p137_chain 早停止损", "reason": reason,
                           "epochs_run": len(hist), "epochs_planned": 20},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据已重建 -> %s（best %.2f @ ep%d）" % (out.name, h[be], be))


def ensure_train() -> bool:
    base = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
            "--tag", TAG, "--module", "VAC", "--img-size", "160",
            "--amp-bf16", "--epochs", "20", "--total-epochs", "20",
            "--word-head", "0.0", "--lr", "2e-4",
            "--mem-budget", "5.2e9", "--seed", "42"]
    last = CK / ("%s-last.pt" % TAG)
    n = 0
    while True:
        if receipt(TAG).exists():
            log("%s 收据已出 ⇒ 完成" % TAG)
            return True
        # 早停检查（只在训练进行中检查）
        if pids_for(TAG):
            stop, why = should_stop()
            if stop:
                log("⏹ 触发早停：%s" % why)
                pids = pids_for(TAG)
                for pid in pids:
                    try:
                        import os
                        os.kill(int(pid), 15)
                    except Exception:
                        pass
                time.sleep(30)          # 等 last.pt 落盘
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
            pf = REPO / "artifacts/logs_p137_lr2e4.log"
            with open(pf, "a", encoding="utf-8") as fh:
                fh.write("\n[CHAIN %s] 重启 resume=%s\n"
                         % (time.strftime("%m-%d %H:%M:%S"), has_ck))
                fh.flush()
                subprocess.Popen(argv, cwd=str(REPO), stdout=fh, stderr=fh)
        time.sleep(60)


def main() -> int:
    log("=== P137 串行链条启动（lr 1e-4 → 2e-4，带早停止损）===")
    ok = ensure_train()
    if not ok:
        log("❌ %s 未拿到收据" % TAG)
        return 1
    subprocess.call(
        [PY, "tools/blank_gov/pair_analyze.py",
         "--a", TAG, "--b", BASE, "--lo", "13", "--hi", "20",
         "--note", "learning rate 1e-4 → 2e-4 的净效应"],
        cwd=str(REPO))
    log("=== 链条结束 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
