"""P123 自适应早停器 —— 收敛就停，不再死跑到 55。

═══════════════════════════════════════════════════════════════
为什么不能 20/30 就停
═══════════════════════════════════════════════════════════════
P123 用的是**官方调度**：MultiStepLR([35,45], γ=0.2)。
⇒ ep35 之前 lr 一直是 1e-4，模型还在"探索期"。

实测对照（P119，衰减点 13/16）：
  衰减前 ep12 = 64.36 → 衰减后 ep16 = 58.90，**4 轮 −5.46 pp**
⇒ 在衰减前停 = 直接扔掉最大的一笔收益，实验等于白跑。

═══════════════════════════════════════════════════════════════
停止判据（预先写死，防作弊）
═══════════════════════════════════════════════════════════════
必须**同时**满足：
  1. 当前 epoch ≥ MIN_EP（给衰减留出观察窗口）
       - ep < 45 时 MIN_EP = 38（第一次衰减 ep35 后至少看 3 轮）
       - ep ≥ 45 时 MIN_EP = 48（第二次衰减 ep45 后至少看 3 轮）
  2. 连续 PATIENCE=6 轮没有出现新低（best_epoch ≤ cur − 6）
  3. 最近 6 轮的 WER 极差 < 噪声 1.556 pp（确认是"平台"不是"振荡"）

用 P119 回测本判据：ep33 触底 56.72 后，
  ep34-39 = 56.86/56.86/56.76/56.90/57.35/57.04（极差 0.59 < 1.556）
  ⇒ 会在 **ep39** 触发停止，而 best 仍是 56.72（best.pt 已存）
  ⇒ 省下 5 轮、结果一个都不丢 ✅

═══════════════════════════════════════════════════════════════
停止方式
═══════════════════════════════════════════════════════════════
p78 只在**正常跑完**时写收据 JSON，被杀就没有。
⇒ 本脚本 SIGTERM 之后**用 last.pt 里的 history 重建收据**，
   并打 `_early_stop` 标记（注明哪些字段拿不到）。
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
MET = REPO / "artifacts/metrics/blank-gov"
CK = REPO / "artifacts/checkpoints"
TAG = "p123-official55bf16"

PATIENCE = 6
NOISE = 1.556
MIN_EP_EARLY = 38      # 第一次衰减后至少 3 轮
MIN_EP_LATE = 48       # 第二次衰减后至少 3 轮
HARD_STOP = 55
POLL = 300             # 5 min

import torch  # noqa: E402


def log(*a):
    print("[STOP %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def read_hist():
    """从 last.pt 读 history（训练每轮都覆盖存盘）。"""
    p = CK / ("%s-last.pt" % TAG)
    if not p.exists():
        return None, None
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        log("读 last.pt 失败（可能正在写）：%r" % e)
        return None, None
    return ck.get("history", []), ck


def verdict(hist):
    if not hist:
        return "无数据", None
    cur = max(r["epoch"] for r in hist)
    wers = {r["epoch"]: r["dev_wer_official"] for r in hist}
    best_ep = min(wers, key=wers.get)
    min_ep = MIN_EP_LATE if cur >= 45 else MIN_EP_EARLY
    if cur < min_ep:
        return "continue", (cur, best_ep, wers[best_ep], "还没到最小观察轮数 %d" % min_ep)
    if cur >= HARD_STOP:
        return "continue", (cur, best_ep, wers[best_ep], "已到硬上限 55，让它自然收尾")
    if best_ep > cur - PATIENCE:
        return "continue", (cur, best_ep, wers[best_ep],
                            "最近 %d 轮内还有新低（ep%d）" % (PATIENCE, best_ep))
    last6 = [wers[e] for e in range(cur - PATIENCE + 1, cur + 1) if e in wers]
    rng = max(last6) - min(last6)
    if rng >= NOISE:
        return "continue", (cur, best_ep, wers[best_ep],
                            "最近 6 轮极差 %.2f ≥ 噪声 %.3f（还在振荡）" % (rng, NOISE))
    return "stop", (cur, best_ep, wers[best_ep],
                    "连续 %d 轮无新低且极差 %.2f < %.3f ⇒ 平台"
                    % (PATIENCE, rng, NOISE))


def pids():
    r = subprocess.run(["pgrep", "-f", "tag " + TAG], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def rebuild_receipt(hist, ck, cur, best_ep):
    """用 last.pt 重建收据（p78 被杀不会写 JSON，这里补上）。"""
    cfg = ck.get("config", {})
    best = min(r["dev_wer_official"] for r in hist)
    rec = {
        "experiment": "P78",
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "官方全套超参（55轮 + milestones[35,45] + 官方补帧 + bf16）",
        "official_code": "external/TFNet（Net.py / Module.py / videoAugmentation.py "
                         "/ DataProcessMoudle.py 均直接 import）",
        "official_benchmark_dev": {"TFNet": 42.1, "CorrNet": 47.2, "VAC": 45.1,
                                   "MAM-FSD": 44.9, "SEN": 46.5}.get(
                                       cfg.get("module", "VAC")),
        "best_wer_official": best,
        "best_epoch": best_ep,
        "module": cfg.get("module"),
        "img_size": cfg.get("img_size"),
        "amp_bf16": cfg.get("amp_bf16"),
        "word_head_w": cfg.get("word_head"),
        "word_boundary": cfg.get("word_boundary"),
        "lr_milestones": [35, 45],
        "lr_gamma": 0.2,
        "official_pad": cfg.get("official_pad"),
        "backbone_ckpt": cfg.get("backbone_ckpt") or None,
        "warmup_ep": cfg.get("warmup_ep"),
        "mem_budget_GB": round(cfg.get("mem_budget", 0) / 1e9, 2),
        "vocab": {"wordSetNum": ck.get("wordSetNum"),
                  "note": "官方 Word2Id，train+dev+test 全收、无截断"},
        "config": cfg,
        "resumed_from": "p123-official55bf16-last.pt（ep31 续训）",
        "total_epochs_planned": 55,
        "n_params_M": 35.25,
        "history": hist,
        "_early_stop": {
            "by": "tools/blank_gov/p123_watch_stop.py",
            "epoch_reached": cur,
            "criterion": "连续 %d 轮无新低 且 最近6轮极差 < %.3f" % (PATIENCE, NOISE),
            "note": "⚠️ 本收据由 last.pt 重建（进程被提前停止，未走 p78 正常收尾）",
            "fields_unavailable": ["nan_steps_skipped", "minutes",
                                   "short_samples_padded"],
        },
        "_history_caveat": "ep1-31 的 dev_wer_official 是用坏的解码截断算的，已作废；"
                           "ep31 真实值 58.66（见 p123-pad-diag.json）",
    }
    dst = MET / ("%s-official-tfnet.json" % TAG)
    dst.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据已重建 -> %s（best %.2f%% @ ep%d）" % (dst, best, best_ep))


def main():
    dry = "--dry-run" in sys.argv
    log("早停器启动（dry_run=%s, patience=%d, noise=%.3f, min_ep=%d/%d）"
        % (dry, PATIENCE, NOISE, MIN_EP_EARLY, MIN_EP_LATE))
    stopped = False
    while True:
        hist, ck = read_hist()
        if hist:
            v, info = verdict(hist)
            cur, best_ep, bw, why = info
            log("ep%02d  best %.2f%% @ ep%d  ⇒ %s（%s）"
                % (cur, bw, best_ep, v, why))
            if v == "stop" and not dry:
                log("触发早停 ⇒ SIGTERM 训练进程")
                for pid in pids():
                    try:
                        os.kill(pid, signal.SIGTERM)
                        log("  已发 SIGTERM 给 %d" % pid)
                    except Exception as e:
                        log("  kill %d 失败：%r" % (pid, e))
                time.sleep(30)
                hist2, ck2 = read_hist()
                if hist2:
                    cur2 = max(r["epoch"] for r in hist2)
                    be2 = min({r["epoch"]: r["dev_wer_official"] for r in hist2}.items(),
                              key=lambda kv: kv[1])[0]
                    rebuild_receipt(hist2, ck2, cur2, be2)
                log("早停完成；链条会继续下一步（P124a）")
                stopped = True
                break
        if dry:
            break
        if stopped:
            break
        # 训练进程没了就不用再等
        if hist and max(r["epoch"] for r in hist) >= HARD_STOP:
            log("已跑满 %d 轮，退出" % HARD_STOP)
            break
        if not pids():
            log("训练进程已不在（可能正常结束），退出")
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
