"""P124 串行队列 —— 🔴 严格遵守「一次只训练一个」（用户 2026-10-12 明确要求）。

队列顺序：
  0. P124d（bf16 验证）        ← 已在跑，本脚本负责「等它完成 + 崩了就续训重启」
  1. P124e（词级头净效应）      bf16 + hflip + word_head=0 + 20 轮
  2. P124a 续训（ep06 → 25）
  3. P124b 探针（25 轮版，双指标判据）

设计要点：
- **全程串行**：前一个没结束绝不启动下一个（GPU 上下文被抢占 ⇒ sticky error）
- **崩溃自愈**：训练进程没了但收据没出 ⇒ 等 90s 让 GPU 释放，用 --resume 重启
  （最多 3 次）。用户办公时 GPU 可能被抢，这是必需的。
- **损坏文件校验**：final.pt < 1 MB 视为坏（CUDA sticky error 会写出 18KB 残file）
- **已完成即跳过**：收据存在就跳过，方便中断后重跑本脚本
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = str(REPO / "venv/bin/python")
MET = REPO / "artifacts/metrics/blank-gov"
CK = REPO / "artifacts/checkpoints"
LOGS = REPO / "artifacts"

POLL = 60           # 轮询间隔（秒）
WAIT_GPU = 90       # 崩溃后等 GPU 释放
MAX_RESTART = 3     # 单个任务最多重启次数


def log(*a):
    print("[QUEUE %s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                             " ".join(str(x) for x in a)), flush=True)


def receipt(tag: str) -> Path:
    return MET / ("%s-official-tfnet.json" % tag)


def pids_for(tag: str):
    r = subprocess.run(["pgrep", "-f", "tag " + tag], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def run(cmd, logname, tries=1, wait=WAIT_GPU):
    ok = False
    for k in range(tries):
        log("启动：...%s" % " ".join(cmd[-5:]))
        lp = LOGS / logname
        with open(lp, "w") as f:
            rc = subprocess.call(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
        log("结束 rc=%d（日志 %s）" % (rc, lp.name))
        if rc == 0:
            ok = True
            break
        if k + 1 < tries:
            log("失败 ⇒ 等 %ds 让 GPU 释放后重试" % wait)
            time.sleep(wait)
    return ok


def ensure_train(tag: str, cmd_builder, logname: str):
    """保证这个 tag 的训练跑完：等着；进程死了就 --resume 重启。"""
    if receipt(tag).exists():
        log("%s 已有收据 ⇒ 跳过" % tag)
        return True

    # 若当前没在跑，且已有 last.pt ⇒ 主动续训拉起
    restarts = 0
    while True:
        if receipt(tag).exists():
            log("✅ %s 完成" % tag)
            return True
        if pids_for(tag):
            time.sleep(POLL)
            continue
        if restarts >= MAX_RESTART:
            log("❌ %s 重启 %d 次仍未完成，放弃（保留现场）" % (tag, restarts))
            return False
        restarts += 1
        log("⚠️ %s 进程已退出但无收据 ⇒ 续训重启（第 %d/%d 次）"
            % (tag, restarts, MAX_RESTART))
        time.sleep(WAIT_GPU)
        cmd = cmd_builder()
        last = CK / ("%s-last.pt" % tag)
        if last.exists():
            cmd += ["--resume", str(last)]
            log("  续训起点 %s" % last.name)
        run(cmd, logname, tries=1)


def train_cmd(tag: str, word_head: str, warmup: str | None = None) -> list:
    cmd = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
           "--tag", tag,
           "--module", "VAC",
           "--img-size", "160",
           "--amp-bf16",
           "--epochs", "20", "--total-epochs", "20",
           "--word-head", word_head,
           "--mem-budget", "5.2e9",
           "--seed", "42"]
    if warmup:
        cmd += ["--word-boundary", "ctc_align", "--warmup-ep", warmup]
    return cmd


def ensure_ssl():
    """P124a 续训到 25 轮。返回 final.pt 路径或 None。"""
    tag = "p124a-simclr"
    final = CK / ("%s-final.pt" % tag)

    if final.exists():
        mb = final.stat().st_size / 1e6
        if mb > 1.0:
            log("P124a 已有 final.pt（%.1f MB）⇒ 跳过训练" % mb)
            return final
        log("⚠️ final.pt 只有 %.1f KB（sticky error 写出的残file）⇒ 删除" % (mb * 1000))
        final.unlink()

    for k in range(MAX_RESTART):
        eps = sorted(CK.glob("%s-ep*.pt" % tag))
        resume = eps[-1] if eps else None
        cmd = [PY, "-u", "tools/blank_gov/p124_ssl_pretrain.py",
               "--tag", tag, "--epochs", "25", "--batch", "128",
               "--size", "160", "--frames-per-video", "12"]
        if resume is not None:
            cmd += ["--resume", str(resume)]
            log("续训起点 %s" % resume.name)
        run(cmd, "logs_p124a_resume.log", tries=1)
        if final.exists() and final.stat().st_size > 1e6:
            log("✅ P124a 完成（%.1f MB）" % (final.stat().st_size / 1e6))
            return final
        log("⚠️ P124a 未完成（第 %d/%d 次），等 %ds 再试"
            % (k + 1, MAX_RESTART, WAIT_GPU))
        time.sleep(WAIT_GPU)
    log("❌ P124a 多次未成，放弃")
    return None


def main():
    log("=== P124 串行队列启动（一次只跑一个）===")

    # ---- 0. P124d（已在跑）----
    ensure_train("p124d-bf16-wh03",
                 lambda: train_cmd("p124d-bf16-wh03", "0.3", "3"),
                 "logs_p124d_bf16.log")

    # ---- 1. P124e：词级头净效应（只差 word_head 0.3 → 0）----
    ensure_train("p124e-wh0-bf16",
                 lambda: train_cmd("p124e-wh0-bf16", "0.0"),
                 "logs_p124e_wh0.log")

    # ---- 2. P124a 续训到 25 轮 ----
    final = ensure_ssl()
    if final is None:
        log("队列终止：P124a 未产出可用权重")
        return

    # ---- 3. 探针（25 轮版）----
    out_dir = "artifacts/frame_feats/p124_ssl25"
    run([PY, "-u", "tools/blank_gov/p124_frame_probe.py", "extract",
         "--ckpt", str(final), "--out", out_dir, "--max-videos", "2000"],
        "logs_p124b_extract_ssl25.log", tries=2)
    run([PY, "-u", "tools/blank_gov/p124_frame_probe.py", "compare",
         "--a", "artifacts/frame_feats/p124_base",
         "--b", out_dir,
         "--out", "artifacts/metrics/blank-gov/p124b-probe-ep25.json"],
        "logs_p124b_compare_ep25.log", tries=2)

    log("=== 队列全部结束 ===")


if __name__ == "__main__":
    main()
