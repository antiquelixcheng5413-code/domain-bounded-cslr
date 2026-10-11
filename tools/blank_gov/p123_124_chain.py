"""P123 + P124 自动链条（等 P121 → 官方全套超参 → 自监督预训练 → 探针 → 微调）。

顺序（串行占用 GPU）：
  0. 等 P121（p121-disamb）收据出现
  1. P123：官方全套超参（55 轮 + milestones[35,45] + 官方 collate 补帧）
  2. P124a：SimCLR 域内自监督预训练（30 轮，2D resnet18）
  3. P124b：判别力探针（Cohen's d + bootstrap CI），判据预先写死
  4. 仅当探针**判活**才跑 P124c：20 轮微调（与 P119 ep13-20 逐轮配对）

⚠️ 每一步的判据都写在被调用的脚本里，本文件只做调度。
"""
from __future__ import annotations

import json
import torch
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = str(REPO / "venv/bin/python")
MET = REPO / "artifacts/metrics/blank-gov"
LOGS = REPO / "artifacts"
CK = REPO / "artifacts/checkpoints"


def log(*a):
    print("[CHAIN %s] %s" % (time.strftime("%m-%d %H:%M:%S"),
                             " ".join(str(x) for x in a)), flush=True)


def run(tag, args, logfile, check=False):
    lf = LOGS / logfile
    log("启动 %s：%s" % (tag, " ".join(args[2:])))
    with open(lf, "w", encoding="utf-8") as fh:
        p = subprocess.run(args, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
    log("%s 结束：rc=%d（日志 %s）" % (tag, p.returncode, lf))
    if check and p.returncode != 0:
        log("⚠️ %s 失败，链条中止" % tag)
        sys.exit(1)
    return p.returncode


def wait_receipt(name, timeout_h=30):
    f = MET / ("%s-official-tfnet.json" % name)
    t0 = time.time()
    while not f.exists():
        if (time.time() - t0) / 3600 > timeout_h:
            log("⚠️ 等 %s 收据超时（%dh），放弃等待" % (name, timeout_h))
            return False
        if int(time.time() - t0) % 1800 < 60:
            log("等 %s 收据…（已等 %.1f h）" % (name, (time.time() - t0) / 3600))
        time.sleep(60)
    log("✅ 收到 %s 收据" % name)
    return True


def main():
    # ---------- 0. 等 P121 ----------
    wait_receipt("p121-disamb", timeout_h=30)

    # ---------- 1. P123：官方全套超参 ----------
    # 🔴 2026-10-11 17:2x 改：加 --amp-bf16
    #    依据：P121（bf16, word_head 0）vs P93（fp32, word_head 0）单变量对比
    #      ep13-20 配对 mean **-3.747** / std 0.630 / 胜 8/8 / SNR 2.41 ⇒ **可判**
    #      best 59.61 vs 62.81（-3.20 pp）
    #    且官方本身就是 autocast+GradScaler，bf16 比 fp32 更接近官方。
    #    代价：P123 相对 P119 的差别变成三个变量（轮数/里程碑、官方补帧、bf16），
    #          归因由后续 P124d（bf16 单变量）与隔离跑补回。
    # 🔴 断点续训：若已有 last.pt 但收据未出 ⇒ 上轮是被中断的，接着跑
    p123_ck = CK / "p123-official55bf16-last.pt"
    p123_rec = MET / "p123-official55bf16-official-tfnet.json"
    p123_args = [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                 "--tag", "p123-official55bf16",
                 "--module", "VAC",
                 "--img-size", "160",
                 "--amp-bf16",
                 "--epochs", "55", "--total-epochs", "55",
                 "--word-head", "0.3", "--word-boundary", "ctc_align",
                 "--warmup-ep", "3",
                 "--official-pad",
                 "--mem-budget", "5.2e9",
                 "--seed", "42"]
    if p123_ck.exists() and not p123_rec.exists():
        p123_args += ["--resume", str(p123_ck)]
        log("检测到未完成的 P123 权重（%s）⇒ **续训**（非重跑）" % p123_ck.name)
        log("⚠️ 历史里 ep1~%s 的 WER 是用**坏的解码截断**算的，已作废"
            % "(见 receipt epoch)")
    run("P123", p123_args, "logs_p123_official55.log")

    # ---------- 2. P124a：自监督预训练 ----------
    # 12 帧/视频 × 4973 ≈ 60k 帧；batch 128 ⇒ ~470 步/轮，约 7 min/轮
    run("P124a", [PY, "-u", "tools/blank_gov/p124_ssl_pretrain.py",
                  "--tag", "p124a-simclr",
                  "--epochs", "40", "--batch", "128",
                  "--size", "160", "--frames-per-video", "12"],
        "logs_p124a_simclr.log", check=False)

    ssl_ck = CK / "p124a-simclr-final.pt"
    if not ssl_ck.exists():
        log("⚠️ 没找到 %s，跳过探针与微调" % ssl_ck)
        return

    # ---------- 3. P124b：判别力探针 ----------
    fa = REPO / "artifacts/frame_feats/r18_imagenet"
    fb = REPO / "artifacts/frame_feats/r18_simclr"
    run("P124b-ext-A", [PY, "-u", "tools/blank_gov/p124_frame_probe.py",
                        "extract", "--out", str(fa),
                        "--size", "160", "--max-videos", "2000"],
        "logs_p124b_extract_a.log")
    run("P124b-ext-B", [PY, "-u", "tools/blank_gov/p124_frame_probe.py",
                        "extract", "--ckpt", str(ssl_ck), "--out", str(fb),
                        "--size", "160", "--max-videos", "2000"],
        "logs_p124b_extract_b.log")

    probe_out = MET / "p124b-ssl-vs-imagenet.json"
    rc = run("P124b-probe", [PY, "-u", "tools/blank_gov/p124_frame_probe.py",
                             "compare", "--a", str(fa), "--b", str(fb),
                             "--name-a", "ImageNet", "--name-b", "P124a-SimCLR",
                             "--out", str(probe_out)],
             "logs_p124b_compare.log")

    verdict = "未知"
    if probe_out.exists():
        verdict = json.loads(probe_out.read_text(encoding="utf-8")).get(
            "verdict", "未知")
    log("探针判定 = %s（rc=%d；rc!=0 视为不判活）" % (verdict, rc))

    # ---------- 4. P124d：bf16 对照组（**无论探针结果如何都跑**）----------
    # 一箭双雕：
    #   (a) bf16 效应的**第二次独立验证**：P124d vs CTRL（fp32，其余全同）
    #       —— P121 那次是 word_head=0 配置下测的，这次是 word_head=0.3+hflip 配置
    #   (b) 自监督微调的**配对对照组**（与 P124c 只差主干初始化）
    # ⚠️ 不加 --official-pad：CTRL/P119 都没有，保持 bf16 单变量
    run("P124d", [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                  "--tag", "p124d-bf16ctrl",
                  "--module", "VAC",
                  "--img-size", "160",
                  "--amp-bf16",
                  "--epochs", "20", "--total-epochs", "20",
                  "--word-head", "0.3", "--word-boundary", "ctc_align",
                  "--warmup-ep", "3",
                  "--mem-budget", "5.2e9",
                  "--seed", "42"],
        "logs_p124d_bf16ctrl.log")

    # ---------- 5. 仅当探针判活才做 SSL 微调 ----------
    if verdict == "判活":
        run("P124c", [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                      "--tag", "p124c-ssl-ft",
                      "--module", "VAC",
                      "--img-size", "160",
                      "--amp-bf16",
                      "--epochs", "20", "--total-epochs", "20",
                      "--word-head", "0.3", "--word-boundary", "ctc_align",
                      "--warmup-ep", "3",
                      "--backbone-ckpt", str(ssl_ck),
                      "--mem-budget", "5.2e9",
                      "--seed", "42"],
            "logs_p124c_ssl_ft.log")
        log("P124c 与 P124d 逐轮配对 ⇒ 自监督预训练的净效应")
    else:
        log("探针未判活 ⇒ 不投 9 小时做 SSL 微调"
            "（P124d 仍跑，用于独立复核 bf16 效应）")

    log("链条结束")


if __name__ == "__main__":
    main()
