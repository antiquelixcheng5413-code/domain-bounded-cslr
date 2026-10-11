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
    # 与 P119 的唯一差别：**训练轮数/里程碑**（55 + [35,45]）和**官方补帧**
    # （其余：hflip 开、word_head 0.3、warmup 3、160res、fp32、seed42）
    run("P123", [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                 "--tag", "p123-official55",
                 "--module", "VAC",
                 "--img-size", "160",
                 "--epochs", "55", "--total-epochs", "55",
                 "--word-head", "0.3", "--word-boundary", "ctc_align",
                 "--warmup-ep", "3",
                 "--official-pad",
                 "--mem-budget", "5.2e9",
                 "--seed", "42"],
        "logs_p123_official55.log")

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

    # ---------- 4. 仅当判活才微调 ----------
    if verdict == "判活":
        run("P124c", [PY, "-u", "tools/blank_gov/p78_train_official_tfnet.py",
                      "--tag", "p124c-ssl-ft",
                      "--module", "VAC",
                      "--img-size", "160",
                      "--epochs", "20", "--total-epochs", "20",
                      "--word-head", "0.3", "--word-boundary", "ctc_align",
                      "--warmup-ep", "3",
                      "--backbone-ckpt", str(ssl_ck),
                      "--mem-budget", "5.2e9",
                      "--seed", "42"],
            "logs_p124c_ssl_ft.log")
    else:
        log("探针未判活 ⇒ 不投 9 小时做 20 轮微调（省下的时间用于其他方向）")

    log("链条结束")


if __name__ == "__main__":
    main()
