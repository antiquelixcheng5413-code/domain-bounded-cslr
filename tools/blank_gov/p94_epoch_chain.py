"""P94/P95/P96 自动串联：20 → 30 → 40 → 50 epoch。

🔴 用法要点（这是 resume 能力的实际用法）：
   每一段的 `--total-epochs` 都等于**该段的终点**，
   于是 lr 里程碑会按该终点重排：
     30 轮⇒ ms=[19, 25]
     40 轮 ⇒ ms=[25, 33]
     50 轮 ⇒ ms=[32, 41]
   ⚠️ 每次续训会打印「上轮 planned=20 ≠ 本轮 T=30」的警告，
      这是**预期行为**（衰减点按新 T 重排，是逐级扩展的设计）。

为什么串行而非并行：单卡 8GB，一次只能跑一个。

判据（沿用 P93 定的）：
   每个节点记录 best WER 与逐 epoch 曲线
   连续 2 个节点改善 < 0.5pp ⇒ 视为饱和，打印提示但**不自动停止**
   （停止的判断权交给用户，我只负责把数据摆出来）
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
M = REPO / "artifacts/metrics/blank-gov"
CK = REPO / "artifacts/checkpoints"
PY = REPO / "venv/bin/python"
SCRIPT = REPO / "tools/blank_gov/p78_train_official_tfnet.py"

STAGES = [
    (30, "p94-30ep"),
    (40, "p95-40ep"),
    (50, "p96-50ep"),
]
BASE = "p93-20ep"          # 20 轮那一段的tag
COMMON = ["--module", "VAC", "--img-size", "160", "--no-hflip",
          "--mem-budget", "5.2e9", "--seed", "42"]


def best_of(tag):
    p = M / ("%s-official-tfnet.json" % tag)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    h = d.get("history", [])
    if not h:
        return None
    b = min(x["dev_wer_official"] for x in h)
    return {"best": b, "epochs": len(h), "minutes": d.get("minutes", 0),
            "curve": [round(x["dev_wer_official"], 2) for x in h]}


def main():
    print("=" * 84)
    print("自动串联：20 → 30 → 40 → 50 epoch")
    print("=" * 84)
    prev_tag = BASE
    prev_best = (best_of(BASE) or {}).get("best")
    print("起点 %s：best = %s" % (BASE,
          "%.2f%%" % prev_best if prev_best else "未找到"))

    for target, tag in STAGES:
        last = CK / ("%s-last.pt" % prev_tag)
        if not last.exists():
            print("\n❌ 找不到 %s，无法续训。终止。" % last)
            return 1
        print()
        print("=" * 84)
        print("▶ %s：续训 %s → %d epoch" % (tag, prev_tag, target))
        print("  lr 里程碑将按 %d 重排 ⇒ [%d, %d]"
              % (target, round(35 * target / 55), round(45 * target / 55)))
        print("=" * 84)
        cmd = [str(PY), "-u", str(SCRIPT),
               "--epochs", str(target), "--total-epochs", str(target),
               "--resume", str(last), "--tag", tag] + COMMON
        print("  命令: %s" % " ".join(cmd[-14:]))
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(REPO))
        # 只打印关键行，避免 400MB 的 OOM 噪声刷屏
        for line in (r.stdout or "").splitlines():
            if any(k in line for k in ("续训", "已完成", "scheduler 恢复",
                                       "里程碑", "警告", "ep0", "最佳",
                                       "收据", "Traceback", "Error",
                                       "❌", "⚠️")):
                print("  " + line)
        if r.returncode != 0:
            print("\n❌ %s 失败（exit=%d）" % (tag, r.returncode))
            print((r.stderr or "")[-1500:])
            return r.returncode

        cur = best_of(tag)
        if cur:
            d = cur["best"] - prev_best if prev_best else float("nan")
            print("\n  ⇒ %s best = %.2f%%（%+.2f pp vs %s）"
                  % (tag, cur["best"], -d, prev_tag))
            print("     耗时 %.0f 分钟" % cur["minutes"])
            print("     曲线 %s" % cur["curve"])
            prev_best = cur["best"]
        prev_tag = tag

    print()
    print("=" * 84)
    print("全部完成")
    print("=" * 84)
    for _, tag in [(0, BASE)] + STAGES:
        b = best_of(tag)
        if b:
            print("  %-14s best %6.2f%%  (%d epoch, %.0f 分钟)"
                  % (tag, b["best"], b["epochs"], b["minutes"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())