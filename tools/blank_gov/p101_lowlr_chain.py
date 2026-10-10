"""P101：验证「lr 低时词级头才有效」假设 —— 两组串行。

🔴 为什么要对照组：
   P100 是**单次观察**，得出「ep14-20（lr衰减后）连续 7/7 为正」。
   若现在只跑「低lr 加大权重」一组，则两个因素混在一起：
     因素A = lr 低时词级头才有益
     因素 B = 权重更大（而不是更有益）
   ⇒ 必须跑 mult=1.0 的对照组，才能分离。

三组对照关系：
  P93        无词级头            （基线，已有）
  P100       词级头 ×1.0 全程    （已有 ⇒ 等价于本次对照组）
  P101-B     词级头 ×1.0 全程    （**同 seed 重跑**⇒ 测抖动基线，最关键）
  P101-A     词级头 ×1.0→×2.0   （本次新增：低 lr 增强）

⚠️ P101-B 是「同配置重跑」，它测的是**同 seed 下的 CUDA 抖动**。
   P100 的 warmup 期（ep01-03）给出 std=0.82pp，那是单次估计；
   P101-B 能给出更可靠的抖动基线，用于判断 P101-A 的改善是否真实。
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = REPO / "venv/bin/python"
SCRIPT = REPO / "tools/blank_gov/p78_train_official_tfnet.py"

RUNS = [
    # (tag, mult, 说明)
    ("p101b-ctrl", 1.0, "对照组：权重恒定 ×1.0（同配置重跑 ⇒ 抖动基线）"),
    ("p101a-lowlr", 2.0, "实验组：lr 衰减后权重 ×2.0（验证『lr 低时才有效』）"),
]
COMMON = ["--module", "VAC", "--epochs", "20", "--total-epochs", "20",
          "--img-size", "160", "--no-hflip", "--mem-budget", "5.2e9",
          "--seed", "42", "--word-head", "0.3",
          "--word-boundary", "ctc_align", "--warmup-ep", "3"]


def main():
    print("=" * 88)
    print("P101：验证「lr 低时词级头才有效」—— 两组串行")
    print("=" * 88)
    print("  P93无词级头 / P100 ×1.0 全程（已有）")
    print("  P101-B ×1.0 全程（同配置重跑 ⇒ 抖动基线，最关键）")
    print("  P101-A ×1.0→×2.0（低 lr 增强）")
    print()

    for tag, mult, desc in RUNS:
        cmd = [str(PY), "-u", str(SCRIPT), "--tag", tag,
               "--word-head-lowlr-mult", str(mult)] + COMMON
        print("=" * 88)
        print("▶ %s：%s" % (tag, desc))
        print("=" * 88, flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        for line in (r.stdout or "").splitlines():
            if any(k in line for k in ("低 lr", "里程碑", "ep0", "最佳",
                                       "收据", "Traceback", "❌")):
                print("  " + line)
        if r.returncode != 0:
            print("\n❌ %s 失败（exit=%d）" % (tag, r.returncode))
            print((r.stderr or "")[-1200:])
            return r.returncode

    print()
    print("=" * 88)
    print("两组完成")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())