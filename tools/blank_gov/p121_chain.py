r"""P121: 等待 P119 收据 → 自动启动解混淆实验（0B）

## 为什么做这个实验（它是当前最高优先级）
P93 基线与 P100/CTRL/A 之间有**两个**变量同时变化：
| | P93 | P100/CTRL/A |
|---|---|---|
| word_head | 0.0 | 0.3 |
| warmup_ep | 无 | 3 |
⇒ 实测的 −2.0 pp 只能算在「组合」上，**不能单独归给词级头**。

本实验 `word_head=0.0 + warmup_ep=3 + no_hflip`，与 P93 **唯一差异 = warmup**。

🔴 **必须带 `--no-hflip`**：P93 是 `no_hflip=True`（已核实其收据）。
   而 P119 正在跑的是 **hflip 开启**组。
   若不显式关闭，本组相对 P93 会多出「hflip」这一个变量
   ⇒ 变成三方混淆，得不偿失。
   （hflip 本身的效应由 P119 单独判定，见 stage-P119 报告）
⇒ 与 P93 配对 = warmup 的净效应
⇒ 与 P100/CTRL/A 配对 = word_head 的净效应
⇒ 两者相加应 ≈ −2.0 pp（可作一致性校验）

## 判据（启动前写死，防作弊 P84）
噪声基准 std = **1.556 pp**（P108 实测）
- |mean| < 1.556 ⇒ 不可判
- 1.556 ~ 3.1 ⇒ 强提示
- \> 3.1 ⇒ 可判

⚠️ 必须报 epoch 级配对 mean±std + 胜率，禁止只比 best WER。

## 进程保活（P113 铁律）
本脚本**必须**用 background task 模式启动，不要 `nohup setsid`。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
PY = REPO / "venv/bin/python"
SCRIPT = REPO / "tools/blank_gov/p78_train_official_tfnet.py"
METRICS = REPO / "artifacts/metrics/blank-gov"
LOG = REPO / "artifacts/logs_p121_disamb.log"

WAIT_TAG = "p119-hflip"
NEXT_TAG = "p121-disamb"
POLL_SEC = 300


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def receipt_exists(tag):
    """按训练脚本的真实收据命名检查（MEMORY：读字段前先核对命名）"""
    p = METRICS / ("%s-official-tfnet.json" % tag)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log("收据解析失败 %s: %s" % (p, e))
        return None
    h = d.get("history", [])
    if not h:
        return None
    return d


def main():
    log("=== P121 链条启动 ===")
    log("阶段1：等 %s 收据" % WAIT_TAG)
    t0 = time.time()
    waited = False

    while True:
        d = receipt_exists(WAIT_TAG)
        if d is not None:
            h = d["history"]
            log("%s 收据已出：%d epoch，best WER %.2f %%"
                % (WAIT_TAG, len(h), min(e["dev_wer_official"] for e in h)))
            break
        # 训练进程死了但收据还没出 → 报错退出，不要静默等
        alive = subprocess.run(
            ["pgrep", "-f", "p78_train_official_tfnet"],
            capture_output=True).returncode == 0
        if not alive:
            # 可能是刚结束还没写盘，再等一轮
            if waited:
                log("❌ 无收据且训练进程已退出，不再等待")
                return 1
            time.sleep(60)
            if receipt_exists(WAIT_TAG) is None:
                log("❌ 训练进程消失且无收据，链条中止")
                return 1
        if int(time.time() - t0) % 1800 < POLL_SEC:
            log("等待中… 已 %.1f h" % ((time.time() - t0) / 3600))
        time.sleep(POLL_SEC)

    log("=== 阶段2：启动解混淆实验 %s ===" % NEXT_TAG)
    log("配置：word_head=0.0  +  warmup_ep=3  （与 P93 唯一差异 = warmup）")

    cmd = [str(PY), "-u", str(SCRIPT),
           "--tag", NEXT_TAG,
           "--module", "VAC",
           "--img-size", "160",
           "--amp-bf16",
           "--total-epochs", "20",
           "--epochs", "20",           # 🔴 必须同时传：--total-epochs 不限制循环（P119 事故）
           "--word-head", "0.0",      # ← 唯一改动：关掉词级头
           "--warmup-ep", "3",         # ← 保留 warmup
           "--no-hflip",                # 🔴 必须加：P93 基线是 no_hflip=True
           ]
    log("命令：%s" % " ".join(cmd))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    log("%s 退出码 %d" % (NEXT_TAG, rc))
    return rc


if __name__ == "__main__":
    sys.exit(main())
