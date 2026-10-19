#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GPU 能力探测 —— 决定要不要往这台机器派 30~50 小时任务之前，先花 1 分钟跑这个。

回答三个问题：
  1. PyTorch 到底认不认这张卡（N 卡走 CUDA / A 卡走 ROCm-HIP）
  2. 支不支持 bf16 / fp16（这决定要不要去掉 --amp-bf16）
  3. 有多快 —— 用固定微基准测出「每轮大约几小时」，跟另一台机器直接比数字

用法
  ./venv/bin/python -u tools/blank_gov/check_gpu_cap.py

⚠️ 请把**完整输出**发回群里。两台机器都跑一遍，数字直接对比即可。
"""
import time

import torch
import torch.nn as nn

WARMUP = 3
ITERS = 10


def hdr(s):
    print("\n" + "=" * 60)
    print(" " + s)
    print("=" * 60)


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS


def bench_conv(device, autocast):
    """resnet18 逐帧前向+反向：模拟 VAC 的主干开销。"""
    from torchvision.models import resnet18
    m = resnet18(weights=None).to(device)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x = torch.randn(16, 3, 224, 224, device=device)
    tgt = torch.zeros(16, dtype=torch.long, device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with autocast():
            y = m(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        opt.step()

    return timed(step), 16          # 秒/次, 每次 16 帧


def bench_lstm(device, autocast):
    """双向 LSTM：模拟时序头开销。"""
    m = nn.LSTM(1024, 1024, num_layers=2, bidirectional=True).to(device)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x = torch.randn(30, 8, 1024, device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with autocast():
            y, _ = m(x)
        y.float().pow(2).mean().backward()
        opt.step()

    return timed(step), 8           # 秒/次, 每次 8 条样本(各 30 帧)


def main() -> int:
    hdr("1. 基本环境")
    print("  torch        : %s" % torch.__version__)
    print("  torch.version.cuda : %s" % (torch.version.cuda or "None"))
    print("  torch.version.hip  : %s" % (torch.version.hip or "None ← 不是 ROCm 构建"))
    hip = bool(torch.version.hip)

    if not torch.cuda.is_available():
        print("\n  ❌ torch.cuda.is_available() = False")
        print("     ⇒ PyTorch 没找到可用计算后端。")
        try:
            import torch_directml  # noqa: F401
            print("     检测到 torch-directml，但我们的脚本里写死了 .to('cuda')，")
            print("     DirectML 走的是另一套 device API ⇒ 需要改代码，**不建议**。")
        except ImportError:
            pass
        print("     A 卡请确认：装的是 **ROCm 版 PyTorch**（Linux 或 WSL2），")
        print("     且显卡在 ROCm 支持列表里。")
        return 1

    print("  ✅ torch.cuda.is_available() = True")
    print("  设备          : %s" % torch.cuda.get_device_name(0))
    print("  显存          : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("  后端          : %s" % ("ROCm/HIP（A 卡）" if hip else "CUDA（N 卡）"))

    hdr("2. 精度支持（决定要不要去掉 --amp-bf16）")
    ok = {}
    for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        try:
            a = torch.randn(512, 512, device="cuda", dtype=dt)
            b = torch.randn(512, 512, device="cuda", dtype=dt)
            c = a @ b
            torch.cuda.synchronize()
            ok[name] = "✅ 支持"
        except Exception as e:                       # noqa: BLE001
            ok[name] = "❌ 不支持（%s）" % str(e).split("\n")[0][:60]
        print("  %-6s %s" % (name, ok[name]))

    hdr("3. 速度（固定微基准，两台机器直接比数字）")
    print("  说明：下面跑的是 resnet18 与双向 LSTM 各一个固定的前向+反向，")
    print("        与真实训练不完全等价，但足以横向比较两张卡的量级。\n")

    for label, use_amp in (("bf16 AMP", True), ("fp32（无 AMP）", False)):
        if use_amp and not ok.get("bf16", "").startswith("✅"):
            print("  [%s] 跳过：该卡不支持 bf16" % label)
            continue
        try:
            ac = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.autocast("cuda", enabled=False)
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:                        # noqa: BLE001
                pass
            t_c, n_f = bench_conv("cuda", ac)
            t_l, n_s = bench_lstm("cuda", ac)
            try:
                peak = torch.cuda.max_memory_allocated() / 1e9
            except Exception:                        # noqa: BLE001
                peak = float("nan")

            fps = n_f / t_c                          # 帧/秒
            sec_lstm = t_l / n_s                     # 秒/样本(时序头)
            sec_sample = 30.0 / fps + sec_lstm       # 一条样本约 30 帧
            h_per_epoch = 4973 * sec_sample / 3600.0
            print("  [%s]" % label)
            print("    主干   : %.1f 帧/秒" % fps)
            print("    时序头 : %.4f 秒/样本" % sec_lstm)
            print("    单样本 : %.2f 秒  （30 帧 + 时序头）" % sec_sample)
            print("    峰值显存: %.2f GB" % peak)
            print("    ⇒ 预估 1 轮 ≈ %.1f 小时   |  55 轮 ≈ %.0f 小时" % (h_per_epoch, h_per_epoch * 55))
            print()
        except RuntimeError as e:
            msg = str(e).split("\n")[0][:110]
            print("  [%s] ❌ 跑不起来：%s" % (label, msg))
            if "out of memory" in str(e).lower():
                print("      ⇒ 显存不够。这条路线基本判死，别派任务。")
            print()

    hdr("4. 怎么判读")
    print("""  把两台机器的输出一起看：

  · 若 A 卡「1 轮 ≈」在 N 卡的 **1.5 倍以内**   ⇒ 可以跑，接 D3/D4
  · 若在 **1.5~3 倍**                        ⇒ 能跑但很慢，只派一个任务，并且
    必须接受「两机精度不一致」（A 卡多半只能 fp32）
  · 若 **>3 倍 或 直接 OOM**                 ⇒ 别派训练任务。
    改做不占 GPU 的事：数据一致性核对（check_env.py）、
    误差构成分析、文档整理。

  🔴 关于精度不一致（重要，别忽略）：
     bf16 与 fp32 的**最终精度通常只差噪声量级**（bf16 保留 fp32 的指数位，
     不需要 loss scaling），所以「效果不一样」的担心基本不成立。
     真正的问题是另外两条：
       ① fp32 显存占用约为 bf16 的 2 倍 ⇒ 8GB 上更容易 OOM；
       ② 跨卡 + 跨精度会让 D3/D4 的数字混进额外变量，
          做减法时要在报告里注明。
""")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
