#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P156：在**官方 TFNet 权重**上量化「分辨率」与「通道」各值多少 pp

为什么这样做（省时间的关键）
  正常做法是重新训练一个 224 模型（24h）。但官方权重本身就是 224+RGB 训出来的，
  直接改推理侧的输入，就能量出这两项取舍的代价，**零训练成本、十几分钟**。

  代价类型：这是"推理时输入改变"的代价；而我们如果用 160/BGR **从头训练**，
  模型会适应 ⇒ 实际训练侧损失**小于**这里的数 ⇒ 本实验给出的是**上界**。

四个组合（224+RGB 为已知基线 33.92，此处复现以校验）
  224 + RGB  ← 官方原配（baseline）
  160 + RGB  ⇒ 分辨率代价
  224 + BGR  ⇒ 通道代价
  160 + BGR  ⇒ 我们的完整配置（两个取舍叠加）

用法
  python tools/blank_gov/p156_official_ablation.py [n_dev] [组合]
    组合可选 all / fast
    ⚠️ 做数据一致性门禁必须用 all —— fast 会跳过基线 224+RGB
       若只想看代价、数据已确认无误，可用 fast（省 ~8 分钟）
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402

# 权重路径：命令行第 3 参 > 环境变量 OFFICIAL_CKPT > 默认 > 自动搜索
_CK_ENV = os.environ.get("OFFICIAL_CKPT", "")
CK = Path(sys.argv[3] if len(sys.argv) > 3 else
          (_CK_ENV or "/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/"
                     "TFNet-CE-CSL-CSLDaily-32.46.pth"))
if not CK.exists():
    print("[P156] 权重不在 %s ⇒ 自动搜索…" % CK)
    for _base in (Path.cwd(), Path.cwd() / "官方权重",
                  Path.home(), Path.home() / "官方权重"):
        if not _base.exists():
            continue
        try:
            _hits = sorted(_base.glob("*TFNet-CE-CSL*.pth")) or \
                sorted(_base.glob("*/*TFNet-CE-CSL*.pth"))
        except Exception:
            _hits = []
        if _hits:
            CK = _hits[0]
            break
    print("[P156] 使用权重：%s%s" % (CK, "" if CK.exists() else "  ⚠️ 仍然找不到"))

N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MODE = sys.argv[2] if len(sys.argv) > 2 else "fast"
BASE = 33.92                      # 224+RGB（P152 实测）

if MODE == "fast":
    COMBOS = [(160, True), (224, False), (160, False)]   # 跳过 224+RGB
else:
    COMBOS = [(224, True), (160, True), (224, False), (160, False)]


def log(*a):
    print("[P156 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def run(model, word2idx, idx2word, labels, img, rgb_fix):
    P.RGB_FIX = bool(rgb_fix)
    _, tf = P.build_transforms(hflip=False, img_size=img)
    ds = P.RGBSeqDataset("dev", labels, word2idx, tf, False)
    if N_DEV:
        ds.items = ds.items[:N_DEV]
    ls = torch.nn.LogSoftmax(dim=-1)
    refs, hyps, ntok = [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            batch = P.collate([ds[i]])
            vid, _t, _tl, dl, _tl_len, sids, _ids = batch
            out = model(vid.to("cuda"), dl, False)
            lp = ls(out[0])[:, 0, :]
            L = min(P._conv_len(int(dl.view(-1)[0])), lp.shape[0])
            x = lp[:L].to("cpu", torch.float32)
            P.BLANK_BIAS = 0.0
            t = P.greedy_decode(x.unsqueeze(1), torch.LongTensor([[L]]), idx2word)
            hyps += t
            ntok += sum(len(q) for q in t)
            for b in sids:
                refs.append(labels[b][1])
    o = evaluate(refs, hyps)
    g = 100.0 * ntok / max(o["ref_tokens"], 1)
    log("  img=%d %s  WER %6.2f%%  生成率 %5.1f%%  (%.0fs)"
        % (img, "RGB" if rgb_fix else "BGR", o["WER_official"], g, time.time() - t0))
    return {"WER": o["WER_official"], "gen_rate": g}


def main() -> int:
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    model = P.Net.moduleNet(1024, wordSetNum + 1, "TFNet",
                            torch.device("cuda"), "CE-CSL", True).to("cuda")
    sd = torch.load(str(CK), map_location="cpu", weights_only=False,
                    mmap=True)["moduleNet_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    P.OFFICIAL_PAD = True
    labels = P.read_csv_split("dev.csv")
    log("官方 TFNet，dev %d 条，组合 %s" % (len(labels) if not N_DEV else N_DEV, COMBOS))

    rows = {}
    for img, rgb in COMBOS:
        k = "%d_%s" % (img, "RGB" if rgb else "BGR")
        rows[k] = run(model, word2idx, idx2word, labels, img, rgb)
        if k == "224_RGB":
            gap = rows[k]["WER"] - BASE
            if abs(gap) > 2.0:
                log("🔴 门禁告警：基线 224+RGB = %.2f，与官方 33.92 差 %+.2f pp"
                    % (rows[k]["WER"], gap))
                log("   ⇒ 数据/环境可能不一致，先排查再跑 D3/D4（否则可能白跑）")
            else:
                log("✅ 门禁通过：224+RGB = %.2f（官方 33.92，差 %+.2f pp）"
                    % (rows[k]["WER"], gap))

    print()
    log("──── 官方权重上的代价分解（dev，greedy，语料级）────")
    log("  %-12s %10s %12s" % ("组合", "WER", "vs 224+RGB"))
    log("  %-12s %10.2f %12s" % ("224+RGB(基线)", BASE, "—"))
    for k, v in rows.items():
        log("  %-12s %10.2f %+12.2f" % (k, v["WER"], v["WER"] - BASE))
    print()
    if "160_RGB" in rows:
        log("  ⇒ 分辨率代价（224→160，RGB 下）：%+.2f pp" % (rows["160_RGB"]["WER"] - BASE))
    if "224_BGR" in rows:
        log("  ⇒ 通道代价（RGB→BGR，224 下）：%+.2f pp" % (rows["224_BGR"]["WER"] - BASE))
    if "160_BGR" in rows:
        log("  ⇒ 两者叠加（= 我们的取舍）：%+.2f pp" % (rows["160_BGR"]["WER"] - BASE))
    log("  ⚠️ 这是「推理时换输入」的上界；从头训练会适应 ⇒ 训练侧真实损失更小")

    out = REPO / "artifacts/metrics/blank-gov/p156-official-ablation.json"
    out.write_text(json.dumps({
        "ckpt": CK.name, "module": "TFNet", "n": len(labels) if not N_DEV else N_DEV,
        "baseline_224_rgb": BASE, "rows": rows, "mode": MODE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
