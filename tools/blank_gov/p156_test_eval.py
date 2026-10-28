#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P156-test：TFNet 发布权重 零样本 在 test split 的最终验收（一次性，破例）

背景
  P156 在 dev 上给出 TFNet 零样本代价分解：224+RGB 33.92(门禁) / 160+RGB 36.81。
  本脚本把 160+RGB 零样本结果迁移到 test（对外宣称口径）。
  同时跑 224+RGB 作为 test 端门禁（官方 README 标 test 32.46）。

铁律（test 冻结，破例一次）
  1. 不在 test 上扫描任何超参；BLANK_BIAS 固定 0.0（与 p156 dev 一致）。
  2. 只评估两个点：224+RGB(门禁) + 160+RGB(目标)；不重选。
  3. 此为破例：原 P145 已用 test 名额（VAC 模型）；本次是不同模型(TFNet 零样本)，
     用户 2026-10-28 明确授权破例；跑完即归档，不再跑第二次。
  4. 复用 p145 的 test 数据路径；decode 用 p156 修复版（整段 logp 交 greedy_decode，
     不预截断 —— 防止旧 double-_conv_len 把解码帧压成 1/5 的 bug 重现）。

判据（写死，启动前）
  - 门禁：test 224+RGB 必须落在 |gap vs 32.46| <= 3pp；否则告警、不采信 160 数字、先排查。
  - 目标：test 160+RGB 即对外数字；报告时对照 dev 36.81，若 test/dev 差 >5pp 如实标注分布差异。
  - BLANK_BIAS 固定 0.0，绝不扫描。

用法
  python tools/blank_gov/p156_test_eval.py
"""
import json
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

SPLIT = "test"
CK = Path("/mnt/c/Users/su127/Desktop/中文手语识别/官方权重/"
          "TFNet-CE-CSL-CSLDaily-32.46.pth")
TEST_BASE = 32.46                      # 官方 README 标 test


def log(*a):
    print("[P156test %s] %s" % (time.strftime("%H:%M:%S"),
                                " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    dev = "cuda"
    labels = P.read_csv_split("%s.csv" % SPLIT)
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d；%s 样本 %d 条" % (wordSetNum, SPLIT, len(labels)))

    model = P.Net.moduleNet(1024, wordSetNum + 1, "TFNet",
                            torch.device(dev), "CE-CSL", True).to(dev)
    sd = torch.load(str(CK), map_location="cpu", weights_only=False,
                    mmap=True)["moduleNet_state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    P.OFFICIAL_PAD = True
    log("载入 TFNet 发布权重（strict 逐键吻合）")

    ls = torch.nn.LogSoftmax(dim=-1)
    rows = {}
    for img, rgb in [(224, True), (160, True)]:
        P.RGB_FIX = bool(rgb)
        _, tf = P.build_transforms(hflip=False, img_size=img)
        ds = P.RGBSeqDataset(SPLIT, labels, word2idx, tf, False)
        refs, hyps, ntok = [], [], 0
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(ds)):
                batch = P.collate([ds[i]])
                vid, _t, _tl, dl, _tl_len, sids, _ids = batch
                out = model(vid.to(dev), dl, False)
                logp = ls(out[0])          # [T,B,C] 卷积后长度，不预截断
                P.BLANK_BIAS = 0.0
                t = P.greedy_decode(logp, dl, idx2word)  # 内部 _conv_len(dl)
                hyps += t
                ntok += sum(len(q) for q in t)
                for b in sids:
                    refs.append(labels[b][1])
        o = evaluate(refs, hyps)
        gen = 100.0 * ntok / max(o["ref_tokens"], 1)
        k = "%d_%s" % (img, "RGB" if rgb else "BGR")
        rows[k] = {"WER": o["WER_official"], "gen": gen, "n": len(ds)}
        log("  img=%d %s  WER %6.2f%%  生成率 %5.1f%%  (%d条, %.0fs)"
            % (img, "RGB" if rgb else "BGR", o["WER_official"], gen,
               len(ds), time.time() - t0))
        if k == "224_RGB":
            gap = o["WER_official"] - TEST_BASE
            if abs(gap) > 3.0:
                log("🔴 门禁告警：test 224+RGB = %.2f，与官方 test 32.46 差 %+.2f pp"
                    % (o["WER_official"], gap))
                log("   ⇒ 数据/环境可能不一致，先排查再采信 160 数字")
            else:
                log("✅ 门禁通过：test 224+RGB = %.2f（官方 test 32.46，差 %+.2f pp）"
                    % (o["WER_official"], gap))

    print()
    log("──── test 最终验收（TFNet 零样本）────")
    log("  224+RGB(门禁)  %.2f%%" % rows["224_RGB"]["WER"])
    log("  160+RGB(目标)  %.2f%%  分辨率代价 %+.2f pp"
        % (rows["160_RGB"]["WER"],
           rows["160_RGB"]["WER"] - rows["224_RGB"]["WER"]))
    log("  dev 对照：160+RGB dev=36.81，test=%.2f，差 %+.2f pp"
        % (rows["160_RGB"]["WER"], rows["160_RGB"]["WER"] - 36.81))

    out = REPO / "artifacts/metrics/blank-gov/p156-test-final.json"
    out.write_text(json.dumps({
        "split": SPLIT, "ckpt": CK.name, "n_samples": len(labels),
        "test_base_224_rgb": TEST_BASE, "rows": rows,
        "dev_160_rgb": 36.81,
        "note": "test 一次性破例验收；TFNet 零样本 160 RGB；BLANK_BIAS=0.0 固定",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
