"""P129：测试裁切**垂直位置**扫描 —— 检验 P128 的「重心偏下」发现。

背景与动机（必须连起来看）：
  - P128：运动能量重心在 y≈157，而 CenterCrop(160) 保留 [48,208]（中心 128）
    ⇒ **系统性切掉下部**
  - P127：**保留更多区域（224）反而差 13.29 pp**（SNR 8.54）
    ⇒ 说明问题不在「裁掉的面积」，而在「裁掉的**位置**」
    ⇒ 所以不能靠放大分辨率解决，要**平移**裁切窗口

方法：同一个 checkpoint（P123 best ep39，当前最佳 55.84）、全量 dev 515，
      只平移**测试**裁切窗口的 y 起点（x 不变），扫描若干偏移：
        -32 / -16 / 0（官方基准）/ +16 / +29（=重心修正量）/ +48
      训练侧完全不动（RandomCrop 本来就是随机的、覆盖全图含下部）。

⚠️ 已知答案校验：shift=0 必须复现 **55.84**（P123 收据值），否则路径不一致。
⚠️ 解读：这是 train-test 位置不一致的**诊断**，不是最终方案。
      若某个 shift 明显更好 ⇒ 说明固定裁切位置确实是个真实损失来源，
      值得把它做成训练配置（P130）去配对验证。

判据（预先写死）：
  best_shift 使 WER 比 shift=0 低 **≥ 3.0 pp** ⇒ 判活 ⇒ P130（训练时采纳该 shift）
  1.0 ~ 3.0 pp                                ⇒ 弱证据，仍建议 P130
  < 1.0 pp                                    ⇒ 不判活 ⇒ 转 P131（去掉 TemporalRescale）
"""
from __future__ import annotations

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

CK = REPO / "artifacts/checkpoints/p123-official55bf16-best.pt"
IMG = 160
SHIFTS = [-32, -16, 0, 16, 29, 48]
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515


def log(*a):
    print("[P129 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main() -> int:
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    # P123 是开官方补帧训练的 ⇒ 评估必须同口径、截断用 data_len（MEMORY 19）
    P.OFFICIAL_PAD = True
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "ckpt 键名变了：%s" % list(ck)[:8]
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s，strict=True OK）" % (CK.name, ck.get("epoch")))

    ls = torch.nn.LogSoftmax(dim=-1)

    def run(shift: int):
        _, dv_tf = P.build_transforms(hflip=False, img_size=IMG,
                                      y_shift=shift)
        ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
        if MAX_DEV:
            ds.items = ds.items[:MAX_DEV]
        hyps, refs = [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(ds)):
                batch = P.collate([ds[i]])
                vid, _tgt, _tl, dl, _tl_len, sids, _ids = batch
                vid = vid.to(dev)
                out = model(vid, dl, False)
                logp = ls(out[0])
                hyps += P.greedy_decode(logp, dl, idx2word)   # 截断用 data_len
                for b in sids:
                    refs.append(labels_dv[b][1])
        o = evaluate(refs, hyps)
        log("y_shift %+4d（裁 y∈[%3d,%3d]）  WER=%6.2f%%  ref=%d  (%.0fs)"
            % (shift, 48 + shift, 208 + shift, o["WER_official"],
               o.get("n", o.get("ref_tokens", -1)), time.time() - t0))
        return o

    outs = {}
    for s in SHIFTS:
        outs[s] = run(s)

    base = outs[0]["WER_official"]
    ok = abs(base - 55.84) < 0.02
    print()
    log("──── 结论 ────")
    for s in SHIFTS:
        d = outs[s]["WER_official"] - base
        mark = "  ← 基准(官方)" if s == 0 else (
            "  ★最优" if outs[s]["WER_official"] == min(
                outs[k]["WER_official"] for k in SHIFTS) else "")
        log("  shift %+4d : %6.2f%%   Δ %+6.2f pp%s" % (s, outs[s]["WER_official"], d, mark))

    best_s = min(SHIFTS, key=lambda k: outs[k]["WER_official"])
    gain = base - outs[best_s]["WER_official"]      # 正=改善
    if gain >= 3.0:
        verdict = ("判活：shift=%+d 改善 %.2f pp ⇒ P130（训练时采纳该裁切位置）"
                   % (best_s, gain))
    elif gain >= 1.0:
        verdict = ("弱证据：shift=%+d 改善 %.2f pp ⇒ 仍建议 P130" % (best_s, gain))
    else:
        verdict = ("不判活：最优 shift=%+d 仅改善 %.2f pp ⇒ 转 P131"
                   "（去掉 TemporalRescale）" % (best_s, gain))
    log("  判定：%s" % verdict)
    log("  已知答案校验：shift=0 应=55.84 ⇒ %s（实测 %.2f）"
        % ("✅ 通过" if ok else "❌ 不一致", base))

    out = REPO / "artifacts/metrics/blank-gov/p129-cropshift-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "P129 测试裁切垂直位置扫描（检验 P128 重心偏下）",
        "ckpt": str(CK), "epoch": ck.get("epoch"),
        "n_dev": MAX_DEV, "img_size": IMG, "shifts": SHIFTS,
        "official_pad": True, "truncation": "data_len",
        "results": {str(s): outs[s] for s in SHIFTS},
        "baseline_shift0": base,
        "delta_vs_shift0_pp": {str(s): round(outs[s]["WER_official"] - base, 2)
                               for s in SHIFTS},
        "best_shift": best_s, "gain_pp": round(gain, 2),
        "known_answer_check_passed": ok,
        "verdict": verdict,
        "design_note": "只改测试裁切位置，训练侧保持官方 RandomCrop；"
                       "所有 crop 都是 160x160 ⇒ 无尺度不匹配",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
