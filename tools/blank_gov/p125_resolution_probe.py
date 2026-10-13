"""P125：分辨率核对 —— 160 vs 224（官方 RandomCrop(224)）。

背景（2026-10-13 静态核对发现的两处与官方的系统性偏离之一）：
- 官方 `Train.py:76` 训练 `RandomCrop(224)` / 测试 `CenterCrop(224)`，源帧 256×256
  ⇒ 保留 **87.5 %** 区域
- 我们全程用 `--img-size 160`，源帧同为 256×256
  ⇒ 只保留 **62.5 %** 区域；中心裁掉四周 48 px 的边框
  手语的手常在画面边缘 ⇒ **可能系统性裁掉手部**

本脚本的目标：一个**几乎免费**的预判，用来决定要不要花约 2 倍成本重训 224。

方法：同一个 checkpoint、同一批 dev（全量 515），只换评估时的裁剪尺寸。
- A：CenterCrop(160)（= 训练口径，基准）
- B：CenterCrop(224)（官方口径）
- C：CenterCrop(192)（中间点，看单调性）

⚠️ 解读限制（必须写清，否则会误判）：
模型是用 160 训练的 ⇒ B 存在 **train-test 空间尺度不匹配**
（训练时 conv 末端特征图 5×5，评估时 7×7；且评估看到了训练时从未见过的边缘区域）。
⇒ B 是在**不利条件**下测的：
  - B 明显更好（≥3 pp）⇒ 极强信号：信息量差异压过了不匹配，值得重训 224
  - B 略好 / 持平 / 略差 ⇒ 仍值得重训（重训后不匹配消失，收益只会更大）
  - B 大幅变差（>5 pp）⇒ 无法区分「不匹配」与「分辨率本身有害」，需重训才能定论
⇒ 本探针只能给出「要不要重训」的建议，**不能**直接定论 224 的净效应。

判据（预先写死）：
  WER(224) − WER(160) ≤ −3.0 pp  ⇒ 强证据：立即安排 224 重训
  −3.0 < Δ ≤ 0                   ⇒ 弱证据：安排 224 重训（优先级次之）
  Δ > 0                          ⇒ 不能直接否定；仍建议重训一次以分离两因素
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

# 当前最佳（55.84 @ ep39）。⚠️ 它是 --official-pad 训练的 ⇒ 评估必须开补帧 + 用 data_len 截断
CK = REPO / "artifacts/checkpoints/p123-official55bf16-best.pt"
SIZES = [160, 192, 224]
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515


def log(*a):
    print("[P125 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main():
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    # 🔴 P123 是开着官方补帧训练的 ⇒ 评估必须同口径，且截断用 data_len（MEMORY 19）
    P.OFFICIAL_PAD = True

    # 主干恢复：键名是 model_state，必须 strict=True（MEMORY 19 —— 曾因键名写错静默没载入）
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "ckpt 键名变了：%s" % list(ck)[:8]
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s，strict=True OK）" % (CK.name, ck.get("epoch")))

    ls = torch.nn.LogSoftmax(dim=-1)

    def run_eval(size: int):
        _, test_tf = P.build_transforms(hflip=False, img_size=size)
        ds = P.RGBSeqDataset("dev", labels_dv, word2idx, test_tf, False)
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
                logp = ls(out[0])                      # [T,1,C]
                hyps += P.greedy_decode(logp, dl, idx2word)   # 🔴 截断必须用 data_len
                for b in sids:
                    refs.append(labels_dv[b][1])
        o = evaluate(refs, hyps)
        log("CenterCrop(%-3d)  dev=%d  WER=%6.2f%%  ins=%d del=%d sub=%d ref=%d  (%.0fs)"
            % (size, len(ds), o["WER_official"],
               o.get("ins", -1), o.get("del", -1), o.get("sub", -1),
               o.get("n", o.get("ref_tokens", -1)), time.time() - t0))
        return o

    outs = {}
    for s in SIZES:
        outs[s] = run_eval(s)

    base = outs[160]["WER_official"]
    print()
    log("──── 结论 ────")
    for s in SIZES:
        if s == 160:
            continue
        d = outs[s]["WER_official"] - base
        log("  %d vs 160：Δ = %+.2f pp" % (s, d))

    d224 = outs[224]["WER_official"] - base
    if d224 <= -3.0:
        verdict = "强证据：224 在不匹配条件下仍大幅更好 ⇒ 立即安排 224 重训"
    elif d224 <= 0:
        verdict = "弱证据：224 持平/略好 ⇒ 安排 224 重训（优先级次之）"
    else:
        verdict = ("224 更差（%+0.2f pp）⇒ 无法区分「尺度不匹配」与「分辨率本身有害」，"
                   "仍建议重训一次 224 以分离两因素" % d224)
    log("判定：%s" % verdict)

    out = REPO / "artifacts/metrics/blank-gov/p125-resolution-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "P125 分辨率核对（我们 160 vs 官方 224）",
        "ckpt": str(CK), "epoch": ck.get("epoch"),
        "n_dev": MAX_DEV, "official_pad": True, "truncation": "data_len",
        "source_frame": "256x256（与官方一致）",
        "results": {str(s): outs[s] for s in SIZES},
        "delta_vs_160_pp": {str(s): round(outs[s]["WER_official"] - base, 2)
                            for s in SIZES if s != 160},
        "verdict": verdict,
        "caveat": "模型以 160 训练 ⇒ 224/192 评估存在 train-test 空间尺度不匹配",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)


if __name__ == "__main__":
    main()
