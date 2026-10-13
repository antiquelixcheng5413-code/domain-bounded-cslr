"""P126：10-crop TTA —— 判定「CenterCrop(160) 裁掉的边缘有没有判别信息」。

P125 的结论有混淆，不能直接采信：
    CenterCrop(160) 55.84 / (192) 59.08 / (224) 64.95  ⇒ 尺寸越大越差
但模型是**用 160 训练的** ⇒ 192/224 评估混入了 **train-test 空间尺度不匹配**
（resnet 末端特征图 5×5 vs 6×6 vs 7×7）。
⇒ 那 9.11 pp 既可能是"224 有害"，也可能只是"没见过这个尺度"。**无法定论。**

本探针去掉这个混淆：
    **所有 crop 都是 160×160**（与训练尺度**完全一致**），只换**位置**。
    - A：单 center crop（基准，应复现 55.84）
    - B：10-crop 平均（4 角 + 中心，各含 hflip）
⇒ 没有尺度不匹配，只有位置覆盖差异。

判据（预先写死）：
    Δ(B − A) ≤ −3.0 pp  ⇒ 边缘含大量判别信息 ⇒ **224 重训值得**（立即安排）
                          （集成本身通常只值 1~2 pp，−3 已超出该量级）
    −3.0 < Δ ≤ −1.0 pp  ⇒ 弱证据，边缘有些信息 ⇒ 仍建议重训 224
    Δ > −1.0 pp         ⇒ 边缘基本无判别信息 ⇒ **160 不是瓶颈**，
                          该从别处找 10.74 pp 的缺口（如 TemporalRescale、数据侧）

⚠️ 注意：A 必须复现 55.84（P123 best ep39 收据值），否则说明走的路径与训练不一致。
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
CROP = 160           # 与训练尺度一致
FULL = 256           # 源帧尺寸
SUB_BATCH = 2        # 8GB 卡：10 个 crop 一次性前向会 OOM ⇒ 分批累积
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515


def log(*a):
    print("[P126 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def build_model(wordSetNum, dev):
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    assert "model_state" in ck, "ckpt 键名变了：%s" % list(ck)[:8]
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    return model, ck


def ten_crops(vid):
    """vid: [T,3,256,256] → 10 个 [T,3,160,160]（4 角 + 中心，各含 hflip）。

    尺度与训练**完全一致**，只有位置/翻转不同 ⇒ 无尺度不匹配。
    """
    s, f = CROP, FULL
    off = (f - s) // 2                      # 48（center crop 的偏移）
    pos = [(0, 0), (0, f - s), (f - s, 0), (f - s, f - s), (off, off)]
    out = []
    for (y, x) in pos:
        c = vid[:, :, y:y + s, x:x + s]
        out.append(c)
        out.append(torch.flip(c, dims=[-1]))  # hflip（最后一维是 W）
    return torch.stack(out, dim=0)           # [10,T,3,s,s]


def main():
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    # P123 是开官方补帧训练的 ⇒ 评估同口径，截断用 data_len（MEMORY 19）
    P.OFFICIAL_PAD = True
    model, ck = build_model(wordSetNum, dev)
    log("载入 %s（epoch %s，strict=True OK）" % (CK.name, ck.get("epoch")))

    # 🔴 关键：不做任何 crop，保留 256×256 全图，crop 在 tensor 上做
    _, tf_keep = P.build_transforms(hflip=False, img_size=FULL)
    ds = P.RGBSeqDataset("dev", labels_dv, word2idx, tf_keep, False)
    if MAX_DEV:
        ds.items = ds.items[:MAX_DEV]
    log("dev %d 条（保留 %dx%d 全图，crop 在 tensor 层做）" % (len(ds), FULL, FULL))

    ls = torch.nn.LogSoftmax(dim=-1)

    def run(mode: str):
        hyps, refs = [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(ds)):
                batch = P.collate([ds[i]])
                vid, _tgt, _tl, dl, _tl_len, sids, _ids = batch
                vid = vid.to(dev)                       # [1,T,3,256,256]
                v = vid[0]                              # [T,3,256,256]
                if mode == "center":
                    crops = ten_crops(v)[8:9]           # 第 9 个 = center（未翻转）
                else:
                    crops = ten_crops(v)                # [10,T,3,160,160]
                # 🔴 10 个 crop 一次性前向会 OOM（8GB 卡）⇒ 分批累积再平均
                chunks = []
                for j in range(0, crops.size(0), SUB_BATCH):
                    cj = crops[j:j + SUB_BATCH]
                    dlb = dl.repeat(cj.size(0), 1)      # dataLen 必须 CPU [b,1]
                    out = model(cj, dlb, False)
                    chunks.append(ls(out[0]).float())   # [T, b, C]
                logp = torch.cat(chunks, dim=1)         # [T, 10 or 1, C]
                if mode != "center":
                    logp = logp.mean(dim=1, keepdim=True)   # 平均 10 个视角
                hyps += P.greedy_decode(logp, dl, idx2word)  # 截断用 data_len
                for b in sids:
                    refs.append(labels_dv[b][1])
        o = evaluate(refs, hyps)
        log("%-10s dev=%d  WER=%6.2f%%  ref=%d  (%.0fs)"
            % (mode, len(ds), o["WER_official"],
               o.get("n", o.get("ref_tokens", -1)), time.time() - t0))
        return o

    oA = run("center")
    print()
    oB = run("10crop")

    d = oB["WER_official"] - oA["WER_official"]
    if d <= -3.0:
        verdict = ("边缘含大量判别信息（Δ=%+.2f pp，远超集成通常的 1~2 pp）"
                   "⇒ 立即安排 224 重训" % d)
    elif d <= -1.0:
        verdict = ("弱证据：边缘有些信息（Δ=%+.2f pp）⇒ 仍建议重训 224" % d)
    else:
        verdict = ("边缘基本无判别信息（Δ=%+.2f pp）⇒ 160 不是瓶颈，"
                   "缺口要从别处找" % d)
    print()
    log("──── 结论 ────")
    log("  center-1-crop : %.2f%%" % oA["WER_official"])
    log("  10-crop TTA   : %.2f%%" % oB["WER_official"])
    log("  Δ = %+.2f pp" % d)
    log("  判定：%s" % verdict)

    ok = abs(oA["WER_official"] - 55.84) < 0.02
    log("已知答案校验：center-1-crop 应=55.84 ⇒ %s" % ("✅ 通过" if ok else "❌ 不一致"))

    out = REPO / "artifacts/metrics/blank-gov/p126-tta-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "P126 10-crop TTA：边缘区域是否含判别信息",
        "ckpt": str(CK), "epoch": ck.get("epoch"),
        "n_dev": len(ds), "crop_size": CROP, "full_size": FULL,
        "official_pad": True, "truncation": "data_len",
        "A_center_1crop": oA, "B_10crop_tta": oB,
        "delta_B_minus_A_pp": round(d, 2),
        "known_answer_check_passed": ok,
        "verdict": verdict,
        "design_note": "所有 crop 均 160x160（与训练尺度一致）⇒ 无空间尺度不匹配，"
                       "只有位置覆盖差异；baseline 为集成效应（通常 1~2 pp）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)


if __name__ == "__main__":
    main()
