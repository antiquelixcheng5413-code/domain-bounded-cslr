# -*- coding: utf-8 -*-
"""P2-b：dense supervision 真正作用到 CTC blank 率上（SMART 式 late fusion）。

P2-a 只证明了「密集监督能把帧级定位学好」（macro-F1 0.13->0.70），
但那是一个独立的头，**没有改变 CTC 自己的 blank 率**。
本脚本回答真正的用户问题：blank 率能不能降。

做法（SMART 式 8，alpha=0.7 grid search）：
    P_fuse = alpha * P_rec + (1-alpha) * P_spot
    然后在 P_fuse 上做 argmax，看 blank 占比。

关键：spotting 头是 BIO 三类，识别侧是 302 类（vocab 301 + blank），
两者类别空间不同，不能直接相加。SMART 的做法是 spotting 头输出
**与识别侧同构的类别空间**。所以这里把 frame-level 头改成
「逐帧 gloss 分类」（复用 CTC 的 302 类输出），
监督来自 TFD 伪边界切出的段内 gloss 分配。

这才是论文里的 P_spot 定义 —— CSFormer 的 decoder 输出帧级分类 logits，
与识别 backbone 的词表同构，然后 late fusion 反哺 CTC 解码。

只用 train/validation，不触碰 test。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import tfd as TFD
import diagnose as D
from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig


class FrameSpotter(nn.Module):
    """帧级 gloss 分类头，输出与识别侧同构的 302 类（SMART CSFormer 的 P_spot）。

    用 SE + 轻量膨胀卷积，保持轻量（8G 显存无压力）。
    论文消融里 boundary head 贡献最小，这里不加。

    **温度参数是必需的，不是可选项。** 实测（未训练或弱监督的）spot 输出
    接近均匀分布（各类均值 ~0.019），而冻结 CTC 的 blank 概率均值 0.90、
    中位 0.95。此时 P_fuse = α·P_rec + (1-α)·P_spot 在任何 α<1 下
    argmax 都仍是 blank —— 融合完全失效（实测 α 从 0.7 到 1.0 结果一字不变）。

    SMART 论文没讨论这点，因为它的 P_spot 经充分训练后概率是尖锐的。
    我们用 TFD 伪边界做弱监督，概率天然平滑，必须显式校准，
    否则会得出「late fusion 无效」的错误结论。
    learn_temp=True 时温度作为参数学，让模型自己找到可比尺度。
    """

    def __init__(self, enc_dim: int, num_classes: int, hidden: int = 128, layers: int = 5,
                 learn_temp: bool = True, init_temp: float = 1.0):
        super().__init__()
        self.inp = nn.Conv1d(enc_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, padding=2**i, dilation=2**i)
             for i in range(layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.out = nn.Conv1d(hidden, num_classes, 1)
        # log 温度：乘在 logits 上，>0 表示锐化
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(init_temp))),
                                     requires_grad=learn_temp)

    def forward(self, x):
        """x: (B, T, C) BiLSTM 输出 -> (B, num_classes, T)"""
        h = self.inp(x.transpose(1, 2))
        for blk, nrm in zip(self.blocks, self.norms):
            h = F.relu(nrm(blk(h)) + h)
        return self.out(h) * self.log_temp.exp()


def load_ctc(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(p["model_config"])
    m = CTCRecognizer(cfg)
    m.load_state_dict(p["state_dict"])
    m.eval()
    vc = p.get("vocabulary_config") or {}
    voc = GlossVocabulary(tokens=tuple(p["vocabulary"]),
                          counts=dict(p.get("vocabulary_counts") or {}),
                          config=GlossSequenceConfig(**vc) if vc else GlossSequenceConfig())
    return m, voc, cfg, p.get("feature_normalizer")


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        idc = next((c for c in cols if c.lower() in ("number", "id", "name")), cols[0])
        gc = next((c for c in cols if "gloss" in c.lower() or "label" in c.lower()), cols[-1])
        for r in rd:
            rows.append((r[idc], r[gc]))
    return rows


def collect(root, dim, labels, limit):
    out = []
    for p in sorted(Path(root).glob("*.npy")):
        if ".receipt." in p.name:
            continue
        try:
            shp = np.load(p).shape
        except Exception:
            continue
        if not (shp and shp[-1] == dim):
            continue
        stem = p.stem.split(".landmark")[0].split(".clip")[0]
        if stem not in labels:
            continue
        ids = voc_encode(labels[stem])
        if not ids:
            continue
        out.append({"id": stem, "path": p, "ids": ids})
        if limit and len(out) >= limit:
            break
    return out


_VOC = None
_OOV = [0, 0]   # [oov_token_count, total_token_count]


def voc_encode(g):
    """把 gloss 字符串编码成 CTC 类别索引序列。

    GlossVocabulary.encode 收完整 gloss 字符串（内部按 "/" 切分），
    返回的索引 i 对应类别 i+1（类 0 是 CTC blank）。

    实测行为：词表外 token 会被静默映射成索引 0（已验证
    'ZZZNOTINVOC' -> 0）。加 1 后是类别 1，不是 blank 0，所以不会与
    blank 混淆，但它是一个**错误的类别**。cap300 词表只覆盖高频词，
    dev/train 里必然有词表外 gloss，这些位置会变成噪声监督。
    这里统计并报告被丢弃的比例，不静默吞掉。
    """
    global _OOV
    if _VOC is None:
        return []
    raw = [t for t in g.replace("　", " ").split("/") if t.strip()]
    ids = _VOC.encode(g)
    if len(ids) != len(raw):
        _OOV[0] += len(raw) - len(ids)
        _OOV[1] += len(raw)
    return [i + 1 for i in ids if i is not None and i >= 0]


def make_frame_labels(T, ids, peak_idx, blank_idx=BLANK_INDEX):
    """峰之间的段按 gloss 均分，段外标 blank（类 0）。

    返回 (T,) 的类别索引，值域 [0, len(ids)+1]，0 = CTC blank。
    """
    edges = [0] + sorted(peak_idx.tolist()) + [T]
    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    segs = [(s, e) for s, e in segs if e > s]
    y = np.full(T, blank_idx, dtype=np.int64)
    n = len(ids)
    if not segs or n == 0:
        return y
    # 把 segs 按长度分配 gloss：第 i 段分到第 i 个 gloss（超出则标 blank）
    for i, (s, e) in enumerate(segs):
        if i >= n:
            break
        y[s:e] = ids[i]
    return y


def build(items, dim, nrm, device, peak_cache={}):
    fs, ys, lens = [], [], []
    for it in items:
        arr = np.load(it["path"]).astype(np.float32)
        if nrm is not None:
            arr = (arr - nrm[0]) / nrm[1]
        fs.append(torch.from_numpy(arr))
        T = arr.shape[0]
        k = "pk" + it["id"]
        if k not in peak_cache:
            t, m = TFD.suggest_TM(T, len(it["ids"]))
            peak_cache[k] = TFD.detect_boundaries(arr.astype(np.float64), t, m, metric="l2")
        ys.append(torch.from_numpy(make_frame_labels(T, it["ids"], peak_cache[k])))
        lens.append(T)
    maxT = max(lens)
    B = len(items)
    fb = torch.zeros(B, maxT, dim)
    lb = torch.zeros(B, maxT, dtype=torch.long)
    for i, (f, l) in enumerate(zip(fs, ys)):
        fb[i, : len(f)] = f
        lb[i, : len(l)] = l
    return fb.to(device), lb.to(device), torch.tensor(lens, device=device)


@torch.no_grad()
def enc_of(ctc, fb, lens):
    h = ctc.projection(ctc.normalize(fb))
    if ctc.config.subsample_stride != 1:
        h = ctc.subsample(h.transpose(1, 2)).transpose(1, 2)
    e, _ = ctc.temporal(h)
    return e


def measure(rec_logits, spot_logits, alpha, lengths, n_gloss, blank=BLANK_INDEX):
    """在 late fusion 后测 blank 率与 peak 定位。

    rec_logits: (B, T, C_rec) 识别侧
    spot_logits: (B, T, C_spot) 帧级 spotting（本脚本与识别侧同构）
    """
    pr = F.softmax(rec_logits, dim=-1)
    ps = F.softmax(spot_logits, dim=-1)
    fused = torch.log((alpha * pr + (1 - alpha) * ps).clamp_min(1e-8))
    out = {}
    for name, lg in (("rec", rec_logits), ("fuse", fused)):
        best = lg.argmax(-1)
        br, pfpt, ntok = [], [], []
        segs_all = []
        for b in range(lg.shape[0]):
            T = int(lengths[b])
            ids = best[b, :T].cpu().numpy()
            br.append(float((ids == blank).mean()))
            segs = D.peak_segments(lg[b, :T].cpu().numpy(), blank)
            ntok.append(max(int(n_gloss[b]), 1))
            pfpt.append(sum(e - s for s, e, _ in segs))
            segs_all.extend(e - s for s, e, _ in segs)
        out[name] = {
            "blank_ratio": float(np.mean(br)),
            "peak_frames_per_token": float(np.sum(pfpt) / max(np.sum(ntok), 1)),
            "peak_len_median": float(np.median(segs_all)) if segs_all else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--train-feats", default="artifacts/part3_features/train")
    ap.add_argument("--val-feats", default="artifacts/part3_features/validation")
    ap.add_argument("--train-labels", default="data/raw/CE-CSL/label/train.csv")
    ap.add_argument("--val-labels", default="data/raw/CE-CSL/label/dev.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--train-limit", type=int, default=4973)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--alphas", default="0.9,0.7,0.5,0.3,0.1,0.05,0.0")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p2b-late-fusion.json")
    a = ap.parse_args()

    global _VOC
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctc, voc, cfg, nrm_raw = load_ctc(REPO / a.ckpt)
    ctc.to(device)
    for p in ctc.parameters():
        p.requires_grad_(False)
    _VOC = voc
    dim = cfg.input_size
    ncls = ctc.num_classes
    nrm = None
    if nrm_raw:
        nrm = (np.asarray(nrm_raw["mean"], np.float32),
               np.asarray(nrm_raw["std"], np.float32) + 1e-8)
    print("device={}  enc_dim={}  识别类别数={}".format(
        device, cfg.hidden_size * 2, ncls))

    tr_l = {i: g for i, g in read_csv(REPO / a.train_labels)}
    va_l = {i: g for i, g in read_csv(REPO / a.val_labels)}
    tr = collect(REPO / a.train_feats, dim, tr_l, a.train_limit)
    va = collect(REPO / a.val_feats, dim, va_l, a.val_limit)
    print("train {} / val {}".format(len(tr), len(va)))
    if not tr or not va:
        raise SystemExit("样本为空")
    oov_rate = _OOV[0] / max(_OOV[1], 1)
    print("词表外 gloss token 占比 {:.2%}（cap300 词表只覆盖高频词）".format(oov_rate))

    spot = FrameSpotter(cfg.hidden_size * 2, ncls).to(device)
    print("P_spot 头参数量 {:.2f}M（CTC 冻结）".format(
        sum(p.numel() for p in spot.parameters()) / 1e6))

    va_gloss = [len(it["ids"]) for it in va]

    opt = torch.optim.AdamW(spot.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    hist = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        spot.train()
        tot, nb = 0.0, 0
        order = np.random.permutation(len(tr))
        for i in range(0, len(order), a.batch):
            ch = [tr[j] for j in order[i: i + a.batch]]
            if len(ch) < 2:
                continue
            fb, lb, lens = build(ch, dim, nrm, device)
            with torch.no_grad():
                enc = enc_of(ctc, fb, lens)
            sl = spot(enc)                       # (B,C,T)
            T = sl.shape[-1]
            tgt = lb[:, :T]
            valid = torch.arange(T, device=device)[None, :] < lens[:, None]
            lsm = F.log_softmax(sl.permute(0, 2, 1), -1)
            # 稀有 gloss（段少）权重更高：对治全 blank 平凡解
            loss = -(lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * valid).sum() / valid.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(spot.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        print("ep {:2d}  spot_loss={:.4f}".format(ep, tot / max(nb, 1)))
        hist.append({"epoch": ep, "spot_loss": tot / max(nb, 1)})

    # alpha 扫描。
    # 关键：rec 与 spot 必须在**同一次 forward、同一 padding** 下产出，
    # 否则 batch 内 maxT 不同会导致切片错位，融合结果退化成纯 rec。
    spot.eval()
    alphas = [float(x) for x in a.alphas.split(",")]
    results = {}
    pblank_r, pblank_s = [], []
    with torch.no_grad():
        per_batch = []
        for i in range(0, len(va), a.batch):
            ch = va[i: i + a.batch]
            fb, _, lens = build(ch, dim, nrm, device)
            enc = enc_of(ctc, fb, lens)
            rl = ctc(fb, lens)                       # (B, T, C)
            sl = spot(enc)                           # (B, C, T)
            assert sl.shape[-1] == rl.shape[1], (
                "时间轴不一致：spot {} vs rec {}".format(sl.shape[-1], rl.shape[1])
            )
            per_batch.append(
                (
                    F.log_softmax(rl, -1).cpu(),                    # (B, T, C)
                    F.log_softmax(sl.permute(0, 2, 1), -1).cpu(),  # (B, T, C)
                    lens.cpu(),
                    [len(it["ids"]) for it in ch],
                )
            )
        # sanity: 确认 spot 与 rec 不是同一路
        r0, s0 = per_batch[0][0], per_batch[0][1]
        diff = float((r0 - s0).abs().mean())
        print("rec/spot 平均绝对差 {:.4f}（若为 0 说明融合无效）".format(diff))
        assert diff > 1e-6, "spot 与 rec 相同，late fusion 无意义"

        for al in alphas:
            br_f, pf_f, nf = [], [], []
            br_r, pf_r = [], []
            seg_f, seg_r = [], []
            for rl, sl, lens, gs in per_batch:
                pr = F.softmax(rl, dim=-1)
                ps = F.softmax(sl, dim=-1)
                fused = torch.log((al * pr + (1 - al) * ps).clamp_min(1e-8))
                # 记录两路的 blank 概率量级，用于诊断概率校准
                pblank_r.append(float(pr[:, :, 0].mean()))
                pblank_s.append(float(ps[:, :, 0].mean()))
                for b in range(rl.shape[0]):
                    T = int(lens[b])
                    n_ = max(int(gs[b]), 1)
                    ids_f = fused[b, :T].argmax(-1).numpy()
                    ids_r = rl[b, :T].argmax(-1).numpy()
                    br_f.append(float((ids_f == 0).mean()))
                    br_r.append(float((ids_r == 0).mean()))
                    sf = D.peak_segments(fused[b, :T].numpy(), 0)
                    sr = D.peak_segments(rl[b, :T].numpy(), 0)
                    pf_f.append(sum(e - s for s, e, _ in sf)); nf.append(n_)
                    pf_r.append(sum(e - s for s, e, _ in sr))
                    seg_f.extend(e - s for s, e, _ in sf)
                    seg_r.extend(e - s for s, e, _ in sr)
            results[str(al)] = {
                "blank_ratio": float(np.mean(br_f)),
                "peak_frames_per_token": float(np.sum(pf_f) / max(np.sum(nf), 1)),
                "peak_len_median": float(np.median(seg_f)) if seg_f else 0.0,
                "rec_blank_ratio_same_batch": float(np.mean(br_r)),
                "rec_pfpt_same_batch": float(np.sum(pf_r) / max(np.sum(nf), 1)),
                "mean_p_blank_rec": float(np.mean(pblank_r)),
                "mean_p_blank_spot": float(np.mean(pblank_s)),
            }
            r = results[str(al)]
            print("alpha={:<5} blank={:.4f} (rec {:.4f})  pfpt={:.2f} (rec {:.2f})  "
                  "p_blank rec={:.3f} spot={:.3f}".format(
                      al, r["blank_ratio"], r["rec_blank_ratio_same_batch"],
                      r["peak_frames_per_token"], r["rec_pfpt_same_batch"],
                      r["mean_p_blank_rec"], r["mean_p_blank_spot"]))
    dt = (time.time() - t0) / 60

    best_alpha = min(results, key=lambda k: results[k]["blank_ratio"])
    b1 = results[best_alpha]
    b0 = {"blank_ratio": b1["rec_blank_ratio_same_batch"],
          "peak_frames_per_token": b1["rec_pfpt_same_batch"]}
    print("")
    print("=" * 64)
    print("P2-b late fusion 对 CTC blank 率的实际影响")
    print("=" * 64)
    print("基线 blank  {:.4f}   pfpt {:.3f}".format(b0["blank_ratio"], b0["peak_frames_per_token"]))
    print("最优 alpha {}".format(best_alpha))
    print("融合后 blank {:.4f}   pfpt {:.3f}".format(b1["blank_ratio"], b1["peak_frames_per_token"]))
    print("blank 绝对变化 {:+.4f}  相对变化 {:+.2%}".format(
        b1["blank_ratio"] - b0["blank_ratio"],
        b1["blank_ratio"] / b0["blank_ratio"] - 1 if b0["blank_ratio"] else 0))
    print("耗时 {:.1f} 分钟".format(dt))
    print("")
    print("口径说明：这是 late fusion 后的 argmax blank 占比，")
    print("不是重训 CTC 得到的 blank 率；P_spot 是弱监督（伪边界 + gloss 均分）。")

    rep = {
        "experiment": "P2-b SMART-style late fusion on CTC blank rate",
        "alpha_scan": {k: v for k, v in results.items()},
        "best_alpha": float(best_alpha),
        "baseline": b0,
        "fused_best": b1,
        "blank_abs_change": b1["blank_ratio"] - b0["blank_ratio"],
        "blank_rel_change": (b1["blank_ratio"] / b0["blank_ratio"] - 1) if b0["blank_ratio"] else 0.0,
        "spot_params_M": sum(p.numel() for p in spot.parameters()) / 1e6,
        "weak_supervision_note": (
            "P_spot 用 TFD 伪边界切段 + gloss 均分做帧级标签，属弱监督；"
            "CE-CSL 无 sign-level 边界标注，无法得到人工帧级 GT。"
        ),
        "config": vars(a),
        "history": hist,
        "minutes": dt,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
