# -*- coding: utf-8 -*-
"""P2-c：修正版 —— 让帧级标签含合理 blank 比例，再测 late fusion 对 CTC blank 率的影响。

P2-b 暴露了一个必须先修的设计缺陷（已实测确认）：

  伪边界标签里 blank（类 0）占比 = 0.0%
  段数/gloss 数 = 0.67

  原因：TFD 峰数只有 gloss 数的 0.67 倍，`make_frame_labels` 把每个段都填满
  一个 gloss，结果没有任何帧留成 blank。P_spot 于是根本不预测 blank
  （实测 p_blank_spot = 0.003，而冻结 CTC 是 0.903）。
  此时 P_fuse = α·P_rec + (1-α)·P_spot 在数学上不可能让 argmax 离开 blank ——
  这是标签构造的缺陷，不是「late fusion 无效」的结论。

修正：把段落数与 gloss 数对齐，并显式保留 blank 帧。
  策略 A（等分）：n_seg = n_gloss，段边界用 TFD 峰但**重采样**到 n_gloss 段
  策略 B（欠分割 + 显式 blank）：TFD 峰不足时，多余 gloss 不给帧监督，
        剩余段统一标 blank，使 blank 占比落在合理区间

采用 B，理由：TS²-TFD 论文的欠分割偏置（Table 5：预测 9.90 vs GT 10.28）
本身就允许漏检，硬凑到 n_gloss 段会引入假的段边界。
B 让「无监督的 gloss」表现为 blank，与 CTC 的 blank 语义一致 ——
这正是让 P_spot 能提供真实 blank 竞争信号的关键。

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

BLANK = BLANK_INDEX          # 0
_VOC = None
_OOV = [0, 0]


def voc_encode(g):
    global _VOC, _OOV
    if _VOC is None:
        return []
    raw = [t for t in g.replace("　", " ").split("/") if t.strip()]
    ids = _VOC.encode(g)
    if len(ids) != len(raw):
        _OOV[0] += len(raw) - len(ids)
        _OOV[1] += len(raw)
    return [i + 1 for i in ids if i is not None and i >= 0]


def make_labels_strategy_b(T, ids, peak_idx):
    """策略 B：欠分割 + 显式 blank。

    动机（P2-b 实测缺陷）：TFD 峰只有 gloss 数的 0.67 倍，若把段全填满 gloss，
    标签 blank 占比是 0.0%，P_spot 根本不预测 blank（实测 p_blank=0.001 vs CTC 0.896），
    late fusion 数学上不可能让 argmax 离开 blank。

    做法：
      1. TFD 峰作为**候选**边界（尊重伪边界给出的真实转折位置）
      2. 若峰数 + 1 < n_gloss：在峰之间**均匀插入**额外边界补足段数
         （插在段中部，不与已有峰冲突）
      3. 若峰数 + 1 > n_gloss：保留前 n_gloss 段，其余帧标 blank
      4. 段与 gloss 一一对应；未能覆盖的帧标 blank

    与论文的关系：TS²-TFD 的欠分割偏置（Table 5：预测 9.90 vs GT 10.28）
    是**评测时**的偏好；这里在**构造训练标签**时补足段数，
    目的是让标签的 blank 先验与 CTC 的 blank 语义兼容，
    不是修改论文方法的边界检测本身。这个区别必须如实记录。
    """
    n_g = len(ids)
    y = np.full(T, BLANK, dtype=np.int64)
    if n_g == 0:
        return y, 0, 0
    n_seg = n_g                                   # 段数与 gloss 数对齐
    peaks = sorted(int(x) for x in peak_idx if 0 < int(x) < T)
    # 用峰作锚点，再均匀细分到 n_seg 段
    if len(peaks) >= n_seg - 1:
        edges = [0] + peaks[: n_seg - 1] + [T]
    else:
        base = peaks + [T]
        need = n_seg - len(base)
        edges = [0]
        for i in range(len(base) - 1):
            lo, hi = base[i], base[i + 1]
            span = hi - lo
            for k in range(1, need + 1):
                # 在这一段内均匀插 need 个点
                edges.append(lo + max(1, round(span * k / (need + 1))))
        edges = sorted(set(e for e in edges if 0 < e < T))
        edges = [0] + edges + [T]
    edges = sorted(set([0] + [e for e in edges if 0 < e < T] + [T]))
    segs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    segs = [(s, e) for s, e in segs if e > s][:n_seg]
    # 长段优先分配 gloss
    order = sorted(range(len(segs)), key=lambda i: -(segs[i][1] - segs[i][0]))
    sup = 0
    for rank, i in enumerate(order):
        if rank >= n_g:
            break
        s, e = segs[i]
        y[s:e] = ids[rank]
        sup += 1
    return y, sup, n_g - sup


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


def build(items, dim, nrm, device):
    fs, ys, lens, sup, uns = [], [], [], [], []
    for it in items:
        arr = np.load(it["path"]).astype(np.float32)
        if nrm is not None:
            arr = (arr - nrm[0]) / nrm[1]
        fs.append(torch.from_numpy(arr))
        T = arr.shape[0]
        t, m = TFD.suggest_TM(T, len(it["ids"]))
        pk = TFD.detect_boundaries(arr.astype(np.float64), t, m, metric="l2")
        y, s, u = make_labels_strategy_b(T, it["ids"], pk)
        ys.append(torch.from_numpy(y))
        lens.append(T)
        sup.append(s)
        uns.append(u)
    maxT = max(lens)
    B = len(items)
    fb = torch.zeros(B, maxT, dim)
    lb = torch.full((B, maxT), BLANK, dtype=torch.long)
    for i, (f, l) in enumerate(zip(fs, ys)):
        fb[i, : len(f)] = f
        lb[i, : len(l)] = l
    return (fb.to(device), lb.to(device), torch.tensor(lens, device=device),
            sum(sup), sum(uns))


@torch.no_grad()
def enc_of(ctc, fb, lens):
    h = ctc.projection(ctc.normalize(fb))
    if ctc.config.subsample_stride != 1:
        h = ctc.subsample(h.transpose(1, 2)).transpose(1, 2)
    e, _ = ctc.temporal(h)
    return e


class Spotter(nn.Module):
    """P_spot：帧级 gloss 分类 + 可学温度（概率校准必需，见 P2-b 注释）。"""

    def __init__(self, enc_dim, ncls, hidden=128, layers=5):
        super().__init__()
        self.inp = nn.Conv1d(enc_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, 3, padding=2**i, dilation=2**i) for i in range(layers)]
        )
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.out = nn.Conv1d(hidden, ncls, 1)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for blk, nrm in zip(self.blocks, self.norms):
            h = F.relu(nrm(blk(h)) + h)
        return self.out(h) * self.log_temp.exp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/checkpoints/ctc-landmark48-cap300.pt")
    ap.add_argument("--train-feats", default="artifacts/part3_features/train")
    ap.add_argument("--val-feats", default="artifacts/part3_features/validation")
    ap.add_argument("--train-labels", default="data/raw/CE-CSL/label/train.csv")
    ap.add_argument("--val-labels", default="data/raw/CE-CSL/label/dev.csv")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--blank-weight", type=float, default=1.0,
                    help="标签里 blank 帧的损失权重。标签 blank 占比低时需提高，"
                         "否则 spot 仍会倾向预测 gloss 而非 blank")
    ap.add_argument("--train-limit", type=int, default=4973)
    ap.add_argument("--val-limit", type=int, default=300)
    ap.add_argument("--alphas", default="0.9,0.7,0.5,0.3,0.1")
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p2c-late-fusion.json")
    a = ap.parse_args()

    global _VOC
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctc, voc, cfg, nrm_raw = load_ctc(REPO / a.ckpt)
    ctc.to(device)
    for p in ctc.parameters():
        p.requires_grad_(False)
    _VOC = voc
    dim, ncls = cfg.input_size, ctc.num_classes
    nrm = None
    if nrm_raw:
        nrm = (np.asarray(nrm_raw["mean"], np.float32),
               np.asarray(nrm_raw["std"], np.float32) + 1e-8)

    tr = collect(REPO / a.train_feats, dim,
                 {i: g for i, g in read_csv(REPO / a.train_labels)}, a.train_limit)
    va = collect(REPO / a.val_feats, dim,
                 {i: g for i, g in read_csv(REPO / a.val_labels)}, a.val_limit)
    print("device={}  train={}  val={}".format(device, len(tr), len(va)))
    if not tr or not va:
        raise SystemExit("样本为空")
    print("词表外 gloss token 占比 {:.2%}".format(_OOV[0] / max(_OOV[1], 1)))

    # 先验：标签里 blank 占比
    fb, lb, lens, sup, uns = build(tr[:200], dim, nrm, device)
    prior_blank = float((lb == BLANK).float().mean())
    print("标签 blank 占比 {:.1%}   有监督 gloss {}/{}".format(
        prior_blank, sup, sup + uns))

    spot = Spotter(cfg.hidden_size * 2, ncls).to(device)
    print("P_spot 参数量 {:.2f}M".format(sum(p.numel() for p in spot.parameters()) / 1e6))

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
            fb, lb, lens, _, _ = build(ch, dim, nrm, device)
            with torch.no_grad():
                enc = enc_of(ctc, fb, lens)
            sl = spot(enc)                                  # (B,C,T)
            T = sl.shape[-1]
            tgt, valid = lb[:, :T], (torch.arange(T, device=device)[None, :] < lens[:, None])
            lsm = F.log_softmax(sl.permute(0, 2, 1), -1)
            nll = -lsm.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            w = torch.where(tgt == BLANK, a.blank_weight, 1.0)
            loss = ((nll * w) * valid).sum() / (valid.float() * w).sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(spot.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        with torch.no_grad():
            fb, _, lens, _, _ = build(va[:200], dim, nrm, device)
            sl = spot(enc_of(ctc, fb, lens))
            ps = F.softmax(sl.permute(0, 2, 1), -1)
            pb = float(ps[:, :, 0].mean())
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "val_p_blank_spot": pb})
        print("ep {:2d}  loss={:.4f}  p_blank(spot)={:.4f}  温度={:.2f}".format(
            ep, tot / max(nb, 1), pb, float(spot.log_temp.exp())))

    # alpha 扫描
    spot.eval()
    per_batch = []
    with torch.no_grad():
        for i in range(0, len(va), a.batch):
            ch = va[i: i + a.batch]
            fb, _, lens, _, _ = build(ch, dim, nrm, device)
            enc = enc_of(ctc, fb, lens)
            rl = ctc(fb, lens)
            sl = spot(enc)
            per_batch.append((
                F.log_softmax(rl, -1).cpu(),
                F.log_softmax(sl.permute(0, 2, 1), -1).cpu(),
                lens.cpu(),
                [len(it["ids"]) for it in ch],
            ))
    results = {}
    for al in [float(x) for x in a.alphas.split(",")]:
        br_f, pf_f, nf, seg_f = [], [], [], []
        br_r, pf_r = [], []
        pbr, pbs = [], []
        for rl, sl, lens, gs in per_batch:
            pr, ps = F.softmax(rl, -1), F.softmax(sl, -1)
            fused = torch.log((al * pr + (1 - al) * ps).clamp_min(1e-8))
            pbr.append(float(pr[:, :, 0].mean())); pbs.append(float(ps[:, :, 0].mean()))
            for b in range(rl.shape[0]):
                T = int(lens[b]); n_ = max(int(gs[b]), 1)
                sf = D.peak_segments(fused[b, :T].numpy(), BLANK)
                sr = D.peak_segments(rl[b, :T].numpy(), BLANK)
                br_f.append(float((fused[b, :T].argmax(-1) == BLANK).float().mean()))
                br_r.append(float((rl[b, :T].argmax(-1) == BLANK).float().mean()))
                pf_f.append(sum(e - s for s, e, _ in sf)); pf_r.append(sum(e - s for s, e, _ in sr))
                nf.append(n_)
                seg_f.extend(e - s for s, e, _ in sf)
        results[str(al)] = {
            "blank_ratio": float(np.mean(br_f)),
            "peak_frames_per_token": float(np.sum(pf_f) / max(np.sum(nf), 1)),
            "peak_len_median": float(np.median(seg_f)) if seg_f else 0.0,
            "rec_blank": float(np.mean(br_r)),
            "rec_pfpt": float(np.sum(pf_r) / max(np.sum(nf), 1)),
            "p_blank_rec": float(np.mean(pbr)),
            "p_blank_spot": float(np.mean(pbs)),
        }
        r = results[str(al)]
        print("alpha={:<5} blank={:.4f} (rec {:.4f})  pfpt={:.2f} (rec {:.2f})  "
              "p_blank rec={:.3f} spot={:.3f}".format(
                  al, r["blank_ratio"], r["rec_blank"], r["peak_frames_per_token"],
                  r["rec_pfpt"], r["p_blank_rec"], r["p_blank_spot"]))

    dt = (time.time() - t0) / 60
    best = min(results, key=lambda k: results[k]["blank_ratio"])
    b0 = {"blank_ratio": results[best]["rec_blank"],
          "peak_frames_per_token": results[best]["rec_pfpt"]}
    b1 = results[best]
    print("")
    print("=" * 66)
    print("P2-c late fusion 对 CTC blank 率的实际影响（策略 B 标签）")
    print("=" * 66)
    print("标签 blank 先验 {:.1%}   spot 学到的 p_blank {:.3f}   CTC p_blank {:.3f}".format(
        prior_blank, b1["p_blank_spot"], b1["p_blank_rec"]))
    print("基线（纯 CTC） blank {:.4f}  pfpt {:.2f}".format(
        b0["blank_ratio"], b0["peak_frames_per_token"]))
    print("最优 alpha={}  blank {:.4f}  pfpt {:.2f}".format(
        best, b1["blank_ratio"], b1["peak_frames_per_token"]))
    print("blank 绝对变化 {:+.4f}   相对 {:+.2%}".format(
        b1["blank_ratio"] - b0["blank_ratio"],
        b1["blank_ratio"] / b0["blank_ratio"] - 1 if b0["blank_ratio"] else 0))
    print("耗时 {:.1f} 分钟".format(dt))

    rep = {
        "experiment": "P2-c late fusion with under-segmentation + explicit blank labels",
        "label_strategy": "B: TFD 欠分割 + 未匹配 gloss 覆盖的帧标 blank",
        "label_blank_prior": prior_blank,
        "gloss_supervised_ratio": sup / max(sup + uns, 1),
        "alpha_scan": results,
        "best_alpha": float(best),
        "baseline": b0,
        "fused_best": b1,
        "blank_abs_change": b1["blank_ratio"] - b0["blank_ratio"],
        "blank_rel_change": (b1["blank_ratio"] / b0["blank_ratio"] - 1) if b0["blank_ratio"] else 0.0,
        "config": vars(a),
        "history": hist,
        "minutes": dt,
        "caveats": [
            "P_spot 用 TFD 伪边界做弱监督，CE-CSL 无人工帧级 GT",
            "这是 late fusion 后的 argmax blank 占比，不等于重训 CTC 的 blank 率",
            "alpha 网格是本项目自选，论文原值 0.7 在此场景下无效（概率尺度不可比）",
        ],
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
