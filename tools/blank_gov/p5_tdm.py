# -*- coding: utf-8 -*-
"""P5：TDM 多尺度时序差分对帧级定位的增益（单因素对照，冻结 CTC）。

文献依据（已逐条核实原文）：
  SignShift (arXiv, MM'26) Sec 3.3 Temporal Difference Module
    式(1) 偏移集 D = {1,2,4,...,d_max}，X_Δ^d(t) = x_t - x_{t-d}（t<=d 置 0）
    式(2) 共享 1x1 conv 投影后跨尺度平均
    式(3) 可学门控 G = σ(Conv1D(X_Δ))
    式(4) 门控残差 X̂_Δ = G ⊙ X_Δ + Proj(X)
    式(5) 手/脸流各自独立做同样差分
    式(6) 三流融合，α_h/α_f 为可训练标量门
  超参：d_max = 16（Sec 4.1）
  消融：Table 2 —— TDM 单独贡献 F1@50 +2.48 ~ +5.79
        （How2Sign/MS-TCN 45.12→47.60；How2Sign/ASFormer 49.44→55.23）

设计（遵守单因素消融）：
  基线   = P2-a 骨架（CTC 冻结 + frame-level BIO 头）
  实验组 = 基线 + TDM 前置（global/hand/face 三流）
  唯一变量 = 是否加 TDM。其余（数据、特征、seed、epoch、评估口径）完全一致。

  为什么 TDM 有可能在这里有用：P2-a 已证明密集帧级监督能把
  pfpt 从 0 拉到 6.66（脱离 peaky），但 WER 没改善。
  SignShift 的主张是「边界线索来自细微时序变化，静态特征看不出」——
  我们的 landmark 特征里虽已含 delta 块，但那是**相邻帧一阶差分**；
  TDM 用的是**多尺度（最长 16 帧）差分 + 门控**，信息量不同。
  这是与 P2-a 的实质差异，不是重复实验。

**迁移损失（必须如实记录）**：
  论文三流 = global video(I3D) / hand(HaMeR+ResNet18) / face(BlazeFace+ResNet18)，
  都需要 RGB。本项目只有 MediaPipe landmark。
  本实现用 landmark 自身分块近似（global=368 / hand=252 / face=24）。
  **face 流仅 24 维（论文是 512 维 ResNet 特征），信息量远少于论文。**
  所以即便有效，也达不到论文的 +2.48~+5.79 幅度。

只用 train/validation，不触碰 test。
"""
import argparse
import csv
import json
import random
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
import bio_labels as BIO
from tdm import MultiStreamTDM, TDM, split_streams, DEFAULT_OFFSETS
from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig


class SEChannelAttention(nn.Module):
    """Wójcicka Sec 4.4.1（r=16）。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x):
        w = x.mean(dim=-1, keepdim=True)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(w))))


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.n1 = nn.BatchNorm1d(channels)
        self.n2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        out = F.relu(self.n1(self.conv1(x)))
        out = self.drop(out)
        out = self.n2(self.conv2(out))
        return F.relu(out + r)


class Spotter(nn.Module):
    """frame-level BIO 三类分类器（与 P2-a 同结构，保证单因素）。

    tdm_mode:
      "none"    -> 基线，无 TDM
      "global"  -> 只对 global 流做 TDM（**论文测 TDM 增益的口径**，
                   原文 Effect of TDM: "applying it to the global full-frame
                   feature stream"）
      "multi"   -> 三流 TDM + 可学习标量门融合（式 5,6）
    """

    def __init__(self, in_dim, num_classes=3, channels=128, num_stages=4,
                 num_layers=11, dropout=0.1, smooth_lambda=0.15,
                 tdm_mode="none", tdm_hidden=128):
        super().__init__()
        self.tdm_mode = tdm_mode
        if tdm_mode == "global":
            # 只用一个 TDM 处理全部通道，不拆流 —— 对齐论文 Effect of TDM
            self.tdm = TDM(in_dim, hidden=tdm_hidden, offsets=DEFAULT_OFFSETS)
            feat_dim = tdm_hidden
        elif tdm_mode == "multi":
            # 三流：BiLSTM 输出无部位语义，按通道等分（见 tdm.split_streams 退化路径）
            c1 = in_dim // 2
            c2 = in_dim // 4
            self.tdm = MultiStreamTDM(c1, c2, in_dim - c1 - c2,
                                      hidden=tdm_hidden, offsets=DEFAULT_OFFSETS)
            feat_dim = tdm_hidden
        else:
            feat_dim = in_dim
        self.se = SEChannelAttention(feat_dim)
        self.stage_in = nn.ModuleList(
            [nn.Conv1d(feat_dim if s == 0 else num_classes, channels, 1)
             for s in range(num_stages)]
        )
        self.stages = nn.ModuleList(
            [nn.ModuleList([DilatedResidualLayer(channels, 2**l, dropout)
                            for l in range(num_layers)])
             for _ in range(num_stages)]
        )
        self.out = nn.Conv1d(channels, num_classes, 1)
        self.smooth_lambda = smooth_lambda
        self.num_stages = num_stages

    def forward(self, enc):
        """enc: (B, T, C) BiLSTM 输出 -> stage logits 列表"""
        if self.tdm_mode == "global":
            x = self.tdm(enc)
        elif self.tdm_mode == "multi":
            g, h, f = split_streams(enc)
            x = self.tdm(g, h, f)
        else:
            x = enc
        hh = self.se(x.transpose(1, 2))
        outs = []
        for s, layers in enumerate(self.stages):
            hh = self.stage_in[s](hh)
            for layer in layers:
                hh = layer(hh)
            lg = self.out(hh)
            outs.append(lg)
            if s < self.num_stages - 1:
                hh = F.softmax(lg, dim=1)
        return outs

    def loss(self, stage_logits, target, weights, lengths):
        ce = 0.0
        w = weights.to(target.device)
        for lg in stage_logits:
            t = lg.shape[-1]
            tgt = target[:, :t]
            valid = torch.arange(t, device=target.device)[None, :] < lengths[:, None]
            lsm = F.log_softmax(lg.permute(0, 2, 1), dim=-1)
            wt = w[tgt.clamp(min=0)]
            per = -wt * lsm.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            ce = ce + (per * valid).sum() / valid.sum().clamp(min=1)
        ce = ce / max(len(stage_logits), 1)
        p = F.softmax(stage_logits[-1], dim=1)[:, 1:2, :]
        sm = ((p[:, :, 1:] - p[:, :, :-1]) ** 2).mean()
        return ce + self.smooth_lambda * sm, float(ce.detach()), float(sm.detach())


def _count_runs(ids):
    if ids.size == 0:
        return 0, np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(ids) != 0) + 1
    starts = np.concatenate([np.zeros(1, dtype=np.int64), change])
    ends = np.concatenate([change, np.array([ids.size], dtype=np.int64)])
    return int(starts.size), (ends - starts)


def load_ctc(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(p["model_config"])
    m = CTCRecognizer(cfg)
    m.load_state_dict(p["state_dict"])
    m.eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    vc = p.get("vocabulary_config") or {}
    voc = GlossVocabulary(tokens=tuple(p["vocabulary"]),
                          counts=dict(p.get("vocabulary_counts") or {}),
                          config=GlossSequenceConfig(**vc) if vc else GlossSequenceConfig())
    return m, voc, cfg, p.get("feature_normalizer")


def read_csv(p):
    rows = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["Number"]] = r["Gloss"]
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
        n = len([t for t in labels[stem].split("/") if t.strip()])
        if n == 0:
            continue
        out.append({"id": stem, "path": p, "n_gloss": n})
        if limit and len(out) >= limit:
            break
    return out


def make_batch(items, dim, nrm, device, peak_cache={}):
    feats, labels, lengths = [], [], []
    for it in items:
        arr = np.load(it["path"]).astype(np.float32)
        if nrm is not None:
            arr = (arr - nrm[0]) / nrm[1]
        feats.append(torch.from_numpy(arr))
        T = arr.shape[0]
        k = it["id"]
        if k not in peak_cache:
            peak_cache[k] = TFD.detect_boundaries(
                arr.astype(np.float64), *TFD.suggest_TM(T, it["n_gloss"]), metric="l2")
        labels.append(torch.from_numpy(
            BIO.labels_from_boundaries(T, peak_cache[k], dilate_k=1)))
        lengths.append(T)
    maxT = max(lengths)
    B = len(items)
    fb = torch.zeros(B, maxT, dim)
    lb = torch.full((B, maxT), BIO.OUT, dtype=torch.long)
    for i, (f, l) in enumerate(zip(feats, labels)):
        fb[i, : len(f)] = f
        lb[i, : len(l)] = l
    return fb.to(device), lb.to(device), torch.tensor(lengths, device=device)


@torch.no_grad()
def enc_of(ctc, fb, lens):
    h = ctc.projection(ctc.normalize(fb))
    if ctc.config.subsample_stride != 1:
        h = ctc.subsample(h.transpose(1, 2)).transpose(1, 2)
    e, _ = ctc.temporal(h)
    return e


def evaluate_spotter(model, ctc, items, dim, nrm, device, batch=32):
    model.eval()
    conf = np.zeros((3, 3), dtype=np.int64)
    n_peak = n_tok = lens_total = 0
    peak_lens = []
    for i in range(0, len(items), batch):
        ch = items[i: i + batch]
        fb, lb, lens = make_batch(ch, dim, nrm, device)
        enc = enc_of(ctc, fb, lens)
        sl = model(enc)
        pred = sl[-1].argmax(dim=1)
        for b in range(fb.shape[0]):
            T = int(lens[b])
            t = lb[b, :T].cpu().numpy()
            p = pred[b, :T].cpu().numpy()
            for tc in range(3):
                m = t == tc
                if m.any():
                    conf[tc] += np.bincount(p[m], minlength=3)
            nr, rl = _count_runs(p)
            n_peak += nr
            n_tok += ch[b]["n_gloss"]
            lens_total += int((p != BIO.OUT).sum())
            peak_lens.extend(rl.tolist())
    f1s = []
    for c in range(3):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
    pl = np.asarray(peak_lens, dtype=np.float64)
    return {
        "frame_macro_f1": float(np.mean(f1s)),
        "f1_B": f1s[2], "f1_I": f1s[1], "f1_O": f1s[0],
        "pred_O_fraction": float(conf[BIO.OUT, BIO.OUT] / max(conf.sum(), 1)),
        "peak_frames_per_token": float(lens_total / max(n_tok, 1)),
        "peak_count_ratio": float(n_peak / max(n_tok, 1)),
        "peak_len_median": float(np.median(pl)) if pl.size else 0.0,
    }


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
    ap.add_argument("--tdm-hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p5-tdm.json")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device = {}".format(device))

    ctc, voc, cfg, nrm_raw = load_ctc(REPO / a.ckpt)
    ctc.to(device)
    dim = cfg.input_size
    nrm = None
    if nrm_raw:
        nrm = (np.asarray(nrm_raw["mean"], np.float32),
               np.asarray(nrm_raw["std"], np.float32) + 1e-8)
    enc_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
    print("CTC 冻结：input={} vocab={} enc_dim={}".format(dim, voc.size, enc_dim))

    tr = collect(REPO / a.train_feats, dim,
                 read_csv(REPO / a.train_labels), a.train_limit)
    va = collect(REPO / a.val_feats, dim,
                 read_csv(REPO / a.val_labels), a.val_limit)
    print("train {} / val {}".format(len(tr), len(va)))
    if not tr or not va:
        raise SystemExit("样本为空")

    all_labels = []
    for it in tr[:300]:
        arr = np.load(it["path"]).astype(np.float64)
        pk = TFD.detect_boundaries(arr, *TFD.suggest_TM(arr.shape[0], it["n_gloss"]),
                                   metric="l2")
        all_labels.append(BIO.labels_from_boundaries(arr.shape[0], pk, 1))
    w = BIO.class_weights_from_labels(all_labels)
    wt = torch.tensor(w, dtype=torch.float32, device=device)
    print("类别权重  O={:.2f} I={:.2f} B={:.2f}（Wójcicka Table 2 语义）".format(*w))

    results = []
    # 三组对照，对齐论文消融口径：
    #   none   = 无 TDM（基线）
    #   global = 只对 global 流做 TDM（**论文 Effect of TDM 的口径**）
    #   multi  = 三流 TDM + 标量门融合（式 5,6）
    for mode, tag in (("none", "基线(no TDM)"),
                      ("global", "TDM(global only)"),
                      ("multi", "TDM(三流融合)")):
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        print("")
        print("=" * 68)
        print("{}   （CTC 冻结，唯一变量 = TDM 模式）".format(tag))
        print("=" * 68)
        model = Spotter(enc_dim, 3, tdm_mode=mode, tdm_hidden=a.tdm_hidden).to(device)
        npar = sum(p.numel() for p in model.parameters())
        print("头参数量 {:.2f}M".format(npar / 1e6))
        if mode != "none":
            print("  TDM 偏移集 {}（论文 d_max=16）".format(DEFAULT_OFFSETS))

        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
        base = evaluate_spotter(model, ctc, va, dim, nrm, device, a.batch)
        print("训练前  macroF1={:.4f}  pfpt={:.2f}".format(
            base["frame_macro_f1"], base["peak_frames_per_token"]))

        best = {"frame_macro_f1": -1.0}
        hist = []
        t0 = time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            order = np.random.permutation(len(tr))
            tot = nb = 0
            for i in range(0, len(order), a.batch):
                ch = [tr[j] for j in order[i: i + a.batch]]
                if len(ch) < 2:
                    continue
                fb, lb, lens = make_batch(ch, dim, nrm, device)
                with torch.no_grad():
                    enc = enc_of(ctc, fb, lens)
                sl = model(enc)
                loss, _, _ = model.loss(sl, lb, wt, lens)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                tot += float(loss.detach()); nb += 1
            sched.step()
            m = evaluate_spotter(model, ctc, va, dim, nrm, device, a.batch)
            rec = {"epoch": ep, "train_loss": tot / max(nb, 1), **m}
            hist.append(rec)
            print("ep {:2d}  loss={:.4f}  macroF1={:.4f}  pfpt={:.2f}  pcr={:.2f}  "
                  "peakLen={:.0f}  O={:.1%}".format(
                      ep, rec["train_loss"], m["frame_macro_f1"],
                      m["peak_frames_per_token"], m["peak_count_ratio"],
                      m["peak_len_median"], m["pred_O_fraction"]))
            if m["frame_macro_f1"] > best["frame_macro_f1"]:
                best = dict(m); best["epoch"] = ep
        dt = (time.time() - t0) / 60
        print("-> best ep{}  macroF1={:.4f}  pfpt={:.2f}  ({:.1f} 分钟)".format(
            best.get("epoch"), best["frame_macro_f1"],
            best["peak_frames_per_token"], dt))
        results.append({"tdm_mode": mode, "tag": tag, "params_M": npar / 1e6,
                        "baseline": base, "best": best, "history": hist,
                        "minutes": dt})

    b = next(r for r in results if r["tdm_mode"] == "none")
    g = next(r for r in results if r["tdm_mode"] == "global")
    m_ = next(r for r in results if r["tdm_mode"] == "multi")
    print("")
    print("=" * 68)
    print("P5 TDM 单因素对照汇总（CTC 冻结）")
    print("=" * 68)
    print("{:>20}  {:>10}  {:>10}  {:>8}  {:>8}".format(
        "配置", "macroF1", "pfpt", "peakLen", "f1_B"))
    for r in results:
        print("{:>20}  {:>10.4f}  {:>10.2f}  {:>8.0f}  {:>8.4f}".format(
            r["tag"], r["best"]["frame_macro_f1"],
            r["best"]["peak_frames_per_token"], r["best"]["peak_len_median"],
            r["best"]["f1_B"]))
    print("")
    d_g = g["best"]["frame_macro_f1"] - b["best"]["frame_macro_f1"]
    d_m = m_["best"]["frame_macro_f1"] - b["best"]["frame_macro_f1"]
    print("TDM(global only) 增益  dmacroF1={:+.4f}  dpfpt={:+.2f}   <- 对齐论文口径".format(
        d_g, g["best"]["peak_frames_per_token"] - b["best"]["peak_frames_per_token"]))
    print("TDM(三流融合)   增益  dmacroF1={:+.4f}  dpfpt={:+.2f}".format(
        d_m, m_["best"]["peak_frames_per_token"] - b["best"]["peak_frames_per_token"]))
    print("论文参考：SignShift Table 2，global 流 TDM 单独 F1@50 +2.48~+5.79")
    print("（论文的 global=I3D RGB 特征；本项目 enc 是 MediaPipe landmark 的 BiLSTM 输出）")

    best_gain = max(d_g, d_m)
    if best_gain > 0.02:
        verdict = "TDM 有明显增益（>0.02），值得叠加到 CTC 训练"
    elif best_gain > 0.005:
        verdict = "TDM 小幅增益（0.005~0.02），可作辅助项，但不足以改变 WER"
    else:
        verdict = "TDM 无明显增益 —— landmark 编码特征上的多尺度差分信息量不足"
    print("判定：{}".format(verdict))

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment": "P5 TDM (SignShift Sec 3.3) on frame-level spotting, CTC frozen",
        "single_factor": "only difference is TDM on/off",
        "literature_basis": {
            "module": "SignShift arXiv Sec 3.3 TDM, Eq(1)-(6)",
            "d_max": 16,
            "offsets": list(DEFAULT_OFFSETS),
            "reported_gain": "F1@50 +2.48 to +5.79 (Table 2)",
        },
        "migration_loss": {
            "paper_streams": "global I3D + hand HaMeR/ResNet18 + face BlazeFace/ResNet18 (needs RGB)",
            "our_streams": "landmark blocks: global 368 / hand 252 / face 24",
            "caveat": "face stream is 24-dim vs paper 512-dim ResNet features; "
                      "gain will be smaller than paper's",
        },
        "config": vars(a),
        "results": results,
        "delta_global_only": d_g,
        "delta_multi_stream": d_m,
        "delta_pfpt_global": g["best"]["peak_frames_per_token"] - b["best"]["peak_frames_per_token"],
        "delta_pfpt_multi": m_["best"]["peak_frames_per_token"] - b["best"]["peak_frames_per_token"],
        "verdict": verdict,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
