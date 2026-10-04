# -*- coding: utf-8 -*-
"""P2-a 单因素对照：冻结 CTC，只训 frame-level BIO 头。

目的：验证「密集帧级监督」本身能否把 peak_frames_per_token 从 0.26 拉上去。
这是消融纪律要求的最干净一步 —— 唯一变量是新增的 frame-level 监督，
CTC 侧完全冻结，所以结果不会被「模型整体重训」污染。

设计（依据论文）：
  - 标签：TS²-TFD 免训练伪边界 -> BIO 三类，B 标签 ±1 帧膨胀
          （Wójcicka Sec 4.2.1，t±k = 3 帧窗口）
  - 损失：加权 CE（Wójcicka Table 2: λ_Out=0.5, λ_Beg=5.0, λ_In=1.0）
          + 可选 SignShift 式 L_smooth（λ=0.15）
  - 骨干：SE 通道注意力 + MS-TCN 4 stage × 11 层，d_l=2^l，128ch
          （Wójcicka Sec 4.4.2）
  - CTC 侧：完全冻结（requires_grad=False），eval 模式

严格约束：
  - 只读 train / validation，**不触碰 test**
  - 主指标 peak_frames_per_token（用冻结 CTC 的 logits 口径，前后一致）
  - 副指标 frame macro-F1、O 占比（防退化为全 O 平凡解）
  - 收据落盘 artifacts/metrics/blank-gov/

用法:
  ./venv/bin/python tools/blank_gov/p2_frame_level.py --epochs 12
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
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
import diagnose as D
from cslr.recognition.model import CTCRecognizer, ctc_config_from_dict, BLANK_INDEX
from cslr.recognition.gloss_sequence import GlossVocabulary, GlossSequenceConfig


# ----------------------------------------------------------------------
# 模型：SE + MS-TCN（Wójcicka Sec 4.4）
# ----------------------------------------------------------------------
class SEChannelAttention(nn.Module):
    """SE 通道重标定，r=16（论文 Sec 4.4.1）。

    对本项目尤其相关：368 维里混了坐标、delta、presence mask 三种量级，
    论文与仓库注释都记录过未标准化会导致输出坍缩。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x):
        """x: (B, C, T) —— 与 Conv1d 期望一致。"""
        w = x.mean(dim=-1, keepdim=True)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w))))
        return x * w


class DilatedResidualLayer(nn.Module):
    """膨胀残差层，d_l = 2^l（论文 Sec 4.4.2）。"""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.n1 = nn.BatchNorm1d(channels)
        self.n2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        out = F.relu(self.n1(self.conv1(x)))
        out = self.drop(out)
        out = self.n2(self.conv2(out))
        return F.relu(out + r)


class FrameLevelSpotter(nn.Module):
    """SE + MS-TCN 帧级 BIO 分类器。

    论文结构：4 stage × 11 层，d_l=2^l，128ch，
    stage 2~4 吃上一 stage 的 softmax 输出，多阶段损失每 stage 求和。
    """

    def __init__(self, enc_dim: int, num_classes: int = 3, channels: int = 128,
                 num_stages: int = 4, num_layers: int = 11, dropout: float = 0.1,
                 smooth_lambda: float = 0.15):
        """enc_dim: BiLSTM 输出的特征维度（不是原始特征维度）。
        CTCRecognizer 的 hidden=256 且双向，所以 enc_dim = 512。"""
        super().__init__()
        self.se = SEChannelAttention(enc_dim)
        self.stage_in = nn.ModuleList(
            [nn.Conv1d(enc_dim if s == 0 else num_classes, channels, 1)
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

    def forward(self, x):
        """x: (B, T, C) —— BiLSTM 输出的布局，需转成 (B, C, T) 再走卷积。"""
        h = self.se(x.transpose(1, 2))
        stage_logits = []
        for s, layers in enumerate(self.stages):
            h = self.stage_in[s](h)
            for layer in layers:
                h = layer(h)
            lg = self.out(h)
            stage_logits.append(lg)
            if s < self.num_stages - 1:
                h = F.softmax(lg, dim=1)
        return stage_logits

    def loss(self, stage_logits, target, weights, lengths):
        """多阶段加权 CE + 平滑正则。target: (B,T) long"""
        ce = 0.0
        w = weights.to(target.device)
        for lg in stage_logits:
            b, c, t = lg.shape
            tgt = target[:, :t]
            valid = torch.arange(t, device=target.device)[None, :] < lengths[:, None]
            lg = lg.permute(0, 2, 1)                       # (B,T,C)
            lsm = F.log_softmax(lg, dim=-1)
            wt = w[tgt.clamp(min=0)]
            per = -wt * lsm.gather(-1, tgt.clamp(min=0, max=c - 1).unsqueeze(-1)).squeeze(-1)
            per = per * valid
            ce = ce + (per.sum() / valid.sum().clamp(min=1))
        ce = ce / max(len(stage_logits), 1)

        # SignShift 式 L_smooth（式 17），只用最后一 stage
        p = F.softmax(stage_logits[-1], dim=1)[:, 1:2, :]
        sm = ((p[:, :, 1:] - p[:, :, :-1]) ** 2).mean()
        return ce + self.smooth_lambda * sm, float(ce.detach()), float(sm.detach())


# ----------------------------------------------------------------------
def load_ctc(ckpt_path):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ctc_config_from_dict(payload["model_config"])
    model = CTCRecognizer(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)          # 冻结：单因素对照的关键
    vc = payload.get("vocabulary_config") or {}
    gconf = GlossSequenceConfig(**vc) if vc else GlossSequenceConfig()
    voc = GlossVocabulary(tokens=tuple(payload["vocabulary"]),
                          counts=dict(payload.get("vocabulary_counts") or {}),
                          config=gconf)
    return model, voc, cfg, payload.get("feature_normalizer")


@torch.no_grad()
def encode_ctc(ctc, feats, lengths, device):
    """取 BiLSTM 编码（冻结，不回传梯度）。"""
    h = ctc.projection(ctc.normalize(feats))
    if ctc.config.subsample_stride != 1:
        h = ctc.subsample(h.transpose(1, 2)).transpose(1, 2)
    enc, _ = ctc.temporal(h)
    return enc


def read_split_csv(csv_path, split_name):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        idcol = next((c for c in cols if c.lower() in ("number", "id", "name")), cols[0])
        gcol = next((c for c in cols if "gloss" in c.lower() or "label" in c.lower()), cols[-1])
        for r in rd:
            rows.append((r[idcol], r[gcol]))
    return rows
    out = []
    for p in sorted(feat_root.glob("*.npy")):
        if ".receipt." in p.name:
            continue
        try:
            shp = np.load(p).shape
        except Exception:
            continue
        if not (shp and shp[-1] == npy_dim):
            continue
        stem = p.stem.split(".landmark")[0].split(".clip")[0]
        if stem not in labels_by_id:
            continue
        g = labels_by_id[stem]
        toks = tuple(t for t in g.split("/") if t.strip())
        if need_gloss and not toks:
            continue
        out.append({"id": stem, "path": p, "n_gloss": len(toks)})
        if limit and len(out) >= limit:
            break
    return out


def make_batch(items, dim, nrm, device, voc=None, need_tokens=False, tfd_cache={}):
    feats, labels, lengths = [], [], []
    for it in items:
        arr = np.load(it["path"]).astype(np.float32)
        if nrm is not None:
            arr = (arr - nrm[0]) / nrm[1]
        feats.append(torch.from_numpy(arr))
        T = arr.shape[0]
        if "tfd" not in tfd_cache:
            peaks = TFD.detect_boundaries(arr.astype(np.float64),
                                         *TFD.suggest_TM(T, it["n_gloss"]),
                                         metric="l2")
            tfd_cache["tfd"] = peaks
        y = BIO.labels_from_boundaries(T, tfd_cache["tfd"], dilate_k=1)
        labels.append(torch.from_numpy(y))
        lengths.append(T)
        tfd_cache.pop("tfd", None)
    maxT = max(lengths)
    B = len(items)
    fb = torch.zeros(B, maxT, dim)
    lb = torch.full((B, maxT), BIO.OUT, dtype=torch.long)
    for i, (f, l, T) in enumerate(zip(feats, labels, lengths)):
        fb[i, :T] = f
        lb[i, :T] = l
    return (fb.to(device), lb.to(device), torch.tensor(lengths, device=device))


def build_samples(feat_root, npy_dim, labels_by_id, limit):
    """收集特征路径与 gloss 数。

    labels_by_id 必须来自与 feat_root 对应的划分（train.csv / dev.csv），
    否则 train 目录的 id 在 dev 标签里查不到，会被全部过滤掉。
    """
    out = []
    for p in sorted(Path(feat_root).glob("*.npy")):
        if ".receipt." in p.name:
            continue
        try:
            shp = np.load(p).shape
        except Exception:
            continue
        if not (shp and shp[-1] == npy_dim):
            continue
        stem = p.stem.split(".landmark")[0].split(".clip")[0]
        if stem not in labels_by_id:
            continue
        toks = [t for t in labels_by_id[stem].split("/") if t.strip()]
        if not toks:
            continue
        out.append({"id": stem, "path": p, "n_gloss": len(toks)})
        if limit and len(out) >= limit:
            break
    return out


def _count_runs(ids):
    """返回 (游程个数, 各游程长度)。

    口径说明：这里数的是「极大同类游程」的个数。
    CTC 侧的 peak 是非 blank 连续段（diagnose.peak_segments），
    两者语义一致 —— 都是「一段连续同标签」。
    区别：CTC 用 argmax 的非 blank 段，这里用非 O 段。
    """
    if ids.size == 0:
        return 0, np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(ids) != 0) + 1
    starts = np.concatenate([np.zeros(1, dtype=np.int64), change])
    ends = np.concatenate([change, np.array([ids.size], dtype=np.int64)])
    return int(starts.size), (ends - starts)


def eval_frame_level(model, ctc, loader_items, dim, nrm, device, voc, batch=32):
    model.eval()
    conf = np.zeros((3, 3), dtype=np.int64)   # 混淆矩阵 (true, pred)
    n_peak_total = 0
    n_tok_total = 0
    lens_total = 0
    peak_lens_all = []
    with torch.no_grad():
        for i in range(0, len(loader_items), batch):
            chunk = loader_items[i: i + batch]
            fb, lb, lens = make_batch(chunk, dim, nrm, device)
            enc = encode_ctc(ctc, fb, lens, device)
            sl = model(enc)
            # sl[-1]: (B, C, T) -> argmax(dim=1) 直接得 (B, T) 类别序列
            pred = sl[-1].argmax(dim=1)
            for b in range(fb.shape[0]):
                T = int(lens[b])
                t = lb[b, :T].cpu().numpy()
                p = pred[b, :T].cpu().numpy()
                # 混淆矩阵 (true, pred)
                for tc in range(3):
                    m = t == tc
                    if m.any():
                        conf[tc] += np.bincount(p[m], minlength=3)
                # 指标口径必须与 diagnose.py 一致，否则前后不可比：
                #   peak_frames_per_token = 非 O 帧数 / gloss 数
                #   peak_count_ratio      = peak **段数** / gloss 数（不是帧数）
                # 段按极大同类游程切分，等价于 pipeline._run_onsets 的 run 起点。
                n_runs, run_lens = _count_runs(p)
                n_peak_total += n_runs
                n_tok_total += chunk[b]["n_gloss"]
                lens_total += int((p != BIO.OUT).sum())
                peak_lens_all.extend(run_lens.tolist())
    # macro F1（论文口径：不用 accuracy，会被 O 掩盖）
    f1s = []
    for c in range(3):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    o_frac = conf[BIO.OUT, BIO.OUT] / max(conf.sum(), 1)
    pl = np.asarray(peak_lens_all, dtype=np.float64)
    return {
        "frame_macro_f1": float(np.mean(f1s)),
        "frame_f1_per_class": {"O": f1s[0], "I": f1s[1], "B": f1s[2]},
        "pred_O_fraction": float(o_frac),
        "peak_frames_per_token": float(lens_total / max(n_tok_total, 1)),
        "peak_count_ratio": float(n_peak_total / max(n_tok_total, 1)),
        "peak_len_median": float(np.median(pl)) if pl.size else 0.0,
        "peak_len_p90": float(np.percentile(pl, 90)) if pl.size else 0.0,
        "confusion": conf.tolist(),
    }


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
    ap.add_argument("--train-limit", type=int, default=1200)
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/metrics/blank-gov/p2-frame-level.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    print("device = {}".format(device))

    ctc, voc, cfg, nrm_raw = load_ctc(REPO / a.ckpt)
    ctc.to(device)
    nrm = None
    if nrm_raw:
        nrm = (np.asarray(nrm_raw["mean"], dtype=np.float32),
               np.asarray(nrm_raw["std"], dtype=np.float32) + 1e-8)
    dim = cfg.input_size
    print("CTC 冻结：input_size={} vocab={} hidden*2={}".format(
        dim, voc.size, cfg.hidden_size * 2))

    # train / val 各自的标签文件（id 空间不同，不能混用）
    tr_labels = {rid: g for rid, g in read_split_csv(REPO / a.train_labels, "train")}
    va_labels = {rid: g for rid, g in read_split_csv(REPO / a.val_labels, "dev")}

    tr = build_samples(REPO / a.train_feats, dim, tr_labels, a.train_limit)
    va = build_samples(REPO / a.val_feats, dim, va_labels, a.val_limit)
    print("train {} 条 / val {} 条".format(len(tr), len(va)))
    if not tr or not va:
        raise SystemExit("样本为空：检查特征目录与维度")

    # 类别权重（Wójcicka Table 2 实测值为基底 + 数据比例几何平均）
    all_labels = []
    for it in tr[:300]:
        arr = np.load(it["path"]).astype(np.float64)
        pk = TFD.detect_boundaries(arr, *TFD.suggest_TM(arr.shape[0], it["n_gloss"]), metric="l2")
        all_labels.append(BIO.labels_from_boundaries(arr.shape[0], pk, 1))
    w = BIO.class_weights_from_labels(all_labels)
    print("类别权重  O={:.2f} I={:.2f} B={:.2f}".format(w[0], w[1], w[2]))
    wt = torch.tensor(w, dtype=torch.float32, device=device)

    enc_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)
    model = FrameLevelSpotter(enc_dim, num_classes=3).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print("frame-level 头参数量 {:.2f}M（CTC 侧已冻结）".format(nparam / 1e6))

    # 基线（训练前）
    base = eval_frame_level(model, ctc, va, dim, nrm, device, voc, a.batch)
    print("")
    print("训练前  frame_macro_f1={:.4f}  peak_frames_per_token={:.3f}  pred_O={:.1%}".format(
        base["frame_macro_f1"], base["peak_frames_per_token"], base["pred_O_fraction"]))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    history = []
    best = {"frame_macro_f1": -1.0}
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        np.random.shuffle(tr)
        tot, tot_ce, tot_sm, nb = 0.0, 0.0, 0.0, 0
        for i in range(0, len(tr), a.batch):
            chunk = tr[i: i + a.batch]
            if len(chunk) < 2:
                continue
            fb, lb, lens = make_batch(chunk, dim, nrm, device)
            with torch.no_grad():
                enc = encode_ctc(ctc, fb, lens, device)
            sl = model(enc)
            loss, ce_v, sm_v = model.loss(sl, lb, wt, lens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()); tot_ce += ce_v; tot_sm += sm_v; nb += 1
        sched.step()
        m = eval_frame_level(model, ctc, va, dim, nrm, device, voc, a.batch)
        rec = {
            "epoch": ep,
            "train_loss": tot / max(nb, 1),
            "train_ce": tot_ce / max(nb, 1),
            "train_smooth": tot_sm / max(nb, 1),
            "val_frame_macro_f1": m["frame_macro_f1"],
            "val_peak_frames_per_token": m["peak_frames_per_token"],
            "val_peak_count_ratio": m["peak_count_ratio"],
            "val_pred_O_fraction": m["pred_O_fraction"],
            "f1_B": m["frame_f1_per_class"]["B"],
            "f1_I": m["frame_f1_per_class"]["I"],
        }
        history.append(rec)
        print("ep {:2d}  loss={:.4f}  macroF1={:.4f}  "
              "pfpt={:.2f}  pcr={:.2f}  peakLen={:.0f}  O={:.1%}".format(
                  ep, rec["train_loss"],
                  rec["val_frame_macro_f1"], rec["val_peak_frames_per_token"],
                  rec["val_peak_count_ratio"], m["peak_len_median"],
                  rec["val_pred_O_fraction"]))
        if m["frame_macro_f1"] > best["frame_macro_f1"]:
            best = dict(m)
            best["epoch"] = ep
    dt = time.time() - t0

    print("")
    print("=" * 62)
    print("P2-a 单因素对照结果（CTC 冻结，仅加 frame-level 密集监督）")
    print("=" * 62)
    print("训练前  pfpt={:.3f}  macroF1={:.4f}".format(
        base["peak_frames_per_token"], base["frame_macro_f1"]))
    print("训练后  pfpt={:.3f}  macroF1={:.4f}  (best epoch {})".format(
        best["peak_frames_per_token"], best["frame_macro_f1"], best.get("epoch")))
    print("")
    pfpt_gain = best["peak_frames_per_token"] - base["peak_frames_per_token"]
    print("pfpt 提升  {:+.3f}  (判据：>3.0 才算脱离 peaky 病理)".format(pfpt_gain))
    print("O 占比     {:.1%} -> {:.1%}  (若趋近 100% 说明退化为全 O 平凡解)".format(
        base["pred_O_fraction"], best["pred_O_fraction"]))
    print("耗时       {:.1f} 分钟".format(dt / 60))

    verdict = ("脱离 peaky" if best["peak_frames_per_token"] >= 3.0
               else "部分改善" if pfpt_gain > 0.5 else "无明显改善")
    print("判定       {}".format(verdict))

    rep = {
        "experiment": "P2-a frame-level BIO supervision with frozen CTC",
        "single_factor": True,
        "ctc_frozen": True,
        "checkpoint": a.ckpt,
        "config": vars(a),
        "class_weights": {"O": float(w[0]), "I": float(w[1]), "B": float(w[2])},
        "frame_head_params_M": nparam / 1e6,
        "baseline": base,
        "best": best,
        "pfpt_gain": pfpt_gain,
        "verdict": verdict,
        "history": history,
        "minutes": dt / 60,
    }
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print("收据已落盘: {}".format(out))


if __name__ == "__main__":
    main()
