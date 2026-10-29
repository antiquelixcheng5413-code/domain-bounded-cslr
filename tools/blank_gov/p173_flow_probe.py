#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P173：运动线索探针 —— 把前后帧差作为额外通道拼进 VAC@160（3->5 通道）。

动机（为什么值得一试）
  - 诊断结论：我们落后 20pp 主因是骨干判别力不足（峰锐度 0.624 vs 官方 0.864）。
  - 强骨干(TFNet 3D MAM)在 8G 卡 ~6h/轮不可行（P172 蒸馏也证死）。
  - 轻量替代：给 2D 骨干显式喂运动线索（帧差），看是否补一点判别力。
    （VAC 已有 TemporalConv+BiLSTM 隐式建模时序，故这是“冗余但有否增益”的探针，预期小。）

方法（最小侵入）
  - monkeypatch torchvision.models.resnet18：conv1 由 3->5 入通道，拷贝前 3 通道 ImageNet 预训练权重，后 2 通道随机（Kaiming）。
  - 数据集（FlowRGBSeqDataset）：每帧拼 [RGB, diff_prev, diff_next] -> 5 通道；RGB 归一到 [-1,1]，diff 归一到 ~[-1,1]*DIFF_SCALE(=2)。
  - 学生 VAC(resnet18 2D)@160 RGB，从零训（同 P150 条件：lr=1e-4, wd=1e-4, batch=2, MultiStepLR[35,45]@55, bf16）。
  - 不挂 word head（P150 的 53.73 含 word_head 0.3 正则；本探针先看“通道增量”，结论会注明此 caveat）。

写死判据（启动前）
  - 与 P150 早期轨迹对照（P150: ep1=81.56, ep4=75.55, ep8=70.06, 终 53.73）。
  - 若 P173 早期轨迹明显优于 P150 同期（>=2pp 且持续），说明运动线索有用 -> 上 word_head 跑满。
  - 若持平/更差 -> 运动线索对 VAC 无益，定稿 A（发布权重零样本 35.66%）。

用法
  python tools/blank_gov/p173_flow_probe.py --epochs 55 --tag p173-flow
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as _tvmodels

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external" / "TFNet"))
sys.path.insert(0, str(REPO / "src"))

import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402

DIFF_SCALE = 2.0
CACHE = REPO / "artifacts" / "checkpoints"


def log(*a):
    print("[P173 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


# ---- monkeypatch resnet18 -> 5 入通道（拷贝 RGB 预训练权重） ----
_orig_r18 = _tvmodels.resnet18


def _r18_5ch(pretrained=True, progress=True, **kwargs):
    m = _orig_r18(pretrained=pretrained, progress=progress, **kwargs)
    old = m.conv1
    new = torch.nn.Conv2d(5, old.out_channels, old.kernel_size,
                          old.stride, old.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = old.weight          # 拷贝 ImageNet RGB 预训练
        # 通道 3,4 保留默认 Kaiming 初始化
    m.conv1 = new
    return m


_tvmodels.resnet18 = _r18_5ch


# ---- 动态通道数的 collate（p78 的 collate 把视频硬编码成 3 通道，5 通道会撑爆） ----
def flow_collate(batch):
    batch = sorted(batch, key=lambda x: len(x["video"]), reverse=True)
    videos = [b["video"] for b in batch]
    C = videos[0].shape[1]
    if P.OFFICIAL_PAD:
        Ls = [((v.shape[0] + 3) // 4) * 4 + 12 for v in videos]
        T = max(Ls)
        vid = torch.zeros(len(videos), T, C, videos[0].shape[2],
                         videos[0].shape[3])
        for i, v in enumerate(videos):
            vv = torch.cat([v[0][None].expand(6, -1, -1, -1),
                            v,
                            v[-1][None].expand(T - 6 - v.shape[0], -1, -1, -1)],
                           dim=0)
            vid[i, :vv.shape[0]] = vv
        data_len = torch.LongTensor([[l] for l in Ls])
        true_len = torch.LongTensor([v.shape[0] for v in videos])
    else:
        T = max(v.shape[0] for v in videos)
        vid = torch.zeros(len(videos), T, C, videos[0].shape[2],
                         videos[0].shape[3])
        for i, v in enumerate(videos):
            vid[i, :v.shape[0]] = v
        data_len = torch.LongTensor([[v.shape[0]] for v in videos])
        true_len = torch.LongTensor([v.shape[0] for v in videos])
    tgt = torch.cat([torch.LongTensor(b["ids"]) for b in batch])
    tgt_len = torch.LongTensor([len(b["ids"]) for b in batch])
    return (vid, tgt, tgt_len, data_len, true_len,
            [b["sid"] for b in batch], [b["ids"] for b in batch])


# ---- 5 通道数据集（RGB + 前后帧差） ----
class FlowRGBSeqDataset(P.RGBSeqDataset):
    def __getitem__(self, i):
        it = self.items[i]
        files = sorted(f for f in P.os.listdir(it["dir"]) if f.endswith(".jpg"))
        if P.RGB_FIX:
            imgs = [P.cv2.imread(str(it["dir"] / f)) for f in files]
        else:
            imgs = [P.cv2.cvtColor(P.cv2.imread(str(it["dir"] / f)),
                                   P.cv2.COLOR_BGR2RGB) for f in files]
        T = len(imgs)
        imgs_f = [x.astype(np.float32) for x in imgs]
        g = [x.mean(-1, keepdims=True) for x in imgs_f]   # 灰度 [H,W,1]
        diff_prev = ([np.zeros_like(g[0])]
                     + [g[t] - g[t - 1] for t in range(1, T)])
        diff_next = ([g[t + 1] - g[t] for t in range(T - 1)]
                     + [np.zeros_like(g[0])])
        imgs5 = [np.concatenate([imgs_f[t], diff_prev[t], diff_next[t]], axis=-1)
                 for t in range(T)]          # 每个 [H,W,5] float（RGB3+差2）
        seq = self.transform(imgs5)          # VA.Compose：空间增强(+ToTensor)
        seq = seq.float()
        rgb = seq[:, :3] / 127.5 - 1.0
        diff = (seq[:, 3:] / 255.0 * DIFF_SCALE).clamp(-1.0, 1.0)
        seq = torch.cat([rgb, diff], dim=1)  # [T,5,H,W]
        return {"video": seq, "ids": it["ids"], "sid": it["sid"]}


def build_dev_eval(idx2word, labels_dv):
    ls = torch.nn.LogSoftmax(dim=-1)

    def dev_eval(mdl, dv_loader, device):
        mdl.eval()
        refs, hyps, ntok = [], [], 0
        with torch.no_grad():
            for vid, tgt, tgt_len, dl, _tl, sids, _ids in dv_loader:
                vid = vid.to(device)
                out = mdl(vid, dl, False)
                logp = ls(out[0].float().cpu())
                P.BLANK_BIAS = 0.0
                hyps += P.greedy_decode(logp, dl, idx2word)
                ntok += sum(len(q) for q in hyps[-len(sids):])
                for b in sids:
                    refs.append(labels_dv[b][1])
        o = evaluate(refs, hyps)
        gen = 100.0 * ntok / max(o["ref_tokens"], 1)
        mdl.train()
        return o["WER_official"], gen
    return dev_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tag", type=str, default="p173-flow")
    ap.add_argument("--resume", type=str, default=None,
                   help="续跑：载入该 checkpoint 的模型/优化器/调度器，从 epoch+1 继续")
    a = ap.parse_args()

    dev = "cuda"
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    vocab = wordSetNum + 1
    labels_dv = P.read_csv_split("dev.csv")

    model = P.Net.moduleNet(a.hidden, vocab, "VAC", torch.device(dev),
                           "CE-CSL", True).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log("学生 VAC@%d + 5ch(帧差)  参数 %.2f M（conv1 已 3->5，前3ch 预训练）"
        % (a.img_size, n_par))
    assert model.conv2d.conv1.in_channels == 5, "conv1 补丁未生效！"

    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    tr_tf, dv_tf = P.build_transforms(hflip=False, img_size=a.img_size)
    tr_ds = FlowRGBSeqDataset("train", P.read_csv_split("train.csv"),
                              word2idx, tr_tf, True)
    dv_ds = FlowRGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    tr_ld = torch.utils.data.DataLoader(
        tr_ds, batch_size=a.batch, shuffle=True,
        collate_fn=flow_collate, num_workers=0, drop_last=False)
    dv_ld = torch.utils.data.DataLoader(
        dv_ds, batch_size=1, shuffle=False,
        collate_fn=flow_collate, num_workers=0)

    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    ls = torch.nn.LogSoftmax(dim=-1)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr,
                          weight_decay=a.weight_decay)
    T = a.epochs
    ms = [int(round(35 * T / 55)), int(round(45 * T / 55))]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.2)
    dev_eval = build_dev_eval(idx2word, labels_dv)
    log("lr 里程碑 %s（T=%d）" % (ms, T))

    best = float("inf")
    hist = []
    start_ep = 1
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        sched.load_state_dict(ck["sched_state"])
        best = ck.get("best_dev", float("inf"))
        hist = list(ck.get("history", []))
        start_ep = int(ck["epoch"]) + 1
        log("续跑：从 ep%d 起（载入 %s，best=%.2f%%）"
            % (start_ep - 1, a.resume, best))
    t0 = time.time()
    for ep in range(start_ep, a.epochs + 1):
        model.train()
        run_ctc, nstep = 0.0, 0
        for vid, tgt, tgt_len, dl, _tl, sids, _ids in tr_ld:
            vid = vid.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(vid, dl, True)
            logits = out[0].float()
            lgt = out[5]
            loss_ctc = ctc(ls(logits), tgt, lgt,
                           tgt_len.cpu()).mean() * 1.0
            opt.zero_grad(set_to_none=True)
            loss_ctc.backward()
            opt.step()
            run_ctc += float(loss_ctc); nstep += 1
        sched.step()
        dw, gen = dev_eval(model, dv_ld, dev)
        hist.append(dw)
        flag = "★best" if dw < best else ""
        if dw < best:
            best = dw
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(), "best_dev": best,
                "history": hist, "img_size": a.img_size, "channels": 5,
            }, str(CACHE / ("%s-best.pt" % a.tag)))
        if ep % 5 == 0:
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(), "best_dev": best,
                "history": hist,
            }, str(CACHE / ("%s-ep%02d.pt" % a.tag)))
        log("ep%02d  ctc=%.3f  dev=%.2f%%(gen%.1f) %s  (%.0fs)"
            % (ep, run_ctc / max(nstep, 1), dw, gen, flag, time.time() - t0))

    log("✅ P173 结束：best dev=%.2f%%（对照 P150=53.73；P150 ep4=75.55/ep8=70.06）"
        % best)
    out = REPO / ("artifacts/metrics/blank-gov/%s.json" % a.tag)
    out.write_text(json_dumps_ok({
        "tag": a.tag, "epochs": a.epochs, "img_size": a.img_size,
        "channels": 5, "diff_scale": DIFF_SCALE,
        "best_dev": best, "history": hist, "student": "VAC@160+flow5ch",
    }))
    log("收据 -> %s" % out)


def json_dumps_ok(d):
    import json
    return json.dumps(d, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
