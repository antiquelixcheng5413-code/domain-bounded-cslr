#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P174-step2：同分辨率(160)软标签 KD 训练 VAC@160 学生，对齐冻结 TFNet@160 教师。

与 P172 的唯一变量：教师输入 img_size=160（与学生同分辨率），
消除 P172 跨分辨率(224→160)导致的分布/对齐不匹配噪声。

方法（软标签 KD + CTC，沿用 P172 实现）
  - 学生：VAC(resnet18 2D)@160 RGB（P150 同架构），可训。
  - 教师：P174-step1 缓存的 TFNet@160 logits，冻结、按 sid 查表。
  - 时间对齐：教师 T_t 经 F.interpolate 重采样到学生 T_s，逐帧 KL。
  - 损失：L = λ_ctc·CTC(student) + λ_kd·T^2·KL(student_soft || teacher_soft)。
  - BLANK_BIAS=1.0（VAC 同 P150 口径，P172 误用 0.0 系统性差 ~0.8pp）。
  - 加 --resume 续跑（防 WorkBuddy 重启杀 WSL 丢进度）。

写死判据（见 stage-P174-samelres-kd.md）
  - 成功线：dev WER < 36.81（教师160零样本基线）；强成功：逼近 33.92。
  - 必须比 VAC@160 53.73 低 >=2pp。
  - 前 10 轮哨兵：若 dev 不降且 kd 占比仍极低 -> 提示判死，不跑满。
  - 不在 dev 扫描 lambda/T；教师严格冻结。

用法
  python tools/blank_gov/p174_train_distill.py --lambda-kd 1.0 --epochs 55
  python tools/blank_gov/p174_train_distill.py --resume artifacts/checkpoints/p174-distill-best.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(REPO / "external/TFNet"))
sys.path.insert(0, str(REPO / "src"))

import p78_train_official_tfnet as P  # noqa: E402
from official_wer import evaluate  # noqa: E402

CACHE = REPO / "artifacts/cache/p174_teacher"
ARR = CACHE / "teacher_logits"
META = CACHE / "teacher_meta.json"


def log(*a):
    print("[P174 %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def build_dev_eval(idx2word, labels_dv):
    ls = torch.nn.LogSoftmax(dim=-1)

    def dev_eval(mdl, dv_loader, device):
        mdl.eval()
        refs, hyps, ntok = [], [], 0
        with torch.no_grad():
            for vid, tgt, tgt_len, dl, _tl, sids, _ids in dv_loader:
                vid = vid.to(device)
                out = mdl(vid, dl, False)
                logp = ls(out[0].float().cpu())          # [T,B,C]
                P.BLANK_BIAS = 1.0
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
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lambda-kd", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=4.0)
    ap.add_argument("--lambda-ctc", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--amp-bf16", action="store_true", default=True)
    ap.add_argument("--no-hflip", action="store_true", default=False)
    ap.add_argument("--tag", type=str, default="p174-distill")
    ap.add_argument("--resume", type=str, default="")
    a = ap.parse_args()

    dev = "cuda"
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    vocab = wordSetNum + 1
    labels_dv = P.read_csv_split("dev.csv")

    meta = json.loads(META.read_text(encoding="utf-8"))

    model = P.Net.moduleNet(a.hidden, vocab, "VAC", torch.device(dev),
                           "CE-CSL", True).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log("学生 VAC@%d  参数 %.2f M" % (a.img_size, n_par))

    P.OFFICIAL_PAD = True
    P.RGB_FIX = True
    tr_tf, dv_tf = P.build_transforms(hflip=not a.no_hflip,
                                      img_size=a.img_size)
    tr_ds = P.RGBSeqDataset("train", P.read_csv_split("train.csv"),
                           word2idx, tr_tf, True)
    dv_ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    tr_ld = torch.utils.data.DataLoader(
        tr_ds, batch_size=a.batch, shuffle=True,
        collate_fn=P.collate, num_workers=0, drop_last=False)
    dv_ld = torch.utils.data.DataLoader(
        dv_ds, batch_size=1, shuffle=False,
        collate_fn=P.collate, num_workers=0)

    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    ls = torch.nn.LogSoftmax(dim=-1)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr,
                          weight_decay=a.weight_decay)
    T = a.epochs
    ms = [int(round(35 * T / 55)), int(round(45 * T / 55))]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms,
                                                 gamma=0.2)
    dev_eval = build_dev_eval(idx2word, labels_dv)
    log("lr 里程碑 %s（T=%d）；λ_ctc=%.2f λ_kd=%.2f T=%.1f"
        % (ms, T, a.lambda_ctc, a.lambda_kd, a.temp))

    best = float("inf")
    hist = []
    start_ep = 1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        sched.load_state_dict(ck["sched_state"])
        best = float(ck.get("best_dev", "inf"))
        hist = list(ck.get("history", []))
        start_ep = int(ck["epoch"]) + 1
        log("续跑：从 ep%d 起（载入 %s，best=%.2f%%）"
            % (ck["epoch"], a.resume, best))

    t0 = time.time()
    for ep in range(start_ep, a.epochs + 1):
        model.train()
        run_ctc, run_kd, nstep = 0.0, 0.0, 0
        for vid, tgt, tgt_len, dl, _tl, sids, _ids in tr_ld:
            vid = vid.to(dev)
            with torch.amp.autocast("cuda", enabled=a.amp_bf16,
                                    dtype=torch.bfloat16):
                out = model(vid, dl, True)
            logits = out[0].float()                  # [T_s,B,C]
            lgt = out[5]                             # [B,1] 卷积后长度
            main_lp = ls(logits)
            loss_ctc = ctc(main_lp, tgt, lgt,
                           tgt_len.cpu()).mean() * a.lambda_ctc

            loss_kd = torch.zeros(1, device=dev)
            if a.lambda_kd > 0:
                B = logits.shape[1]
                T_s = logits.shape[0]
                stu_soft = F.log_softmax(logits / a.temp, dim=-1)
                teach_soft = torch.zeros_like(stu_soft)
                for b in range(B):
                    sid = str(sids[b])
                    _sk, T_raw, _cT, _dl = meta["index"][sid]
                    tlog = torch.from_numpy(
                        np.load(str(ARR / ("%s.npy" % sid)))
                        .astype("float32")).to(dev)
                    if tlog.shape[0] != T_s:
                        tlog = F.interpolate(
                            tlog.transpose(0, 1).unsqueeze(0), size=T_s,
                            mode="linear", align_corners=False
                        ).squeeze(0).transpose(0, 1)
                    teach_soft[:, b, :] = F.softmax(tlog / a.temp, dim=-1)
                kdt = F.kl_div(stu_soft, teach_soft, reduction="none").sum(-1)
                mask = (torch.arange(T_s, device=dev).unsqueeze(1)
                        < lgt.to(dev).view(1, -1)).float()
                loss_kd = (kdt * mask).sum() / mask.sum().clamp_min(1.0)
                loss_kd = loss_kd * (a.temp ** 2) * a.lambda_kd

            loss = loss_ctc + loss_kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_ctc += float(loss_ctc); run_kd += float(loss_kd); nstep += 1
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
                "history": hist, "lambda_kd": a.lambda_kd, "temp": a.temp,
                "lambda_ctc": a.lambda_ctc, "img_size": a.img_size,
            }, str(REPO / "artifacts/checkpoints/p174-distill-best.pt"))
        if ep % 5 == 0:
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(), "best_dev": best,
                "history": hist,
            }, str(REPO / ("artifacts/checkpoints/p174-distill-ep%02d.pt" % ep)))
        log("ep%02d  ctc=%.3f kd=%.3f  dev=%.2f%%(gen%.1f) %s  (%.0fs)"
            % (ep, run_ctc / max(nstep, 1), run_kd / max(nstep, 1),
               dw, gen, flag, time.time() - t0))
        if ep == 10:
            log("⚠ 早期哨兵 ep10：dev=%.2f%% best=%.2f%% kd_avg=%.3f"
                % (dw, best, run_kd / max(nstep, 1)))

    log("✅ 训练结束：best dev=%.2f%%  （目标<36.81；教师160零样本=36.81；"
        "教师224=33.92；VAC@160 基线=53.73）" % best)
    out = REPO / ("artifacts/metrics/blank-gov/%s.json" % a.tag)
    out.write_text(json.dumps({
        "tag": a.tag, "epochs": a.epochs, "img_size": a.img_size,
        "lambda_ctc": a.lambda_ctc, "lambda_kd": a.lambda_kd, "temp": a.temp,
        "best_dev": best, "history": hist,
        "teacher": "TFNet@160 (P174cache)", "student": "VAC@160",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)


if __name__ == "__main__":
    main()
