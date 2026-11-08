#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P172-step2：知识蒸馏训练 VAC@160 学生，对齐冻结 TFNet@224 教师。

方法（CRKD 式响应级/分类特征蒸馏 + CTC）
  - 学生：VAC(resnet18 2D)@160 RGB（P150 同架构），可训。
  - 教师：P172-step1 缓存的 TFNet@224 logits，冻结、按 sid 查表。
  - 时间对齐：教师按 canonical_T=_conv_len(dl) 从前截断，与学生 out[5] 同长，逐帧对齐。
  - 损失：L = λ_ctc·CTC(student) + λ_kd·T²·KL(student_soft ‖ teacher_soft)，T=4。
  - λ_kd=0 即纯 VAC@160 对照（隔离 KD 增量）。

写死判据（见 stage-P172-distill-crkd.md）
  - 成功线：dev WER < 36.81（教师160零样本基线）；强成功：逼近 33.92。
  - 必须比 VAC@160 53.73 低 ≥2pp 且配对胜率≥75%。
  - 不在 dev 扫描 λ/T；教师严格冻结。

用法
  python tools/blank_gov/p172_train_distill.py --lambda-kd 1.0 --epochs 55
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

CACHE = REPO / "artifacts/cache/p172_teacher"
ARR = CACHE / "teacher_logits"
META = CACHE / "teacher_meta.json"


def log(*a):
    print("[P172 %s] %s" % (time.strftime("%H:%M:%S"),
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
    ap.add_argument("--tag", type=str, default="p172-distill")
    a = ap.parse_args()

    dev = "cuda"
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    vocab = wordSetNum + 1
    labels_dv = P.read_csv_split("dev.csv")

    # --- 教师缓存（按样本 .npy）---
    meta = json.loads(META.read_text(encoding="utf-8"))

    # --- 学生 ---
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
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
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
                T_s = logits.shape[0]                      # batch 级时间维
                stu_soft = F.log_softmax(logits / a.temp, dim=-1)
                teach_soft = torch.zeros_like(stu_soft)    # [T_s,B,C]
                for b in range(B):
                    sid = str(sids[b])
                    _sk, T_raw, _cT, _dl = meta["index"][sid]
                    tlog = torch.from_numpy(
                        np.load(str(ARR / ("%s.npy" % sid)))
                        .astype("float32")).to(dev)        # [T_t,vocab] T_t=教师T_raw
                    # 3D教师(T_t) vs 2D学生(T_s) 下采样率不同 → 时间维重采样对齐
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
            }, str(REPO / "artifacts/checkpoints/p172-distill-best.pt"))
        # 周期安全存盘（防 WorkBuddy 重启杀 WSL 丢进度）
        if ep % 5 == 0:
            torch.save({
                "epoch": ep, "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "sched_state": sched.state_dict(), "best_dev": best,
                "history": hist,
            }, str(REPO / ("artifacts/checkpoints/p172-distill-ep%02d.pt" % ep)))
        log("ep%02d  ctc=%.3f kd=%.3f  dev=%.2f%%(gen%.1f) %s  (%.0fs)"
            % (ep, run_ctc / max(nstep, 1), run_kd / max(nstep, 1),
               dw, gen, flag, time.time() - t0))

    log("✅ 训练结束：best dev=%.2f%%  （目标<36.81；教师160零样本=36.81；"
        "教师224=33.92；VAC@160 基线=53.73）" % best)
    out = REPO / ("artifacts/metrics/blank-gov/%s.json" % a.tag)
    out.write_text(json.dumps({
        "tag": a.tag, "epochs": a.epochs, "img_size": a.img_size,
        "lambda_ctc": a.lambda_ctc, "lambda_kd": a.lambda_kd, "temp": a.temp,
        "best_dev": best, "history": hist,
        "teacher": "TFNet@224 (P172cache)", "student": "VAC@160",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)


if __name__ == "__main__":
    main()
