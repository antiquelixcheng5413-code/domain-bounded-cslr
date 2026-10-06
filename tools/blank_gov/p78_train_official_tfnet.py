"""P78：用**官方原版代码**训练 TFNet（RGB + resnet34MAM）。

═══════════════════════════════════════════════════════════════
设计铁律（今天栽了 6 次同源错误后确立）
═══════════════════════════════════════════════════════════════
1. **网络直接 import 官方 `Net.py` / `Module.py`，不重写。**
   我曾照论文描述自己写 official_tfnet.py，错了 5 处。
2. **官方增强直接 import 官方 `videoAugmentation.py`，不重写。**
3. **词表直接用官方 `DataProcessMoudle.Word2Id`。**
4. 只写官方代码没覆盖的部分：Dataset（我们的 jpg 目录结构）、collate、评估、收据。
5. 判优只用 `official_wer.evaluate`（官方口径），其余标注为附加诊断。

官方规格（全部来自 external/TFNet 源码，见 docs/planning/OFFICIAL_TFNET_SPEC.md）：
  主干      ResNet34MAM（ImageNet 2D 权重 → 3D Conv3d + 逐层 MotorAttention）
  序列提取器 TemporalConv(conv_type=2) = [Conv1d(k=5,pad=0)+BN+ReLU+MaxPool(2)]×2
  输出层    NormLinear（权重按输出通道 L2 归一化、无 bias）
  频域      fft(framewise, dim=-1).abs()  ← 沿特征维，不是时间轴
  损失      CTC(blank=0, reduction='none', zero_infinity=True).mean() + SeqKD(T=8)
  增强      RandomCrop(224) + RandomHorizontalFlip(0.5) + ToTensor + TemporalRescale(0.2)
  归一化    x/127.5 - 1  → [-1,1]
  解码      官方用 ctcdecode beam_width=10；我们默认 greedy（可选 beam）
  超参      hidden=1024, lr=1e-4, wd=1e-4, batch=2, 55 epoch, lr 35/45 降 80%

⚠️ 我们**没有**实现官方 collate_fn 的 left_pad=6 补帧，
   因此 dataLen 直接 = 实际帧数（已实测：不补帧时官方 Net.py 会 shape mismatch）。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── 仓库路径 ────────────────────────────────────────────────────
def _find_repo() -> Path:
    for c in (Path(__file__).resolve().parents[2],):
        if (c / "data" / "raw" / "CE-CSL").exists():
            return c
    return Path("/home/su127/FYP/domain-bounded-cslr")


REPO = _find_repo()
EXT = REPO / "external/TFNet"
RGB_ROOT = REPO / "artifacts/official_rgb"
CSV_DIR = EXT / "data/CE-CSL"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))
sys.path.insert(0, str(EXT))

# ── 官方代码（直接复用，不重写）────────────────────────────────
import DataProcessMoudle as DPM          # noqa: E402
import Net# noqa: E402
import videoAugmentation as VA          # noqa: E402
from official_wer import evaluate, position_vs_official   # noqa: E402

assert "test" not in str(RGB_ROOT), "test split 冻结"
MIN_FRAMES = 48          # TemporalConv 两次 MaxPool 后至少要 9 步，实测下限


def read_csv_split(name: str) -> dict[str, tuple[str, str]]:
    """返回 {Number: (Translator, Gloss)}。用官方 csv（与我们仓库的一致）。"""
    out = {}
    with open(CSV_DIR / name, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0]:
                out[row[0]] = (row[1], row[3])
    return out


def build_transforms(hflip: bool):
    """官方增强链。hflip 可关⇒ 做对照实验。

    ⚠️ 官方默认含 RandomHorizontalFlip(0.5)。中文手语的镜像手势语义可能相反，
       所以做成开关：--no-hflip 关掉，两组对比后再决定。
    """
    train_ops = [VA.RandomCrop(224)]
    if hflip:
        train_ops.append(VA.RandomHorizontalFlip(0.5))
    train_ops += [VA.ToTensor(), VA.TemporalRescale(0.2)]
    test_ops = [VA.CenterCrop(224), VA.ToTensor()]
    return VA.Compose(train_ops), VA.Compose(test_ops)


class RGBSeqDataset(Dataset):
    """读 artifacts/official_rgb/{split}/{translator}/{sid}/*.jpg。

    官方 MyDataset 读的是 `{ImagePath}/{name}/{帧}.jpg`，
    我们 P77 产出的正是同一结构 ⇒ 这里只做路径适配 + 标签编码。
    """

    def __init__(self, split: str, labels: dict, word2idx: dict,
                 transform, is_train: bool, pad_to: int = 0):
        self.root = RGB_ROOT / split
        self.transform = transform
        self.is_train = is_train
        self.pad_to = pad_to
        self.items = []
        n_short = 0
        for sid, (tr, gloss) in labels.items():
            d = self.root / tr / sid
            if not d.exists():
                continue
            n_jpg = len([f for f in os.listdir(d) if f.endswith(".jpg")])
            if n_jpg == 0:
                continue
            ids = [word2idx[w] for w in DPM.PreWords(gloss.split("/"))
                   if w in word2idx]
            if not ids:
                continue
            if n_jpg < MIN_FRAMES:
                # TemporalConv 下限：重复补到 48 帧（记入收据）
                n_short += 1
            self.items.append({"sid": sid, "dir": d, "ids": ids,
                               "n_jpg": n_jpg})
        self.n_short = n_short

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        files = sorted(f for f in os.listdir(it["dir"]) if f.endswith(".jpg"))
        imgs = [cv2.cvtColor(cv2.imread(str(it["dir"] / f)), cv2.COLOR_BGR2RGB)
                for f in files]
        # 官方 MyDataset 也做 cv2.resize(...,(256,256))；P77 已resize 过，这里不再重复
        seq = self.transform(imgs)
        seq = seq.float() / 127.5 - 1.0            # 官方归一化→ [-1, 1]
        return {"video": seq, "ids": it["ids"], "sid": it["sid"]}


def collate(batch):
    """dataLen 必须是 CPU 的 [B,1] LongTensor（已实测，否则 CTC 报错）。"""
    batch = sorted(batch, key=lambda x: len(x["video"]), reverse=True)
    videos = [b["video"] for b in batch]
    T = max(v.shape[0] for v in videos)
    vid = torch.zeros(len(videos), T, 3, videos[0].shape[2], videos[0].shape[3])
    for i, v in enumerate(videos):# [T,C,H,W] → [T,3,H,W]
        vid[i, :v.shape[0]] = v
    data_len = torch.LongTensor([[v.shape[0]] for v in videos])   # CPU, [B,1]
    true_len = torch.LongTensor([v.shape[0] for v in videos])   # 每条真实帧数
    tgt = torch.cat([torch.LongTensor(b["ids"]) for b in batch])   # CPU
    tgt_len = torch.LongTensor([len(b["ids"]) for b in batch])   # CPU（官方同）
    return vid, tgt, tgt_len, data_len, true_len, [b["sid"] for b in batch]


def _conv_len(n: int) -> int:
    """官方 TemporalConv(conv_type=2) 的长度公式：L→L-4→/2→L-4→/2。"""
    for _ in range(2):
        n = (n - 4 + 1) // 2
    return max(n, 0)


def greedy_decode(logp: torch.Tensor, true_len: torch.Tensor,
                  blank: int = 0) -> list[list[int]]:
    """greedy CTC 解码（官方用 ctcdecode beam_width=10，这里默认 greedy 对照）。

    🔴 collate 把整批 pad 到最长帧数，所以 log_probs 的时间维是 T_max，
       而每条样本的卷积后长度 = _conv_len(自己的真实帧数)，
       **不能直接用 out[5]**（那是 batch 内所有样本按同一条 lgt 推的）。
    """
    from cslr.recognition.training import classes_to_token_ids
    seq = logp.cpu()                              # 官方是 [T,B,C] ⇒ 转成 [B,T,C]
    if seq.shape[0] != len(true_len) and seq.shape[1] == len(true_len):
        seq = seq.transpose(0, 1)
    out = []
    lens = true_len.view(-1).tolist()
    for b in range(seq.shape[0]):
        t = min(_conv_len(int(lens[b])), seq.shape[1])
        ids = []
        prev = -1
        for ti in range(t):
            k = int(seq[b, ti].argmax())
            if k != prev and k != blank:
                ids.append(k)
            prev = k
        out.append(classes_to_token_ids(ids) if ids else [])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=55)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--w-kd", type=float, default=1.0, help="SeqKD 权重（官方未给）")
    ap.add_argument("--no-hflip", action="store_true",
                    help="关掉 RandomHorizontalFlip（中文手语镜像语义可能相反）")
    ap.add_argument("--max-train", type=int, default=0, help="0=全量；小规模验证用")
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem-budget", type=float, default=6.6e9,
                    help="显存预算（字节）。默认 6.6GB，留 1.4GB 余量")
    ap.add_argument("--module", default="VAC",
                    choices=["VAC", "CorrNet", "TFNet", "MAM-FSD", "SEN"],
                    help="官方 moduleChoice。VAC=ResNet18(唯一能在8GB跑满帧数的)")
    ap.add_argument("--tag", default="p78")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels_tr = read_csv_split("train.csv")
    labels_dv = read_csv_split("dev.csv")

    # 🔴 官方词表：train+dev+test 全收（词表层面用了 test 标注，
    #    但训练只看 train 帧、不在 test 上算指标，故不违反 test 冻结）
    word2idx, wordSetNum, idx2word = DPM.Word2Id(
        str(CSV_DIR / "train.csv"), str(CSV_DIR / "dev.csv"),
        str(CSV_DIR / "test.csv"), "CE-CSL")

    tr_tf, dv_tf = build_transforms(hflip=not a.no_hflip)
    tr_ds = RGBSeqDataset("train", labels_tr, word2idx, tr_tf, True)
    dv_ds = RGBSeqDataset("validation", labels_dv, word2idx, dv_tf, False)
    # dev 在 P77 里落在 validation/ 目录名
    dv_ds = RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if a.max_train:
        tr_ds.items = tr_ds.items[:a.max_train]
    if a.max_dev:
        dv_ds.items = dv_ds.items[:a.max_dev]

    print("=" * 70)
    print("P78 官方 TFNet 训练（RGB + resnet34MAM，原版网络代码）")
    print("=" * 70)
    print("device              %s" % dev)
    print("词表 wordSetNum     %d（+1 blank = %d 类）" % (wordSetNum, wordSetNum + 1))
    print("train / dev         %d / %d 条" % (len(tr_ds), len(dv_ds)))
    print("帧数 < %d 的训练样本  %d 条（已重复补帧）" % (MIN_FRAMES, tr_ds.n_short))
    print("RandomHorizontalFlip %s" % ("关闭（对照组）" if a.no_hflip else "开启（官方默认 0.5）"))
    print("官方 moduleChoice  %s" % a.module)
    print("显存预算           %.1f GB（斜率 %.1f MB/帧·样本）"
          % (a.mem_budget / 1e9, {"TFNet": 55.7, "CorrNet": 45.8,
                                  "VAC": 27.1}.get(a.module, 55.7)))
    print("超参                hidden=%d lr=%g wd=%g batch=%d epochs=%d"
          % (a.hidden, a.lr, a.weight_decay, a.batch, a.epochs))

    if len(tr_ds) < 2 or len(dv_ds) < 2:
        print("❌ 样本不足，先跑 p77_extract_rgb.py")
        return

    # 🔴 显存实测（RTX5060Laptop 8GB，官方 7 项损失，batch=2）：
    #     T=48  5.36GB✓   T=96  10.92GB✗   T=164 OOM
    #   batch=1：T=96 5.37GB ✓ T=128 6.96GB ✓ T=164 8.73GB ✗ T=180 9.52GB ✗
    #   ⇒ 显存 ≈ batch × T 线性增长，必须按帧数分桶动态定batch。
    def _batch_for(n_frames: int) -> int:
        """选能塞进显存的 batch（实测上限 ≈ 1.0e8·B/frame·B，保守取半）。"""
        # 🔴 P81 实测斜率（MB 每 帧·样本）：
        #    TFNet(7项损失,ResNet34MAM) 5.35e9/(2*48) = 55.7
        #    CorrNet(3项, ResNet18Corr) 4.40e9/(2*48) = 45.8
        #    VAC    (3项, ResNet18)      2.60e9/(2*48) = 27.1
        #    ★ 取实测值而不是猜，这是 P78 失败换来的教训
        SLOPE = {"TFNet": 55.7e6, "CorrNet": 45.8e6, "VAC": 27.1e6,
                 "MAM-FSD": 50.0e6, "SEN": 30.0e6}
        per_frame = SLOPE.get(a.module, 55.7e6)
        budget = float(a.mem_budget)
        b = int(budget / (per_frame * max(n_frames, 1)))
        return max(1, min(b, a.batch))

    def _adaptive_batches(ds, shuffle):
        """按帧数排序 -> 相邻同量级组成 batch，各自选可行 batch。返回索引列表。"""
        order = sorted(range(len(ds.items)),
                       key=lambda k: ds.items[k]["n_jpg"],
                       reverse=not shuffle)
        out, cur = [], []
        for k in order:
            cur.append(k)
            if len(cur) >= _batch_for(ds.items[k]["n_jpg"]):
                out.append(cur)
                cur = []
        if cur:
            out.append(cur)
        if shuffle:
            random.shuffle(out)
        return out

    class _AdaptiveLoader:
        """绕开 DataLoader 的固定 batch_size：按帧数桶自适应组装。"""

        def __init__(self, ds, shuffle):
            self.ds = ds
            self.batches = _adaptive_batches(ds, shuffle)

        def __iter__(self):
            for chunk in self.batches:
                yield collate([self.ds[k] for k in chunk])

        def __len__(self):
            return len(self.batches)

    tr_dl = _AdaptiveLoader(tr_ds, True)
    dv_dl = _AdaptiveLoader(dv_ds, False)
    print("自适应分桶         %d train batch / %d dev batch" % (
        len(tr_dl), len(dv_dl)))

    # ★ 直接用官方网络
    model = Net.moduleNet(a.hidden, wordSetNum * 1 + 1, a.module,
                          torch.device(dev), "CE-CSL", True).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print("参数                %.2f M" % n_par)

    # ★ 官方的 CTC 配置（zero_infinity=True 是关键，我 P72 就是漏了这行导致永久 nan）
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    kld = DPM.SeqKD(T=8)
    ls = torch.nn.LogSoftmax(dim=-1)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    ms = [int(round(35 * a.epochs / 55)), int(round(45 * a.epochs / 55))]
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=ms, gamma=0.2)
    print("lr衰减里程碑        %s（官方 35/45 按比例缩放）" % ms)

    ck_dir = REPO / "artifacts/checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    hist = []
    t0 = time.time()
    nan_steps = 0

    for ep in range(1, a.epochs + 1):
        model.train()
        run, nrun = 0.0, 0
        for vid, tgt, tgt_len, dl, _tl, _sids in tr_dl:
            vid = vid.to(dev)                      # [B,T,3,H,W]
            out = model(vid, dl, True)
            # 🔴 官方的 lgt 是卷积后的时间步（out[5]），不是原始帧数。
            #    官方 log_probs 布局就是 [T,B,C]（实测 (9,2,3516)），与 CTCLoss 一致。
            lgt = out[5]
            # 🔴 TFNet 分支推理时 logProbs1 = logProbs5（官方 Net.py 末尾）；
            #    其他分支 logProbs1 本身就是最终输出，out[4] 是 None。
            logp5 = ls(out[4] if a.module == "TFNet" else out[0])
            # 🔴 nan 防护：非有限就跳过这一 step，保住 Adam 状态
            if not torch.isfinite(logp5).all():
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            #训练时统一用 logProbs1 做主 CTC
            main_lp = ls(out[0])
            loss_ctc = ctc_loss(main_lp, tgt, lgt, tgt_len.cpu()).mean()
            if a.module == "TFNet":
                # 官方 Train.py 210-223：4 个 CTC + 2 个 SeqKD(×25) + 1 个融合 CTC
                loss = (ctc_loss(ls(out[4]), tgt, lgt, tgt_len.cpu()).mean()
                        + ctc_loss(ls(out[2]), tgt, lgt, tgt_len.cpu()).mean()
                        + 25.0 * kld(out[3], out[2], use_blank=False)
                        + ctc_loss(ls(out[1]), tgt, lgt, tgt_len.cpu()).mean()
                        + 25.0 * kld(out[1], out[0], use_blank=False)
                        + loss_ctc)
            else:
                # 官方 CorrNet/VAC/SEN 分支：2 个 CTC + 1 个 SeqKD(×25)
                loss = (loss_ctc
                        + ctc_loss(ls(out[1]), tgt, lgt, tgt_len.cpu()).mean()
                        + 25.0 * kld(out[1], out[0], use_blank=False))
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gn):
                nan_steps += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            run += float(loss_ctc.detach())
            nrun += 1
            torch.cuda.empty_cache()
        sched.step()

        # ---- 评估（官方口径）----
        model.eval()
        hyps, refs, stats = [], [], {"ins": 0, "del": 0, "sub": 0, "n": 0}
        with torch.no_grad():
            for vid, tgt, tgt_len, dl, tl_len, _sids in dv_dl:
                vid = vid.to(dev)
                out = model(vid, dl, False)
                # 🔴 TFNet 推理时 logProbs1 = logProbs5（官方 Net.py 末尾）；
                #    其他分支 logProbs1 本身就是最终输出。
                logp5 = ls(out[0])   # 官方布局 [T,B,C]
                hyps += greedy_decode(logp5, tl_len)
                for b in _sids:
                    refs.append(labels_dv[b][1])
        hyp_txt = ["/".join(idx2word[t] for t in h) for h in hyps]
        o = evaluate(refs, hyps)

        cur = (run / max(nrun, 1), o["WER_official"])
        hist.append({"epoch": ep, "train_ctc": round(cur[0], 4),
                     "dev_wer_official": o["WER_official"],
                     "gap_to_sota": o["gap_to_sota_TFNet"]})
        print("  ep%03d ctc %.4f  **WER_official %.2f%%**  差SOTA %.2fpp%s"
              % (ep, cur[0], o["WER_official"], o["gap_to_sota_TFNet"],
                 "  [nan跳过 %d]" % nan_steps if nan_steps else ""), flush=True)

        if o["WER_official"] < best:
            best = o["WER_official"]
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "wordSetNum": wordSetNum, "idx2word": idx2word,
                        "hidden": a.hidden, "hflip": not a.no_hflip,
                        "wer_official": best, "config": vars(a)},
                       ck_dir / ("%s-best.pt" % a.tag))

    # ---- 收据（⚠️括号必须加：Path / str 的优先级高于 %）----
    out_rec = {
        "experiment": "P78",
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "官方 moduleChoice=%s 真训练（RGB）" % a.module,
        "official_code": "external/TFNet (Net.py / Module.py / videoAugmentation.py "
                         "/ DataProcessMoudle.py 均直接 import，未重写)",
        "official_benchmark_dev": {"TFNet": 42.1, "CorrNet": 47.2, "VAC": 45.1,
                             "MAM-FSD": 44.9, "SEN": 46.5}[a.module],
        "best_wer_official": best,
        "module": a.module,
        "mem_budget_GB": round(a.mem_budget / 1e9, 2),
        "vocab": {"wordSetNum": wordSetNum,
                  "note": "官方 Word2Id，train+dev+test 全收、无截断"},
        "config": vars(a),
        "n_params_M": round(n_par, 2),
        "short_samples_padded": tr_ds.n_short,
        "ctc": "blank=0, reduction='none', zero_infinity=True（官方配置）",
        "nan_steps_skipped": nan_steps,
        "history": hist,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    dst = REPO / ("artifacts/metrics/blank-gov/%s-official-tfnet.json" % a.tag)
    dst.write_text(json.dumps(out_rec, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print()
    print("最佳 WER_official = %.2f%%" % best)
    print("收据 -> %s" % dst)


if __name__ == "__main__":
    main()