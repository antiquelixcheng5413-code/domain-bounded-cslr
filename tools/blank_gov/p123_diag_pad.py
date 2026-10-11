"""P123 事故诊断：官方补帧（--official-pad）把 dev WER 打崩的机理验证。

现象：训练 CTC 正常下降（ep30 = 10.91，与 P119 同期 11.98 同量级），
     但 dev WER 卡在 74.81 %（P119 同期 57.14）⇒ 差 17.7 pp。

假设：**训练侧没事，解码侧截断口径错了**
- 训练：CTC 用模型自己的 lgt = _conv_len(data_len)，data_len 含补帧 ⇒ 正确
- 评估：`greedy_decode(logp5, tl_len, ...)` 用的是 **true_len = 真实帧数**，
  开了官方补帧后 data_len = ceil(T/4)*4+12 ≠ T
  ⇒ 解码少截了尾巴（left_pad=6 还把内容整体右移），句尾 gloss 被切掉

验证方法（决定性）：
  同一个 checkpoint、同一批 dev，只换解码截断长度
    A：tl_len（真实帧数，现行口径）
    B：dl（data_len，含官方补帧）
  ⇒ 若 A≈74.8 而 B 大幅下降，则假设成立

⚠️ 无补帧时 data_len == true_len ⇒ B 与 A **完全等价**，改动不污染历史数字。
"""
from __future__ import annotations

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

CK = REPO / "artifacts/checkpoints/p123-official55bf16-last.pt"
IMG = 160
MAX_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 515


def log(*a):
    print("[DIAG %s] %s" % (time.strftime("%H:%M:%S"),
                            " ".join(str(x) for x in a)), flush=True)


def main():
    dev = "cuda"
    labels_dv = P.read_csv_split("dev.csv")
    word2idx, wordSetNum, idx2word = P.DPM.Word2Id(
        str(P.CSV_DIR / "train.csv"), str(P.CSV_DIR / "dev.csv"),
        str(P.CSV_DIR / "test.csv"), "CE-CSL")
    log("词表 %d" % wordSetNum)

    _, dv_tf = P.build_transforms(hflip=False, img_size=IMG)
    ds = P.RGBSeqDataset("dev", labels_dv, word2idx, dv_tf, False)
    if MAX_DEV:
        ds.items = ds.items[:MAX_DEV]
    log("dev %d 条" % len(ds))

    # 🔴 与训练时同构：isFlag=True（p78 main 就是这么建的）
    model = P.Net.moduleNet(1024, wordSetNum + 1, "VAC",
                            torch.device(dev), "CE-CSL", True).to(dev)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    # 🔴 键名是 **model_state**（我第一版写成 "model" ⇒ strict=False 静默没载入，
    #    模型随机初始化，WER 155% ⇒ 差点得出反向结论）
    assert "model_state" in ck, "ckpt 键名变了：%s" % list(ck)[:6]
    res = model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    log("载入 %s（epoch %s，strict=True OK）" % (CK.name, ck.get("epoch")))

    ls = torch.nn.LogSoftmax(dim=-1)

    def run_eval(pad_on: bool, use_data_len: bool):
        P.OFFICIAL_PAD = pad_on
        hyps, refs, lens_a, lens_b = [], [], [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(len(ds)):
                batch = P.collate([ds[i]])
                vid, _tgt, _tl, dl, tl_len, _sids, _ids = batch
                vid = vid.to(dev)
                out = model(vid, dl, False)
                logp = ls(out[0])                      # [T,1,C]
                L = dl if use_data_len else tl_len
                lens_a.append(int(tl_len.view(-1)[0]))
                lens_b.append(int(dl.view(-1)[0]))
                hyps += P.greedy_decode(logp, L, idx2word)
                for b in _sids:
                    refs.append(labels_dv[b][1])
        o = evaluate(refs, hyps)
        log("pad=%-5s 截断用=%-9s  WER=%.2f%%  ins=%d del=%d sub=%d ref=%d  (%.0fs)"
            % (pad_on, "data_len" if use_data_len else "true_len",
               o["WER_official"], o.get("ins", -1), o.get("del", -1),
               o.get("sub", -1), o.get("n", o.get("ref_tokens", -1)),
               time.time() - t0))
        return o

    print()
    log("──── A/B 对照（同一权重，只换解码截断）────")
    o1 = run_eval(pad_on=True, use_data_len=False)    # 现行（坏）
    o2 = run_eval(pad_on=True, use_data_len=True)     # 修好
    print()
    log("──── 交叉验证：关掉官方补帧（权重是在补帧下训练的，仅看口径）────")
    o3 = run_eval(pad_on=False, use_data_len=False)

    print()
    log("结论：ΔWER(修好−现行) = %+.2f pp"
        % (o2["WER_official"] - o1["WER_official"]))

    out = REPO / "artifacts/metrics/blank-gov/p123-pad-diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps({
        "diag": "P123 official-pad 解码截断事故",
        "ckpt": str(CK), "epoch": ck.get("epoch"),
        "n_dev": len(ds),
        "A_pad_true_len": o1, "B_pad_data_len": o2,
        "C_nopad_true_len": o3,
        "delta_B_minus_A_pp": round(o2["WER_official"] - o1["WER_official"], 2),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("收据 -> %s" % out)


if __name__ == "__main__":
    main()
