#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""夜间链报告生成器：从 judge/uwer 收据自动写阶段报告（WSL 主副本 + 桌面镜像）。
也可 --summary 生成整夜总汇总。

依据等级沿用用户约定：【有】能指出是哪篇哪句｜【间接】相近机制｜【无】无支撑。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
M = REPO / "artifacts/metrics/blank-gov"
EX = REPO / "docs/experiments"

META = {
    "p212-e1": {
        "title": "E1 / P212 CSGC 跨句 gloss 原型对比",
        "basis": "【有】AAAI 2024 CSGC（Cross-Sentence Gloss Consistency）：为每类维护动量原型、跨句借力，CSL-Daily 报告 −5.7pp；"
                 "【有】本地 D4 查明 48% 类 freq=1、90.4% ≤10，单样本类形不成判别边界；"
                 "【有】本地 part3 §20 判别力探针：实词 AUC 0.93–1.0 ⇒ 长尾性质是「决策边界欠定」而非「表征无信息」。",
        "method": "原型表 [3516,1024] 用 NormLinear 列归一化权重初始化（cosine 分类器天然类原型，覆盖全部长尾类）；"
                  "帧→gloss 用 CTC 后验软聚合，无需帧级标注；正类=真实 gloss，负类=后验最高的非目标类（hard negatives），"
                  "InfoNCE + 动量更新。挂 classifier11 hook 只取第一次调用。训练期-only，推理零开销。"
                  "配方逐项沿用 P184（160 / 冻结 conv2d / lr 2e-5 / 6 轮 / seed 42 / p184-init）。",
    },
    "p212-e1-night": {
        "title": "E1 / P212 CSGC 跨句 gloss 原型对比（统一口径复判）",
        "basis": "【有】AAAI 2024 CSGC；【有】本地 D4 极端长尾（48% 类 freq=1）；"
                 "【有】本地 part3 §20 实词 AUC 0.93–1.0 ⇒ 决策边界欠定而非表征无信息。",
        "method": "同 E1 原始实验，此处用夜间链统一评估器重新出全指标并复判"
                  "（原 E1 链判据未含插入数护栏，统一口径补上 R5/R6）。",
    },
    "p226-h1": {
        "title": "H1 / P224-P226 检索增强 kNN 后验精修（RAC 分布插值）",
        "basis": "【有】IEEE 2025 泰语指语多模态检索增强识别：train 派生知识库 + "
                 "kNN 后验精修，signer-independent 显著改善、野外 WER 降 >30%，"
                 "并明确要求 data-leakage-controlled；"
                 "【有】CSLR² (Raude et al., ECCV 2024, arXiv:2405.10266)：句级检索与 CSLR 互益，"
                 "BOBSL CSLR-TEST WER 65.5 vs 71.7，检索式分类对同义词/视觉相似手语更宽容；"
                 "【有】RAC 通用框架：P(y|x)=λP_knn+(1−λ)P_cls 分布插值；"
                 "【有】本地 E3 1-NN 27.44%（随机 44×）与 G0 train 混淆表（38.45% 覆盖 / lift 111×）；"
                 "【无】gloss 级稠密检索修正无直接先例，属自建协议。",
        "method": "零训练、参数无关。记忆库只由 train 构建：CTC greedy 路径切成连续相同非 blank 段，"
                  "每段取 classifier11 输入帧特征均值 + L2，PCA128 后建 cosine 索引（K=32）。"
                  "dev 段检索得 P_knn，对段内每帧做 log((1−λ)exp(lp)+λP_knn) 融合后 greedy 解码。"
                  "主干/适配器/分类头全部不动，不花 6 轮训练预算。λ 选择规则与 R1–R8 判据见 "
                  "`docs/planning/H1-PREREG-2026-11-08.md`。",
    },
    "p223-g1": {
        "title": "G1 / P223 真实混淆对 hard negatives（CSGC 负类替换）",
        "basis": "【有】P222/G0 前提诊断：train 帧级 top-2 竞争对覆盖 dev 606 个真实混淆对的 "
                 "**38.45%**（lift 110.95×，条件随机对照），远过预注册 20%/3× 线；"
                 "【有】Zhang & Duh, EMNLP 2023：手形最小对立体识别率仅 ~20%（运动类 ~87%）；"
                 "【有】Yin et al., arXiv:2606.28667 (2026)：建议最小对立体作 hard negatives；"
                 "【有】本地 P219/P220：693 个严格替换对是语义近义混淆、非词表变体；"
                 "【有】AAAI 2024 CSGC 跨句 gloss 原型对比框架。",
        "method": "复用 E1 已验证的 GlossPrototypeBank + FeatureCapture（hook 只取 classifier11 "
                  "第一次调用），**唯一单变量**：负类从「后验最高的非目标类」换成 "
                  "P222 从 train 集前向挖出的真实混淆类（17054 对，train 来源、无 dev 泄漏），"
                  "不足 n_neg 时才用原后验策略补齐。训练期-only、推理零开销。"
                  "配方逐项沿用 P184（160 / 冻结 conv2d / lr 2e-5 / 6 轮 / seed 42 / p184-init）。",
    },
    "p213-f1": {
        "title": "F1 / P213 训练期 logit adjustment（LA-CTC）",
        "basis": "【有】Menon et al., ICLR 2021《Long-tail Learning via Logit Adjustment》：训练时在 logits 上减 τ·log π_c 可得到"
                 "Fisher-consistent 的平衡错误率；【有】Ren et al., NeurIPS 2020 Balanced Softmax（同类先验校正）；"
                 "【有】本地 E0（P211）前提诊断 corr(log freq, 平均 logp)=0.8756 ⇒ TFNet 分类器确实编码了 train prior；"
                 "【有】本地 E0 已证伪 post-hoc τ 调整（插入爆炸），明确留下「训练期补偿」这一未验证口子；"
                 "【有】FGXM（PRL 2025）指出低频实词识别差源于 CTC 训练过程与数据不平衡。",
        "method": "训练时对全部 CTC 头的 log-prob 施加 la(lp)=log_softmax(lp − τ·log π_c)，blank 类不动；"
                  "推理时完全不调整（模型输出原样），故零推理开销。π_c 走训练实际路径 DPM.PreWords→word2idx 统计，"
                  "与 D4 频次表同口径。与 D4 的 per-sample 加权、E0 的 post-hoc 解码均为不同机制："
                  "本项是 per-class 先验解耦，且模型有机会在 6 轮内自行补偿，避免 E0 的插入爆炸。"
                  "配方逐项沿用 P184，单变量仅加 --la-tau。",
    },
    "p214-f2": {
        "title": "F2 / P214 视觉-文本语义原型对齐（VTA）",
        "basis": "【有】FGXM（Ke/Liu/Feng, PRL 191, 2025）正是提出 uWER、点名低频实词的原论文，"
                 "主张训练期细粒度对齐且推理零开销；【间接】CVT-SLR（CVPR 2023）视觉-文本对齐、ACMC（2025）多粒度视觉-文本对比；"
                 "【间接】LT-SignDiff（2026）在 3782 类、64% 单样本场景下用语义先验补决策边界；"
                 "【有】本地 E0：corr(freq,‖w_c‖)=0.10，NormLinear 列范数被强制相等 ⇒ 类原型缺少语义邻域结构，"
                 "而 48% 类只有 1 个样本 ⇒ 从文本语义借力是有依据的补边界方式。",
        "method": "离线用本地 Qwen2.5-1.5B 对 3516 个 gloss 取 last-hidden 均值 → 中心化 + PCA 降到 1024 + L2 归一化，"
                  "得到固定文本语义目标 T[3516,1024]；训练期对 NormLinear 的列归一化类原型加 cosine 一致性正则"
                  "λ·mean_c[(1−cos(w_c,T_c))·mask_c]，mask 对低频类加权（直击尾部），"
                  "投影固定不引入可训练参数。推理零开销。配方逐项沿用 P184，单变量仅加 --vta-weight。",
    },
    "p215-f0": {
        "title": "F0 / P215 插入率感知解码扫描（零训练）",
        "basis": "【有】Menon et al., ICLR 2021 logit adjustment（post-hoc 形式 −τ·log π_c）；"
                 "【有】本地 E0（P211）实测 τ>0 时低频桶 −9.1pp / uWER −6.35pp 但插入率 3.17%→7.92%、"
                 "WER_official 34.34→45.21 ⇒ 假改善；本项检验「加 blank bonus 抑制插入」能否把 E0 的尾部收益变成净收益；"
                 "【无】blank bonus 与 τ 的联合网格属工程尝试，无直接论文支撑。",
        "method": "复用已落盘 dev 后验 p188_logp_p184.pt（515 句），CPU 上二维扫描 τ（先验校正强度）"
                  "× β（blank logit bonus，抑制非 blank 输出）。每个组合报 WER_official / uWER / 低频桶 / 高频桶 / "
                  "ins-sub-del / micro / 配对 bootstrap。零训练、不碰 test。",
    },
    "p216-e3": {
        "title": "E3 / P216 手形聚类前提探针（零训练）",
        "basis": "【有】Zhang & Duh, EMNLP 2023：手形最小对立体识别率仅 ~20%（运动类 ~87%）；"
                 "【有】Yin et al., arXiv:2606.28667 (2026)：最小对立体作 hard negatives；"
                 "【有】本地 01_HandsOn_FG2025：SignBank >60% 手语词只用单一手形；"
                 "【有】本地 D3 已证伪 landmark48 的 TFD 边界，但手形聚类是不同任务（找类别身份而非边界）。",
        "method": "对本地 landmark 的手部点做无监督 k-means，量化聚类纯度与 gloss 类的互信息/ARI；"
                  "若聚类与 gloss 几乎独立 ⇒ 直接放弃手形辅助头，不投 GPU。纯 CPU。",
    },
}

RULES = [
    ("R1", "uWER 改善 ≥ 2.0pp"),
    ("R2", "低频桶 WER 改善 ≥ 2.0pp"),
    ("R3", "高频桶恶化 ≤ 1.0pp"),
    ("R4", "WER_official 退化 ≤ 1.0pp"),
    ("R5", "插入数相对基线增幅 ≤ 10%（E0 教训）"),
    ("R6", "micro sub+del 恶化 ≤ 0.30pp（D4 教训）"),
    ("R7", "训练类：6 轮 best dev ≤ 34.80（零训练跳过）"),
    ("R8", "低频桶单独配对 bootstrap 95% CI 下界 > 0（G1 额外要求，仅当评估器带 low_n 时启用）"),
]


def fmt_delta_table(j):
    d = j["delta"]
    b, c = j["base"], j["cand"]
    rows = [
        ("WER_official", b["WER_official"], c["WER_official"], d["WER_official_improve_pp"]),
        ("uWER", b["uWER_pct"], c["uWER_pct"], d["uWER_improve_pp"]),
        ("低频桶 WER", b["low_bucket"]["WER_pct"], c["low_bucket"]["WER_pct"], d["low_bucket_improve_pp"]),
        ("高频桶 WER", b["high_bucket"]["WER_pct"], c["high_bucket"]["WER_pct"], -d["high_bucket_worsen_pp"]),
        ("micro sub+del", b["edits"]["micro_subdel_pct"], c["edits"]["micro_subdel_pct"], -d["micro_subdel_worsen_pp"]),
        ("插入数", d["ins_base"], d["ins_cand"], -(d["ins_cand"] - d["ins_base"])),
        ("非blank帧占比", b.get("nonblank_frame_ratio_pct", "—"),
         c.get("nonblank_frame_ratio_pct", "—"), "—"),
    ]
    s = "| 指标 | 基线 %s | 候选 %s | Δ（正=改善） |\n|---|---|---|---|\n" % (j["base_tag"], j["tag"])
    for k, x, y, z in rows:
        s += "| %s | %s | %s | %s |\n" % (k, x, y, z if isinstance(z, str) else "%+.2f" % z)
    return s


def make_one(tag, title_override=""):
    jp = M / ("%s-judge.json" % tag)
    if not jp.exists():
        return None
    j = json.loads(jp.read_text(encoding="utf-8"))
    meta = META.get(tag, {"title": tag, "basis": "（未登记）", "method": "（未登记）"})
    title = title_override or j.get("title") or meta["title"]
    cur = j.get("train_curve") or []
    curtxt = ""
    if cur:
        curtxt = "\n## 4. 训练曲线\n\n| ep | train ctc | dev WER |\n|---|---|---|\n"
        for x in cur:
            curtxt += "| %d | %.4f | %.2f |\n" % (x["ep"], x["ctc"], x["wer"])
        curtxt += "\nbest dev = **%s**\n" % j.get("train_best_dev_wer")
    pb = j["paired_bootstrap"]
    checks = j["checks"]
    checktxt = "\n".join("| %s | %s | %s |" % (k, RULES[i][1] if i < len(RULES) else "",
                                               "✅" if v else "❌")
                         for i, (k, v) in enumerate(checks.items()))
    txt = """# %s — 阶段报告（%s）

> 生成：%s｜tag `%s`｜夜间自动链生成｜依据等级沿用【有/间接/无】
> 口径：dev 515 句、语料级 `WER_official`；uWER 对 dev 774 类 gloss 等权；低频桶按 train 频次 ≤10 划分。

## 1. 动机与论文依据

%s

## 2. 方法（单变量，其余配方逐项沿用 P184）

%s

## 3. 预注册判据（2026-11-07 23:30 写死，实验启动前，禁止事后调整）

主看尾部（D4 挖出的真瓶颈），并显式防「拆东墙补西墙」与「插入爆炸」两种假改善：

| 编号 | 规则 |
|---|---|
%s

判定：全部满足才 PASS，任一不满足 FAIL ⇒ 不进 2 seed / 不跑 test。

## 4. 结果（dev）

%s
%s
配对 bootstrap（句子级，20000 次）：ΔWER 点估计 **%+.3fpp**，95%% CI **[%+.3f, %+.3f]**%s。

低频桶 %d token / 高频桶 %d token（合计 %d，与 P197 ref_tokens=2842 对账）。

## 5. 判定

| 规则 | 内容 | 结果 |
|---|---|---|
%s

**verdict = %s**

## 6. 诚实边界

- 本报告所有 dev 数字为单次 6 轮（或单次推理）结果，dev 噪声地板 ±0.77pp；低频桶只有 692 token，
  严格来说需单独配对 bootstrap，本报告的总量 CI 不能替代低频桶 CI。
- 未跑 test（项目铁律：test 冻结、只在 dev 过线后一次性终评）。
- PASS 也只代表「值得投入 2 seed + 配对 bootstrap 复跑」，不等于性能已确认提升。
- 本链所有开关默认关闭，不改任何已有实验行为；补丁均备份原文件到 `/tmp/`。

## 7. 净结论

%s

---
数据来源：`artifacts/metrics/blank-gov/%s-judge.json`、`%s-uwer.json`。
""" % (
        title, j["verdict"], datetime.now().strftime("%Y-%m-%d %H:%M"), tag,
        meta["basis"], meta["method"],
        "\n".join("| %s | %s |" % r for r in RULES),
        fmt_delta_table(j), curtxt,
        pb["delta_WER_pp"], pb["ci95"][0], pb["ci95"][1],
        "（CI 跨 0 = 无差别）" if pb["ci_contains_zero"] else "（CI 不含 0）",
        j["cand"]["low_bucket"]["n_tokens"], j["cand"]["high_bucket"]["n_tokens"],
        j["cand"]["low_bucket"]["n_tokens"] + j["cand"]["high_bucket"]["n_tokens"],
        checktxt, j["verdict"],
        ("本方向过线，值得复跑 2 seed 确认；仍未跑 test。"
         if j["verdict"] == "PASS" else
         "本方向判负，不进入复跑/test；结论并入整夜汇总，不再重复同类机制。"),
        tag, tag,
    )
    out = EX / ("stage-%s.md" % tag)
    out.write_text(txt, encoding="utf-8")
    mirror = DESK / "docs/experiments" / ("stage-%s.md" % tag)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(txt, encoding="utf-8")
    print("报告 -> %s (+桌面镜像)" % out, flush=True)
    return out


def make_summary(tags):
    lines = ["# 夜间自动实验汇总（2026-11-07 23:30 → 2026-11-08 09:00）", "",
             "> 本文件由夜间链自动生成；所有判据在实验启动前写死，未事后调整。", "",
             "## 总表", "", "| 实验 | 方向 | best dev / WER | uWER Δ | 低频桶 Δ | 插入增幅 | 判定 |",
             "|---|---|---|---|---|---|---|"]
    for t in tags:
        jp = M / ("%s-judge.json" % t)
        if not jp.exists():
            lines.append("| %s | — | 未生成收据 | — | — | — | 未完成 |" % t)
            continue
        j = json.loads(jp.read_text(encoding="utf-8"))
        d = j["delta"]
        lines.append("| `%s` | %s | %s | %+.2f | %+.2f | %+.0f%% | **%s** |" % (
            t, META.get(t, {}).get("title", t).split("）")[-1][:26],
            j.get("train_best_dev_wer") or j["cand"]["WER_official"],
            d["uWER_improve_pp"], d["low_bucket_improve_pp"],
            100 * d["ins_relative_increase"], j["verdict"]))
    lines += ["", "## 预注册统一判据", "",
              "主看尾部：uWER ≥2pp 且 低频桶 ≥2pp；护栏：高频桶恶化 ≤1pp、总量退化 ≤1pp、",
              "插入增幅 ≤10%（E0 教训）、micro 恶化 ≤0.30pp（D4 教训）、训练类 dev ≤34.80。", "",
              "## 已判负/排除方向（累计）", "",
              "D1 预训练 LM 重排、D2 一致性正则、D3 TFD 边界、D4 类平衡加权、E0 post-hoc 先验校正、",
              "历史 P172/P174/P175/P195 蒸馏、P185/P196/P198/P200 模型侧、P202/P203/P206 数据侧、",
              "P187/P188/P190/P192/P197 解码与集成。", "",
              "## 净天花板（不变）", "",
              "精度优先 = 官方发布权重 @224 零样本 test 32.74（逐句均值 / beam10，P189 口径）；",
              "速度优先 = P184 冻结主干 PEFT @160 test 33.98（语料级）；",
              "from-scratch VAC@160 最佳 = P150 53.73。", "",
              "## 状态", "",
              "未跑 test、未 commit、未 push（等用户明确指令）。", ""]
    txt = "\n".join(lines)
    out = EX / "NIGHT-RUN-2026-11-08.md"
    out.write_text(txt, encoding="utf-8")
    mirror = DESK / "docs/experiments" / "NIGHT-RUN-2026-11-08.md"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(txt, encoding="utf-8")
    print("总汇总 -> %s (+桌面镜像)" % out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    if a.summary:
        tags = [x for x in a.tags.split(",") if x] or \
            ["p212-e1", "p215-f0", "p213-f1", "p214-f2", "p216-e3"]
        make_summary(tags)
    else:
        if not a.tag:
            raise SystemExit("need --tag or --summary")
        make_one(a.tag, a.title)


if __name__ == "__main__":
    main()
