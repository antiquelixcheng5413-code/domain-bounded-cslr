"""P64：汇总 91 份实验收据的实测效果，供「论文方法计划表」标注用

目标：把每个方法 → 对应实验 → 实测数字 → 结论（有效/无效/待验）
结构化导出，供计划表引用。**不重新跑实验，只读收据。**
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
D = REPO / "artifacts/metrics/blank-gov"

# 手工标注每份收据对应的方法与结论（来自我这几天的实验记录）
# 只收录「与论文方法可对应」的条目；纯 bug 修复类单列
NOTES = {
    "p0-diagnosis": ("起点诊断", "blank 率 96.98%（argmax），gloss 1118/peak 290",
                     "baseline"),
    "p1-tfd-probe": ("TFD 时频域", "未落地（依赖 RGB）", "待验"),
    "p2-frame-level": ("帧级特征 vs 句级", "未落地", "待验"),
    "p2b-late-fusion": ("后融合", "未落地", "待验"),
    "p3-joint-loss": ("联合损失（CTC+帧级）", "ΔWER ±0.004", "无效"),
    "p5-tdm": ("TDM 动态对齐", "未落地", "待验"),
    "p6-vocab-ablation-long": ("词表消融300/1828/3517",
                               "WER 0.851/0.926/0.955 —— **但 early-stop 伪造（ep2~4 拿 best），结论作废**",
                               "作废待重验"),
    "p7-cross-signer": ("跨 signer 泛化", "signer 间 CV 0.018，LOSO 退化 -0.0025",
                        "无效"),
    "p8-error-attribution": ("误差归因", "工具类", "工具"),
    "p9-wer-bias": ("WER 偏差审计", "工具类", "工具"),
    "p10-oov-fix": ("OOV 处理", "未落地", "待验"),
    "p11-scp-count-prior": ("SCP 词频先验", "未落地", "待验"),
    "p12-scp-validity": ("SCP 有效性检验", "未落地", "待验"),
    "p13-real-candidate-probe": ("候选词探针", "未落地", "待验"),
    "p14-real-constrained": ("受限场景", "未落地", "待验"),
    "p15-modality-ablation": ("模态消融", "未落地", "待验"),
    "p16-output-degeneracy": ("输出退化观察",
                              "**首次记录输出退化**，但未量化贡献", "发现"),
    "p17-root-cause": ("根因分析", "未落地", "待验"),
    "p18-degeneracy-fix": ("退化修复", "未落地", "待验"),
    "p19-overfit-vs-feature": ("过拟合 vs 特征",
                               "1-NN 静态可分性 0.0762 —— 但探针任务设定有误（P36 铁律）",
                               "作废"),
    "p20-sanity-overfit": ("过拟合自检", "未落地", "待验"),
    "p21-ctc-degenerate": ("CTC 退化", "未落地", "待验"),
    "p23-off-by-one": ("🔴 CTC target索引 +1 修复",
                       "**dev 严格 WER 0.976 → 0.702**；8/8样本解码全对", "✅ 关键成功"),
    "p24-after-fix": ("修复后复测", "WER 0.7019 严格口径", "✅ 成功"),
    "p25-readable-output": ("可读输出", "逐句可读化", "✅ 成功"),
    "p26-blank-after-fix": ("blank 率复测",
                            "blank 96.98% → 90.47%（理论健康 79.08%）", "✅ 有效"),
    "p28-training-length": ("加训练轮数 30→150",
                            "dev WER 只动-0.0063，gap 扩到 0.38", "❌ 无效"),
    "p30-capacity": ("容量控制 4 配置",
                     "base 2.88M 最优；tiny 0.30M 差 0.148", "❌ 无效"),
    "p31-feature-layer": ("原始特征层诊断",
                          "双手检出 mean 0.0000缺失；运动量正常；时间std/样本std=2.03",
                          "诊断"),
    "p32-face-extract-feasibility": ("面部特征可行性", "未落地", "待验"),
    "p33-face128-pilot": ("面部 128 点小样本", "未落地", "待验"),
    "p34-face-probe": ("面部扩容 8→128 点", "**1.01x**（无收益）", "❌ 无效"),
    "p35-augmentation": ("数据增强 4 种（shift/mask/scale/combo）",
                         "train WER 升、dev 降；最好 combo **-0.0060**", "❌ 无效"),
    "p36-rgb-probe": ("CLIP RGB 线性探针",
                      "d=0.1006 极弱；heldout 0.0222 < 基线 0.1833", "❌ 无效"),
    "p37-upper-block": ("upper 块（pose+face）",
                        "Cohen d +0.3041 —— **全场最清晰的信号**", "✅ 值得用"),
    "p38-verify-upper-deltas": ("upper/deltas 复核",
                               "body_deltas d=+0.4796 全场最高", "✅ 值得用"),
    "p39-freq-calibration": ("频次校准", "未落地", "待验"),
    "p39-n800": ("样本量扫描 n=800", "见收据", "诊断"),
    "p39-n1500": ("样本量扫描 n=1500", "见收据", "诊断"),
    "p39-n3000": ("样本量扫描 n=3000", "见收据", "诊断"),
    "p40-pilot300": ("RGB 路线小样本 pilot",
                     "rgb+lm0.5314 / rgb_only 0.6949（lm_only 更好）", "❌ RGB 无收益"),
    "p40-full": ("RGB 全量 100 epoch",
                 "lm_only **0.5211** / rgb+lm 0.5314 / rgb_only 0.6949", "✅ lm_only 最好"),
    "p42-lm-final": ("lm_only 全量 100 epoch 最佳",
                     "ep100 **0.5211**（ep90 = 0.5000）；train WER 0.0065；exact 18/514",
                     "✅ 当前最佳"),
    "p43-speed-bugs": ("推理提速 17.8s→4.2s", "**4.2x**，同帧三模型并行逐位一致", "✅ 成功"),
    "p44-wer-audit": ("WER 口径审计",
                      "折叠送分 0.1755；剔 unk 真实 WER 0.7919；逐句等权 2.8774", "✅ 关键"),
    "p44b-norm-mismatch": ("🔴 服务端归一化口径不一致",
                           "训练口径 0.5211 vs 服务口径 **1.2851**", "✅ 修复"),
    "p44c-online-vs-offline": ("🔴 左右手标签恒取 [0]",
                               "检出率 0.44/0.54；输出种类 **1 种**", "✅ 修复"),
    "p45-diagnosis": ("输出坍缩诊断", "前5 词占真实词输出 49.5%", "诊断"),
    "p45b-signer-overlap": ("signer 重叠核查", "train/dev 12 signer **100% 重叠**", "✅ 排除"),
    "p45c-error-attribution": ("误差归因", "见收据", "诊断"),
    "p45-vocab-bottleneck": ("词表贡献量化", "扩表收益上限 0.14 WER（当时估计）", "诊断"),
    "p46-unk-attribution": ("unk 来源归因",
                            "该吐 59.0% / 白吐 35.2%；模型自身错误 40.7%", "诊断"),
    "p47-controlled-decode": ("🔴 受控解码（词频先验）",
                              "WER 剔unk 0.7919 → 0.8171/0.8452/0.8710（单调变差）", "❌ 无效"),
    "p47b-why-prior-fails": ("解码失效机制",
                             "正确词 top-1 仅 **10.6%**、top-5 44.8%；blank 帧 87.4%",
                             "诊断"),
    "p47c-root-cause-test": ("8 条过拟合 + 词级线性探针",
                             "过拟合 WER 0.0000；词级探针 **macroAUC 0.7222**", "✅ 关键"),
    "p48-overfit-offline-vs-realtime": ("离线 vs 实时过拟合",
                                        "两种特征都能过拟合 0.0000（全新模型）", "✅"),
    "p48b-domain-shift": ("🔴 train-serve skew",
                          "train exact 96.7%(旧) vs 3.3%(新)；dev 仅差 4.7pp", "✅ 关键"),
    "p48-diagnosis": ("skew 诊断汇总", "提取器不同源（holistic vs tasks API）", "诊断"),
    "p50-wer-audit": ("🔴 WER 算法审计",
                      "穷举/手写/jiwer 三方一致 → 算法无误，问题在报告口径", "✅ 关键"),
    "p50b-jiwer-crosscheck": ("jiwer 交叉验证", "差 0.000000", "✅"),
    "p51c-feature-source": ("🟡 新旧特征判别力对比",
                            "macroAUC 0.7791(旧) vs 0.7744(新)，五块差值均<0.03",
                            "✅ 无显著差异"),
    "p51-decision": ("选型定论", "选离线 → 但P52 路线1失败，转路线 2", "决策"),
    "p52-legacy-repro": ("路线 1（旧 API 复现）",
                         "base 段 cos 0.996~0.9998✅；deltas 段 cos 0.32~0.96❌（占49%维度）",
                         "❌ 失败"),
    "p53-extract-tasksapi": ("路线 2 全量重提", "5488 条", "进行中"),
    "p53b-quality-check": ("新特征质量校验", "shape/NaN/presence 合格", "✅"),
    "p55_ce_csl_related_work": ("CE-CSL 官方基准整理",
                                "TFNet dev 42.1 / test 41.9", "✅"),
    "p56-gap-to-official": ("与官方差距分解",
                            "表面 −10.1pp → 口径 +27.1 → 词表地板 +30.4 → **真实能力差 6.7pp**",
                            "✅ 关键"),
    "p57-answers": ("Q1/Q2/Q3 三问", "词表来源、能否去unk、per-gloss 二分类 AUC 0.7082",
                    "✅"),
    "p58-vocab-origin": ("300 词表溯源",
                         "2026-06-17 a6a7100 引入；整句可表达率仅 8.9%", "✅"),
    "p59-what-to-do": ("词表/提取解耦", "扩词表不需重提特征；3516 可行（峰值 32MB）", "✅"),
    "p59b-feasibility": ("3516 词表可行性", "峰值 64.8MB；参数 3.07→3.89M；更快", "✅"),
    "p60-dataset-mixing": ("数据集混用论证", "四层不可混用（协议层最致命）", "✅"),
    "p61-modality": ("模态对照", "CE-CSL 只有 RGB；MediaPipe z 是相对深度非米制", "✅"),
    "p62b-correction": ("🔴 撤回 P57 结论",
                        "strip_variant_numbering 默认 True 已归一化；dev unk 实为 **0%**", "✅ 修正"),
    "p63b-preflight": ("训练前预检",
                       "**拦下 presence 语义错位**；3516 词表 CTC 约束通过(需T≤25<48)", "✅"),
    "p63c-presence-verify": ("presence 修复验证", "7 项全通过；新特征检出率反而更高", "✅"),
}


def main() -> None:
    rows = []
    for p in sorted(D.glob("*.json")):
        stem = p.stem
        note = NOTES.get(stem)
        if note is None:
            note = ("(未标注)", "见收据", "?")
        size = p.stat().st_size
        rows.append({"receipt": stem, "method": note[0],
                     "measured": note[1], "verdict": note[2],
                     "bytes": size})

    print("=" * 78)
    print("实验收据汇总：%d 份（标注 %d，未标注 %d）"
          % (len(rows), sum(1 for r in rows if r["method"] != "(未标注)"),
             sum(1 for r in rows if r["method"] == "(未标注)")))
    print("=" * 78)
    cnt = {}
    for r in rows:
        cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
    print("\n按结论分布：")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print("  %-12s %2d" % (k, v))

    print("\n" + "=" * 78)
    print("✅ 有效 / 成功")
    print("=" * 78)
    for r in rows:
        if any(k in r["verdict"] for k in ("✅", "有效", "成功")):
            print("  [%s] %-26s %s" % (r["receipt"][:26], r["method"][:26],
                                       r["measured"][:60]))

    print("\n" + "=" * 78)
    print("❌ 无效 / 失败")
    print("=" * 78)
    for r in rows:
        if any(k in r["verdict"] for k in ("❌", "无效", "作废", "失败")):
            print("  [%s] %-26s %s" % (r["receipt"][:26], r["method"][:26],
                                       r["measured"][:60]))

    print("\n" + "=" * 78)
    print("⏳ 待验 / 未落地（可从论文里挑）")
    print("=" * 78)
    for r in rows:
        if any(k in r["verdict"] for k in ("待验", "?")):
            print("  [%s] %-26s" % (r["receipt"][:26], r["method"][:30]))

    out = REPO / "artifacts/metrics/blank-gov/p64-receipts-summary.json"
    out.write_text(json.dumps({"rows": rows, "verdict_counts": cnt},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n收据 -> %s" % out)


if __name__ == "__main__":
    main()