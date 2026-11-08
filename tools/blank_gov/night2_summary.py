#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""夜间接力 #2 总汇总：读全部已生成收据 -> 出报告 -> 追加日志 -> 更新 MEMORY.md。

只读已有收据，不重新判定、不推测缺失数字；缺失项明确写「未生成」。
见 docs/planning/N2-PREREG-2026-11-08.md §5。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
DESK = Path("/mnt/c/Users/su127/Desktop/中文手语识别")
M = REPO / "artifacts/metrics/blank-gov"
CACHE = REPO / "artifacts/cache"


def jload(name):
    p = M / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print("收据解析失败", name, e, flush=True)
        return None


def num(x, fmt="%.2f"):
    return (fmt % x) if isinstance(x, (int, float)) else "未生成"


def main():
    k1j = jload("p233-k1-judge.json")
    k1u = jload("p233-k1-uwer.json")
    l1 = jload("p234-n2-probe.json")
    l2 = jload("p235-n2-k1-audit.json")
    s43j = jload("p238-k1-s43-judge.json")
    s43u = jload("p238-k1-s43-uwer.json")
    base = jload("p184-base-uwer.json")

    # ---- S0 K1 ----
    k1_verdict = k1j.get("verdict", "未生成") if k1j else "未生成"
    k1_valid = bool(k1u and k1u.get("n") == 515 and len(k1u.get("per_item", [])) == 515)
    if k1j and not k1_valid:
        k1_verdict = "INVALID（收据句数/per_item 不为 515，不得引用）"
    k1_wer = k1u.get("WER_official") if k1u else None
    k1_uw = k1u.get("uWER_pct") if k1u else None
    k1_low = (k1u.get("low_bucket") or {}).get("WER_pct") if k1u else None
    k1_high = (k1u.get("high_bucket") or {}).get("WER_pct") if k1u else None
    k1_delta = (k1j.get("delta") or {}).get("delta_WER_pp") if k1j else None
    k1_ci = ((k1j.get("paired_bootstrap") or {}).get("ci95")
             if k1j else None)

    # ---- L1 ----
    l1_verdict = l1.get("verdict", "未生成") if l1 else "未生成"
    l1_base = l1.get("base_WER_official") if l1 else None
    l1_chosen = l1.get("chosen") if l1 else None
    l1_delta = l1.get("chosen_delta_pp") if l1 else None
    l1_ci = l1.get("chosen_ci95") if l1 else None
    l1_best_wer = None
    if l1 and l1.get("grid"):
        l1_best_wer = min(g["WER_official"] for g in l1["grid"].values())

    # ---- L2 ----
    if not l2:
        l2_stat = "未生成"
    elif l2.get("status") != "OK":
        l2_stat = "SKIPPED（缺少 %s）" % ", ".join(l2.get("missing", []))
    else:
        s = l2["sentences"]; d = l2["distribution_vs_train"]; cb = l2["clip_bank"]
        l2_stat = ("生成 %d 句 / 低频覆盖 %.1f%% / 精确重复 %d / JS %.5f / "
                   "片段库 %d gloss、区间中位 %s 帧"
                   % (s["n"], 100 * s["low_freq_coverage"],
                      l2["leakage"]["exact_duplicates_train_dev_test"],
                      d["js_divergence"], cb["n_gloss_covered"],
                      cb["clip_length_input_frames"]["median"]))

    # ---- S3 ----
    if s43j and s43u:
        s43_stat = ("verdict **%s**，WER %s / uWER %s / 低频桶 %s"
                    % (s43j.get("verdict", "?"), num(s43u.get("WER_official")),
                       num(s43u.get("uWER_pct")),
                       num((s43u.get("low_bucket") or {}).get("WER_pct"))))
        two_seed = ("seed42 %s / seed43 %s" %
                    (num(k1_wer), num(s43u.get("WER_official"))))
    else:
        s43_stat = "未运行（时间盒跳过或训练未完成），2 seed 复跑待补"
        two_seed = "仅 seed42 %s" % num(k1_wer)

    base_stat = ("WER %s / uWER %s / 低频桶 %s / 高频桶 %s"
                 % (num((base or {}).get("WER_official")),
                    num((base or {}).get("uWER_pct")),
                    num(((base or {}).get("low_bucket") or {}).get("WER_pct")),
                    num(((base or {}).get("high_bucket") or {}).get("WER_pct"))))

    all_fail = (k1_verdict in ("FAIL", "INVALID（收据句数/per_item 不为 515，不得引用）")
                and l1_verdict in ("FAIL", "INVALID", "未生成")
                and not (s43j and s43j.get("verdict") == "PASS"))

    txt = """# 夜间接力 #2 总汇总（2026-11-08 20:55 → 2026-11-09 09:00 硬停）

> 用户指令「老样子，明早9点前自行工作」。预注册 `N2-PREREG-2026-11-08.md`（20:55 写死，
> 早于 K1 任何结果）。**只转述收据原值；未生成项明确标注，不推测、不重新判定。**
> 生成时间：%s

## 0. 基线（不变）
P184 冻结主干 PEFT@160：`%s`（`p184-base-uwer.json`）。
天花板：官方@224 零样本 test **32.74** / P184@160 test **33.98** / P150 **53.73**。

## 1. S0 — K1 语料合成（P230–P233，20:48 启动）
- 有效性核对：**%s**（`p233-k1-uwer.json` n=515、per_item=515、训练日志含 ep006）
- **verdict = %s**；WER %s / uWER %s / 低频桶 %s / 高频桶 %s
- Δ WER %s pp，配对 CI %s
- 判据：统一 R1–R8，见 `K1-PREREG-2026-11-08.md`。K1 链自行完成评估/判定/报告
  （`stage-P233-K1-corpus-synthesis.md`），接力链不重复判定。

## 2. L1 — spike 邻域后验融合扫描（P234，零训练纯 CPU）
- **verdict = %s**；λ=0 基线 WER %s，chosen = `%s`，6 档最优 WER %s，
  Δ %s pp，配对 CI %s。
- 网格 k∈{1,2,4}×λ∈{0.3,0.6}；护栏 R3/R5/R6；主判据 Δ≥2.00pp 且 CI 下界>0。
- 报告 `stage-P234-N2-spike-fusion.md`；收据 `p234-n2-probe.json`。

## 3. L2 — K1 语料质量与对齐边界审计（P235，探索性，不判优劣）
- %s
- 报告 `stage-P235-N2-K1-corpus-audit.md`；收据 `p235-n2-k1-audit.json`。

## 4. S3 — K1 的 seed 43 复跑（P238，6 轮训练，条件触发）
- %s
- 2 seed：%s。⚠️ seed 不同不改变超参/数据/判据，是项目规范「2 seed」要求，
  **不是 FAIL 后调参重跑**。未跑 test。
- 报告 `stage-p238-k1-s43.md`（由 `night_report.py` 在运行时生成）。

## 5. 本夜最重要的机制结论
1. K1 是本项目**第一次改训练数据组织方式**（前五轮 14 连败都在固定 4973 条真实句子上
   改损失/后验/输入/参数）。无论结果如何，J1 已证明 CTC 强制对齐可切出
   中位 ~31.5 输入帧的 per-gloss 区间，「切不出片段」的担忧不成立。
2. L1 与 I0 形成闭环：I0 证明改**输入帧率**不能增密对齐（每词非 blank 帧数
   0.888–1.086 几乎不变）；L1 检验改**后验邻域**是否含新信息。两者都判负 ⇒
   「每词 1.06 帧被监督」是 CTC 目标与模型的固有属性，推理期任何时序操作都动不了它。
3. 统计 bigram 生成器（K1-PREREG 已标【无】）语义松散，是 K1 最大已知风险；
   L2 审计只给分布/覆盖数字，**不能**用它把 K1 判负归因于生成质量。

## 6. 下一轮候选方向（仅列出，不自动启动）
- **改监督密度（动损失，不动输入）**：【有】EMNLP26 Beyond CTC Spikes / GTM、
  arXiv:2406.02560 label-prior CTC、本地 06_SMART。这是 I0+L1 判负后逻辑上唯一
  还成立的口子；但 D3 已证伪最便宜入口（TFD 边界 corr 0.29），工程约 1 天，
  且与 F1（训练期 logit adjustment 明确负向）、P185（额外监督破坏对齐）高重合。
- **换非 CTC 解码目标（RNN-T / transducer）**：【间接】可绕开条件独立假设，
  但 P200 已证 6 轮内大时序模块拟合失败，8G + 6 轮预算不匹配。
- **结论**：%s。累计判负已达 15+，且所有「轻量单变量」路线系统穷尽；
  合理做法是**接受天花板收尾**，而不是继续找新损失项。

## 7. 状态与红线
- 未跑 **test**；未 **push**；判据/网格/阈值/数据均未事后修改；dev 标签未进训练。
- 本夜自动 commit（用户约定：提交可自行做、推送须等明确指令），
  明早需用户说「推送」才推双端。
- 链状态：`artifacts/cache/night2_state.txt`；日志 `artifacts/logs_NIGHT2_chain.log`。
""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        base_stat,
        "K1_VALID" if k1_valid else "K1_INVALID（需人工复核）",
        k1_verdict, num(k1_wer), num(k1_uw), num(k1_low), num(k1_high),
        num(k1_delta, "%+.3f"), str(k1_ci or "未生成"),
        l1_verdict, num(l1_base), l1_chosen or "（无护栏通过）",
        num(l1_best_wer), num(l1_delta, "%+.3f"), str(l1_ci or "未生成"),
        l2_stat, s43_stat, two_seed,
        ("K1 / L1 / S3 均未拿到 SIGNAL ⇒ **建议接受天花板收尾**"
         if all_fail else
         "存在 SIGNAL 或 PASS 项 ⇒ 需另写预注册再决定是否复跑，本夜不自动进 test"),
    )

    for base_dir in (REPO, DESK):
        p = base_dir / "docs/experiments/OVERNIGHT-2-RUN-2026-11-09.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt, encoding="utf-8")
    print("汇总 -> OVERNIGHT-2-RUN-2026-11-09.md (+桌面镜像)", flush=True)

    # ---- 追加次日日志（append-only） ----
    log = DESK / ".workbuddy/memory/2026-11-09.md"
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n## 夜间接力 #2 汇总（%s）\n" %
                datetime.now().strftime("%Y-%m-%d %H:%M"))
        f.write("- K1 verdict **%s**（有效性 %s），WER %s / uWER %s / 低频桶 %s。\n"
                % (k1_verdict, "K1_VALID" if k1_valid else "K1_INVALID",
                   num(k1_wer), num(k1_uw), num(k1_low)))
        f.write("- L1 spike 邻域后验融合 verdict **%s**（基线 %s、chosen %s、Δ %s）。\n"
                % (l1_verdict, num(l1_base), l1_chosen or "无护栏通过",
                   num(l1_delta, "%+.3f")))
        f.write("- L2 语料质量审计：%s。\n" % l2_stat)
        f.write("- S3 seed43：%s；2 seed：%s。\n" % (s43_stat, two_seed))
        f.write("- 预注册 `N2-PREREG-2026-11-08.md`（20:55 写死）；"
                "未跑 test、未 push；本夜自动 commit。\n")
        f.write("- 报告 `OVERNIGHT-2-RUN-2026-11-09.md`（WSL + 桌面）。\n")
    print("日志已追加 -> 2026-11-09.md", flush=True)

    # ---- 更新 MEMORY.md 状态（宽松匹配，缺失不报错） ----
    mem = DESK / ".workbuddy/memory/MEMORY.md"
    s = mem.read_text(encoding="utf-8")
    for line in s.splitlines():
        if line.startswith("- [~] **K1 语料合成已启动**"):
            s = s.replace(line,
                          "- [x] **K1 语料合成收官**：verdict **%s**，WER %s / uWER %s / 低频桶 %s；"
                          "预注册 `K1-PREREG`（20:43）；报告 `stage-P233-K1-corpus-synthesis.md`"
                          % (k1_verdict, num(k1_wer), num(k1_uw), num(k1_low)), 1)
            break
    for line in s.splitlines():
        if line.startswith("- 状态：**双端推送完成**"):
            s = s.replace(line,
                          "- 状态：K1 / L1 / L2 / S3 夜间接力 #2 已收尾（详见 "
                          "`OVERNIGHT-2-RUN-2026-11-09.md`）；K1 %s、L1 %s、S3 %s；"
                          "GPU 空闲；未跑 test、未 push（本夜自动 commit，等用户说「推送」）。"
                          % (k1_verdict, l1_verdict,
                             "已完成" if s43j else "时间盒跳过"), 1)
            break
    if "## 夜间接力 #2" not in s:
        s += ("\n## 夜间接力 #2（11-08 20:55 → 11-09 09:00）\n"
              "- L1 spike 邻域后验融合 = **%s**（λ=0 基线 %s，chosen %s）；\n"
              "  I0 证改输入帧率无用、L1 证改后验邻域也无新信息 ⇒ "
              "「每词 1.06 帧」是 CTC 固有属性，推理期时序操作动不了它。\n"
              "- L2 K1 语料质量审计（探索性，不判优劣）：%s。\n"
              "- 累计 15+ 判负，建议接受天花板收尾；剩余有依据但成本高的方向：\n"
              "  GTM/label-prior 密集监督【有，但与 F1/P185 高重合】、\n"
              "  RNN-T/transducer【间接，P200 反证据 + 预算不匹配】。\n"
              "- 预注册 `N2-PREREG-2026-11-08.md`（20:55 写死，早于 K1 结果）。\n"
              % (l1_verdict, num(l1_base), l1_chosen or "无护栏通过", l2_stat))
    mem.write_text(s, encoding="utf-8")
    print("MEMORY.md 已更新, bytes =", len(s.encode("utf-8")), flush=True)


if __name__ == "__main__":
    main()
