"""P69：从 CE-CSL 官方论文原文提取评估协议与 WER 定义

用户要求：「之后模型效果的判断按照官方的来，计算方法也是」
⇒ 必须确认官方到底怎么算 WER、有没有做归一化/去重复等处理。

做法：用 pdfplumber / pypdf 抽正文，定位 Evaluation / Metric / WER 段落。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PDF = Path("/mnt/c/Users/su127/Desktop/中文手语识别/_ce-csl-paper.pdf")
OUT = Path("/home/su127/FYP/domain-bounded-cslr/artifacts/metrics/blank-gov"
           "/p69-official-protocol.json")


def get_text() -> str:
    try:
        import pypdf
        r = pypdf.PdfReader(str(PDF))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(str(PDF)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as e:
        print("解析失败: %s" % e)
        return ""


txt = get_text()
print("=" * 74)
print("抽取字符数: %d" % len(txt))
if not txt:
    sys.exit(1)

flat = re.sub(r"[ \t]+", " ", txt)

KEYS = [
    ("metric_section", r"(?:4\s*\.?\s*)?(?:Experiments?)[^\n]*"),
]

print("\n" + "=" * 74)
print("1. 定位「evaluation metric / WER」相关段落")
print("=" * 74)
# 找所有提到 WER / word error rate 的上下文
for m in re.finditer(r"(?i)(word error rate|WER|evaluation metric|"
                     r"Levenshtein|edit distance)", flat):
    s = max(0, m.start() - 320)
    e = min(len(flat), m.end() + 420)
    seg = flat[s:e].replace("\n", " ")
    print("\n--- 命中 %r @%d ---" % (m.group(0), m.start()))
    print("  " + seg)

print("\n" + "=" * 74)
print("2. Evaluation protocol / implementation details 段落")
print("=" * 74)
for kw in [r"(?i)evaluation (?:protocol|setup|criterion)",
           r"(?i)implementation detail",
           r"(?i)we (?:report|measure|compute|evaluate)",
           r"(?i)data (?:pre|post)[- ]?process",
           r"(?i)official .{0,20}split",
           r"(?i)dev(?:elopment)? set",
           r"(?i)beam search"]:
    for m in re.finditer(kw, flat):
        s = max(0, m.start() - 260)
        e = min(len(flat), m.end() + 460)
        print("\n--- %s @%d ---" % (m.group(0), m.start()))
        print("  " + flat[s:e].replace("\n", " "))

print("\n" + "=" * 74)
print("3. 数据集统计与 OOV 说明（确认词表口径）")
print("=" * 74)
for kw in [r"(?i)vocabular", r"(?i)\bOOV\b", r"(?i)gloss (?:unit|sequence|annotation)",
           r"(?i)annotation (?:rule|variation|suffix)"]:
    seen = set()
    for m in re.finditer(kw, flat):
        s = max(0, m.start() - 220)
        e = min(len(flat), m.end() + 380)
        seg = flat[s:e].replace("\n", " ")
        key = seg[:80]
        if key in seen:
            continue
        seen.add(key)
        print("\n--- %s @%d ---" % (m.group(0), m.start()))
        print("  " + seg)
        if len(seen) >= 4:
            break

print("\n" + "=" * 74)
print("4. 与其它数据集对比的表格数字（确认我们引用无误）")
print("=" * 74)
for kw in [r"(?i)TFNet", r"(?i)MAM-FSD", r"(?i)CSL-Daily"]:
    seen = set()
    for m in re.finditer(kw, flat):
        s = max(0, m.start() - 200)
        e = min(len(flat), m.end() + 400)
        seg = flat[s:e].replace("\n", " ")
        key = seg[:60]
        if key in seen:
            continue
        seen.add(key)
        print("\n--- %s ---" % m.group(0))
        print("  " + seg)
        if len(seen) >= 3:
            break

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(
    {"pdf": str(PDF), "chars": len(txt),
     "note": "人工阅读上方输出后填写结论"}, ensure_ascii=False, indent=2),
    encoding="utf-8")
print("\n收据 -> %s" % OUT)