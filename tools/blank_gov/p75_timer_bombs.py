"""P75：第二层检修 —— 找脚本查不出的"定时炸弹"

P74 的 8 个维度体检全绿。但体检只能查「已知的契约」，
查不出「**将来会坏**」的地方。这一层专找定时炸弹：

  E1 **硬编码词表**：重训换词表后，服务端会静默用错
  E2 **硬编码特征根**：路径写死，换特征目录就崩
  E3 **硬编码 checkpoint 路径**：换了 ckpt 名就找不到
  E4 **相对/绝对路径混用**：Windows 副本与 WSL 路径差异
  E5 **训练/推理特征定义漂移**：两处presence 定义是否同时更新
  E6 **未使用的旧代码**：可能仍被import 造成混淆
  E7 **模型文件缺失**：推理依赖的 .task 是否都在
  E8 **变量名与语义不符**：如 epoch 属性名等
"""
from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
WIN = Path("/mnt/c/Users/su127/Desktop/中文手语识别/blank_governance/deploy")

bomb = []


def note(sev, title, detail, action):
    bomb.append({"severity": sev, "title": title, "detail": detail,
                 "action": action})
    print("  [%s] %s" % (sev, title))
    print("       %s" % detail)
    print("       → %s" % action)


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO),
                       timeout=120)
    return (r.stdout or "") + (r.stderr or "")


svc = (REPO / "app/backend/ctc_landmark_service.py").read_text(encoding="utf-8")
rt = (REPO / "app/backend/realtime_landmark.py").read_text(encoding="utf-8")

print("=" * 78)
print("E1  硬编码词表（重训后必炸）")
print("=" * 78)
# 判据要看**实际传给 build_ordered_vocabulary 的 max_tokens 表达式**，
# 而不是文件里有没有出现过 "300" 字面量（注释里也会出现）。
m_call = re.search(r"self\.voc,\s*_\s*=\s*build_ordered_vocabulary\(([^)]*)\)",
                   svc, re.S)
hardcoded = False
if m_call:
    arg = m_call.group(1)
    print("  传给 build_ordered_vocabulary 的参数:")
    print("    %s" % re.sub(r"\s+", " ", arg).strip())
    # 如果直接出现字面量 300（或依赖默认 vocab），就是硬编码
    if re.search(r"max_tokens\s*=\s*\d", arg):
        hardcoded = True
    # 若用变量，则追变量的来源
    else:
        mv = re.search(r"max_tokens\s*=\s*(\w+)", arg)
        if mv:
            var = mv.group(1)
            srcs = re.findall(r"%s\s*=\s*(.+)" % re.escape(var), svc)
            print("    max_tokens 变量 %s 来源: %s" % (var, srcs[:3]))
            if any("300" in s and "getenv" not in s for s in srcs):
                hardcoded = True

ck = list((REPO / "artifacts/checkpoints").glob("p42*.pt"))
if ck:
    import torch
    b = torch.load(str(ck[-1]), map_location="cpu", weights_only=False)
    print("  当前 P42 ckpt vocab_size = %s" % b.get("vocab_size"))
    print("  ckpt 是否带 vocab_params = %s"
          % ("vocab_params" in b))

reads_ckpt = "vocab_params" in svc
env_backstop = "CSLR_VOCAB_MAX_TOKENS" in svc
print("\n  服务端读 ckpt.vocab_params = %s" % reads_ckpt)
print("  服务端有环境变量兜底      = %s" % env_backstop)
if hardcoded and not reads_ckpt:
    note("🔴", "E1 服务端词表硬编码",
         "重训换词表后服务端会用错词表",
         "改为从 checkpoint 读 vocab_params")
elif reads_ckpt:
    print("  ✅ 已改为优先从 checkpoint 读词表参数（2026-10-06 修复）")
elif not hardcoded:
    print("  ✅ 未发现硬编码")

print("\n" + "=" * 78)
print("E2  硬编码特征根")
print("=" * 78)
paths = re.findall(r'["\'](artifacts/[^"\']+)["\']', svc + rt)
print("  代码里的特征路径: %s" % (set(paths) or "无"))
envs = re.findall(r"os\.environ\.get\(\s*[\'\"](CSLR_\w+)", svc)
print("  服务端支持的环境变量: %s" % set(envs))

print("\n" + "=" * 78)
print("E3  硬编码 checkpoint 路径")
print("=" * 78)
m = re.search(r'os\.getenv\(\s*["\']CSLR_CTC_LANDMARK_CKPT["\']', svc)
print("  ckpt 从环境变量读取 = %s" % bool(m))
if not m:
    note("🟡", "E3 checkpoint 路径未参数化",
         "找不到 CSLR_CTC_LANDMARK_CKPT 读取",
         "改成环境变量 + 默认值")
else:
    print("  ✅ 可通过 CSLR_CTC_LANDMARK_CKPT 指定")

print("\n" + "=" * 78)
print("E4  Windows 副本 vs WSL 的路径差异")
print("=" * 78)
for f in ("realtime_landmark.py", "ctc_landmark_service.py"):
    p = WIN / f
    if not p.exists():
        continue
    t = p.read_text(encoding="utf-8")
    # ⚠️ 必须只查**可执行代码**里的路径，不能把注释/docstring 算进去。
    # ⚠️ 两个坑：
    #   1. ast 会把 docstring 也标成 Constant
    #   2. **不能用 id() 比较字符串** —— 临时对象被GC 后 id 会被复用，
    #      导致 docstring 被误判为代码（P75 首次实现踩过）。
    # 正确做法：直接持有字符串对象本身，用 `is` 比较。
    hard_code = []      # (lineno, snippet, value_object)
    docs = []           # docstring 的值对象
    tree = None
    try:
        tree = ast.parse(t)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef, ast.Module)):
                b = getattr(node, "body", None)
                if b and isinstance(b[0], ast.Expr) \
                        and isinstance(b[0].value, ast.Constant) \
                        and isinstance(b[0].value.value, str):
                    docs.append(b[0].value.value)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "/home/su127" in node.value:
                    hard_code.append((node.lineno, node.value[:60],
                                      node.value))
    except SyntaxError as exc:
        print("  %s 解析失败: %s" % (f, exc))
    n_doc = sum(1 for _, _, v in hard_code
                if any(v is d for d in docs))
    code_hits = [(ln, v) for ln, v, val in hard_code
                 if not any(val is d for d in docs)]
    if code_hits:
        print("  %s 可执行代码含 %d 处 WSL 绝对路径" % (f, len(code_hits)))
        note("🟡", "E4 Windows 副本可执行代码含 WSL 路径: %s" % f,
             "行%s 如 %s" % ([ln for ln, _ in code_hits], code_hits[0][1][:50]),
             "改用 REPO 标记文件探测")
    else:
        print("  %s 可执行代码无 WSL 路径 ✓（docstring 提及 %d 处，不影响）"
              % (f, n_doc))

print("\n" + "=" * 78)
print("E5  presence 定义是否两处同步")
print("=" * 78)
for name, path in (("WSL app/backend", REPO / "app/backend/realtime_landmark.py"),
                   ("WSL tools/blank_gov", REPO / "tools/blank_gov/realtime_landmark.py"),
                   ("Windows 副本", WIN / "realtime_landmark.py")):
    if not path.exists():
        print("  %-20s 缺失" % name)
        continue
    t = path.read_text(encoding="utf-8")
    mm = re.search(r"presence\s*=\s*np\.array\(\[(.*?)\]", t, re.S)
    order = re.findall(r"1\.0\s+if\s+(\w+)", mm.group(1)) if mm else []
    ok = order == ["have_l", "have_r", "p", "have_f"]
    print("  %-20s %s %s" % (name, order, "✓" if ok else "✗"))
    if not ok:
        note("🔴", "E5 presence 顺序不一致: %s" % name,
             "实际 %s" % order, "改为 [have_l, have_r, p, have_f]")

print("\n" + "=" * 78)
print("E6  服务端是否有死代码残留")
print("=" * 78)
# 判据：self.nrm 是否恒为 None；以及未使用的 import 是否有**显式标注**
# （`# noqa: F401` + 说明性注释 = 有意保留，不是残留）
if "FeatureNormalizer" in svc:
    cnt_import = svc.count(
        "from cslr.recognition.dataset import FeatureNormalizer")
    cnt_use = len(re.findall(r"self\.nrm\s*=\s*(?!None)\S", svc))
    annotated = ("noqa: F401" in svc
                 and "FeatureNormalizer" in svc
                 and "刻意不再使用" in svc)
    print("  FeatureNormalizer import %d 处；self.nrm 非 None 赋值 %d 处"
          % (cnt_import, cnt_use))
    print("  有显式禁用标注（noqa + 说明）= %s" % annotated)
    if cnt_import and cnt_use == 0 and not annotated:
        note("🟡", "E6 服务端留有未标注的死代码",
             "import FeatureNormalizer 与 `if self.nrm is not None` "
             "都是死分支（self.nrm 恒 None），且无说明",
             "删掉，或加 `# noqa: F401` + 说明为何保留")
    elif cnt_import and cnt_use == 0 and annotated:
        print("  ✅ 死代码已显式标注为「刻意保留」，非隐患")

print("\n" + "=" * 78)
print("E7  推理依赖的模型文件是否齐全")
print("=" * 78)
models = REPO / "models"
need = ["pose_landmarker_lite.task", "hand_landmarker.task",
        "face_landmarker.task"]
for f in need:
    p = models / f
    print("  %-32s %s" % (f, ("%.1f MB" % (p.stat().st_size / 1e6))
                          if p.exists() else "❌ 缺失"))
    if not p.exists():
        note("🔴", "E7 模型文件缺失 %s" % f, "推理会直接失败", "补齐文件")

print("\n" + "=" * 78)
print("E8  服务端属性名与实际是否一致")
print("=" * 78)
blob = None
if ck:
    import torch
    blob = torch.load(str(ck[-1]), map_location="cpu", weights_only=False)
    keys = sorted(blob.keys())
    print("  ckpt keys = %s" % keys)
    for attr in ("epoch", "dev_wer", "vocab_size"):
        # 服务端可能写 .get(k) 或 .get(k, fallback) 或 .get(k1, .get(k2))，
        # 判据只看「有没有读」，由下面的 fallback 解析判断是否真会拿到 None
        used = re.search(r'blob\.get\(\s*["\']%s["\']' % attr, svc)
        print("    blob.get('%s') 使用 = %s" % (attr, bool(used)))
        if not used:
            continue
        if attr in keys:
            continue
        # 解析该get 调用里是否有 fallback
        call = re.search(r'blob\.get\(\s*["\']%s["\']([^)]*)\)' % attr, svc)
        rest = (call.group(1) if call else "").strip()
        has_fb = rest.startswith(",") and "get(" in rest
        if has_fb:
            print("       ✅ 有 fallback（%s）" % rest.strip()[:40])
        else:
            note("🟡", "E8 ckpt 无 '%s' 键且代码无 fallback" % attr,
                 "ckpt keys=%s -> 返回 None，/health 该字段变 null" % keys,
                 "加 fallback：.get('%s', blob.get('其它键'))"
                 % attr)

# ==================================================================
print("\n" + "=" * 78)
print("定时炸弹汇总")
print("=" * 78)
if not bomb:
    print("  ✅ 未发现定时炸弹")
else:
    for b in bomb:
        print("  [%s] %s" % (b["severity"], b["title"]))
        print("       → %s" % b["action"])

p = REPO / "artifacts/metrics/blank-gov/p75-timer-bombs.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({"bombs": bomb}, ensure_ascii=False, indent=2),
             encoding="utf-8")
print("\n收据 -> %s" % p)