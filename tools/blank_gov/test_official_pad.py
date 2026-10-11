"""P123 契约单测：官方 collate 补帧语义（已知答案验证）。

判据全部来自 external/TFNet/DataProcessMoudle.py:435 collate_fn：
  left_pad = 6, total_stride = 4
  right_pad = ceil(max_len/4)*4 - max_len + left_pad
  max_len(最终) = ceil(max_len/4)*4 + 12
  videoLength_i = ceil(len_i/4)*4 + 2*left_pad

⚠️ 第一次写错了判据（把 vid.shape[0] 当时间维、把 batch 顺序当原顺序），
   修正后代码本身是对的。教训见 MEMORY 0.3：「体检脚本必须先用已知答案验证判据」。
"""
import sys
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
sys.path.insert(0, str(REPO / "tools" / "blank_gov"))

import torch  # noqa: E402
import p78_train_official_tfnet as P  # noqa: E402

ok = 0
bad = 0


def chk(name, cond, got=None):
    global ok, bad
    if cond:
        ok += 1
        print("  ✅ %s" % name)
    else:
        bad += 1
        print("  ❌ %s  got=%r" % (name, got))


def mk(n, val):
    """造一条 n 帧视频，第 i 帧全填 val+i（便于核对首尾复制）。"""
    v = torch.zeros(n, 3, 8, 8)
    for i in range(n):
        v[i] = val + i
    return {"video": v, "ids": [1, 2], "sid": "s%d" % n}


# 故意**乱序**输入，顺便验证「按帧数降序」排序
batch = [mk(33, 10.0), mk(48, 100.0), mk(50, 1.0)]
# 排序后顺序：50(val=1) → 48(val=100) → 33(val=10)

print("=== 官方口径（OFFICIAL_PAD=True）===")
P.OFFICIAL_PAD = True
vid, tgt, tgt_len, data_len, true_len, sids, ids = P.collate(batch)

Ls = [((n + 3) // 4) * 4 + 12 for n in (50, 48, 33)]   # [64, 60, 48]
T = max(Ls)                                            # 64
chk("vid 布局 [B,T,C,H,W]，时间维 = 64", vid.shape[1] == T, tuple(vid.shape))
chk("dataLen = [64,60,48]（= ceil(T/4)*4+12）",
    [int(x) for x in data_len[:, 0]] == [64, 60, 48],
    [int(x) for x in data_len[:, 0]])
chk("trueLen 仍是真实帧数 [50,48,33]",
    [int(x) for x in true_len] == [50, 48, 33], [int(x) for x in true_len])
chk("sid 顺序按帧数降序", sids == ["s50", "s48", "s33"], sids)

# --- 样本 0：50 帧，val=1，L=64 ---
v0 = vid[0]
chk("S0 左 6 帧 == 首帧(1.0)", float(v0[:6].mean()) == 1.0, float(v0[:6].mean()))
chk("S0 idx6 == 原第1帧(1.0)", float(v0[6].mean()) == 1.0, float(v0[6].mean()))
chk("S0 idx55 == 原第50帧(50.0)", float(v0[55].mean()) == 50.0,
    float(v0[55].mean()))
chk("S0 idx56~63 为末帧复制(50.0)",
    float(v0[56:].mean()) == 50.0, float(v0[56:].mean()))

# --- 样本 1：48 帧，val=100，L=60（但 batch 补到 64）---
v1 = vid[1]
chk("S1 左 6 帧 == 首帧(100.0)", float(v1[:6].mean()) == 100.0,
    float(v1[:6].mean()))
chk("S1 idx53 == 原第48帧(147.0)", float(v1[53].mean()) == 147.0,
    float(v1[53].mean()))
chk("S1 超出 dataLen(60) 的部分仍是末帧复制(147.0)",
    float(v1[60:].mean()) == 147.0, float(v1[60:].mean()))

# --- 样本 2：33 帧，val=10，L=48 ---
v2 = vid[2]
chk("S2 左 6 帧 == 首帧(10.0)", float(v2[:6].mean()) == 10.0,
    float(v2[:6].mean()))
chk("S2 idx38 == 原第33帧(42.0)", float(v2[38].mean()) == 42.0,
    float(v2[38].mean()))
chk("S2 超出 dataLen(48) 的部分仍是末帧复制(42.0)",
    float(v2[48:].mean()) == 42.0, float(v2[48:].mean()))

print("=== 旧口径（OFFICIAL_PAD=False，回归保护）===")
P.OFFICIAL_PAD = False
vid2, _, _, dl2, tl2, _, _ = P.collate(batch)
chk("旧口径 时间维 = max 真实帧数 50", vid2.shape[1] == 50, tuple(vid2.shape))
chk("旧口径 dataLen = [50,48,33]",
    [int(x) for x in dl2[:, 0]] == [50, 48, 33], [int(x) for x in dl2[:, 0]])
chk("旧口径 尾部补零（非末帧复制）",
    float(vid2[2, 33:].abs().sum()) == 0.0, float(vid2[2, 33:].abs().sum()))

print()
print("通过 %d / 失败 %d" % (ok, bad))
sys.exit(1 if bad else 0)
