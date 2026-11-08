#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P172-step2 启动前冒烟：验证教师缓存 meta/npy 与训练脚本查询接口一致。
不加载大模型，纯数据通路检查。
"""
import json
import numpy as np
from pathlib import Path

REPO = Path("/home/su127/FYP/domain-bounded-cslr")
CACHE = REPO / "artifacts/cache/p172_teacher"
ARR = CACHE / "teacher_logits"
META = CACHE / "teacher_meta.json"

meta = json.loads(META.read_text(encoding="utf-8"))
idx = meta["index"]
keys = list(idx.keys())
print("=== meta ===")
print("n_index =", len(keys))
print("first 3 keys =", keys[:3])
print("first value =", idx[keys[0]])

# 文件名格式核对
fns = sorted(ARR.glob("*.npy"))
print("=== npy files =", len(fns))
print("first 3 npy names =", [f.name for f in fns[:3]])

# 取一个 train key，确认 key 能直接作为文件名、并加载看 shape
sid = keys[0]
npf = ARR / ("%s.npy" % sid)
ok = npf.exists()
print("=== key->npy match: key=%s exists=%s" % (sid, ok))
if ok:
    arr = np.load(str(npf))
    print("npy shape =", arr.shape, "dtype =", arr.dtype)
    # 训练脚本解包 (_sk, T_raw, cT, _dl)
    _sk, T_raw, cT, _dl = idx[sid]
    print("meta T_raw=%s cT=%s dl=%s" % (T_raw, cT, _dl))
    print("npy len=%d  cT=%d  -> [:cT] ok=%s" % (arr.shape[0], cT, arr.shape[0] >= cT))

# 确认所有 index key 都有对应 npy
missing = [k for k in keys if not (ARR / ("%s.npy" % k)).exists()]
print("=== missing npy for index keys =", len(missing), missing[:5])
print("SMOKE_OK" if (ok and len(missing) == 0) else "SMOKE_FAIL")
