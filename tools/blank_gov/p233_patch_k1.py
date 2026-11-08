#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P233/K1：给 p78 加默认关闭的 --k1-manifest（最小侵入，不改任何已有行为）。

两处插入：
  1) 参数区加开关
  2) labels_tr 构造后、RGBSeqDataset 构造前注入合成样本（词表仍由 CSV_DIR 原始 csv 构建）
"""
from pathlib import Path

P = Path("/home/su127/FYP/domain-bounded-cslr/tools/blank_gov/p78_train_official_tfnet.py")
s = P.read_text(encoding="utf-8")
n = 0

a1 = '''    ap.add_argument("--csgc-nneg", type=int, default=16,
                    help="P212/E1：hard negative 数量")'''
b1 = a1 + '''
    ap.add_argument("--k1-manifest", default="",
                    help="P233/K1：合成训练样本 manifest json；空=严格保持原行为")'''
if "--k1-manifest" not in s:
    assert a1 in s, "参数区锚点未找到"
    s = s.replace(a1, b1, 1)
    n += 1

a2 = '''    labels_tr = read_csv_split("train.csv")
    labels_dv = read_csv_split("dev.csv")'''
b2 = '''    labels_tr = read_csv_split("train.csv")
    # P233/K1：合成训练样本只注入内存 labels；
    # 词表仍由 CSV_DIR/*.csv 构建（Word2Id 不读 labels_tr），故 strict load 不受影响。
    if a.k1_manifest:
        _km = __import__("json").loads(
            open(a.k1_manifest, encoding="utf-8").read())
        _add = {str(k): (str(v[0]), str(v[1]))
                for k, v in _km["manifest"].items()}
        labels_tr.update(_add)
        print("K1 合成样本注入 %d 条，train 总数 %d"
              % (len(_add), len(labels_tr)), flush=True)
    labels_dv = read_csv_split("dev.csv")'''
if "K1 合成样本注入" not in s:
    assert a2 in s, "labels_tr 锚点未找到"
    s = s.replace(a2, b2, 1)
    n += 1

P.write_text(s, encoding="utf-8")
print("K1 patch applied edits =", n)
