# -*- coding: utf-8 -*-
"""
整理脚本:按形态分类整理 data/train 全部小区(英文文件夹),去重 incoming 副本。

结构:
  ALL/
    01_deep_fall/      深跌未反弹(总跌超25%且未收复)
    02_v_rebound/      V型反弹(回撤超15%且收复大半)
    03_flat_decline/   横盘阴跌(-25%~-5%)
    04_rising/         逆势上涨(>+5%)
    05_neighbor_hudai/ 邻居板块独有小区(胡埭/马山/太湖)
    06_regular/        其余
  incoming/ 中被 ALL 重复的副本直接删除;解析失败/历史<12月的移到 unused_备份。
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from load_data import load_complex

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ALL = os.path.join(ROOT, "train")
NEW = os.path.join(ROOT, "incoming")
BAK = os.path.join(ROOT, "_unused_20260815")

CATS = {
    "分类_深跌未反弹": "01_deep_fall",
    "分类_V型反弹": "02_v_rebound",
    "分类_横盘阴跌": "03_flat_decline",
    "分类_逆势上涨": "04_rising",
    "分类_邻居板块_胡埭马山太湖": "05_neighbor_hudai",
}
CAT_PRIORITY = ["分类_深跌未反弹", "分类_V型反弹", "分类_横盘阴跌", "分类_逆势上涨",
                "分类_邻居板块_胡埭马山太湖"]  # 形态分类优先于邻居板块
os.makedirs(BAK, exist_ok=True)
for d in list(CATS.values()) + ["06_regular"]:
    os.makedirs(os.path.join(ALL, d), exist_ok=True)


def shape_of(p):
    c = load_complex(p)
    s = c["old"][["date", "price"]].dropna()
    px = s["price"].values
    if len(px) < 2:
        return None
    chg = px[-1] / px[0] - 1
    peak = float(px.max())
    peak_i = int(px.argmax())
    mdd = float((px[peak_i:] - peak).min() / peak) if peak_i < len(px) - 1 else 0.0
    recovery = float(px[-1] / peak - 1)
    return {"n": len(px), "chg": chg, "mdd": mdd, "recovery": recovery}


def auto_bucket(sh):
    """形态自动分桶(用于 ALL 中未标注的小区)。"""
    if sh["mdd"] <= -0.15 and sh["recovery"] > -0.05:
        return "02_v_rebound"
    if sh["chg"] < -0.25 and sh["recovery"] < -0.10:
        return "01_deep_fall"
    if sh["chg"] < -0.05:
        return "03_flat_decline"
    if sh["chg"] > 0.05:
        return "04_rising"
    return "06_regular"


report = {"moved_new": [], "dup_deleted": [], "all_to": {}, "backup": [], "failed": []}

# ---------- 1) 新增文件夹:先去重,按用户分类移入 ALL ----------
new_file_cat = {}   # name -> (primary_cat, path)
for cat in CAT_PRIORITY:
    d = os.path.join(NEW, cat)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        if f.lower().endswith((".xlsx", ".xlsm")) and f not in new_file_cat:
            new_file_cat[f] = (cat, os.path.join(d, f))

all_names = set(os.listdir(ALL)) | set(os.listdir(os.path.join(ALL, "06_regular")))
for f, (cat, src) in new_file_cat.items():
    if f in all_names:          # 与 ALL 内容相同(已验证)→ 删除副本
        os.remove(src)
        report["dup_deleted"].append(f)
        continue
    sh = shape_of(src)
    if sh is None or sh["n"] < 12:   # 无数据/太短 → 备份
        shutil.move(src, os.path.join(BAK, f))
        report["backup"].append((f, "新增-" + cat))
        continue
    dst_dir = os.path.join(ALL, CATS[cat])
    shutil.move(src, os.path.join(dst_dir, f))
    report["moved_new"].append((f, CATS[cat]))

# ---------- 2) ALL 现有文件:自动分桶(跳过子目录) ----------
for f in sorted(os.listdir(ALL)):
    p = os.path.join(ALL, f)
    if os.path.isdir(p) or not f.lower().endswith((".xlsx", ".xlsm")):
        continue
    try:
        sh = shape_of(p)
    except Exception as e:
        shutil.move(p, os.path.join(BAK, f))
        report["failed"].append((f, str(e)[:40]))
        continue
    if sh is None or sh["n"] < 12:
        shutil.move(p, os.path.join(BAK, f))
        report["backup"].append((f, "ALL-数据不足"))
        continue
    b = auto_bucket(sh)
    # 全部归入子目录(regular 也单独放,保持 ALL 下只有分类子目录)
    shutil.move(p, os.path.join(ALL, b, f))
    report["all_to"][f] = b

# ---------- 3) 删除空的新增文件夹 ----------
for cat in list(os.listdir(NEW)):
    d = os.path.join(NEW, cat)
    try:
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    except OSError:
        pass
try:
    if os.path.isdir(NEW) and not os.listdir(NEW):
        os.rmdir(NEW)
except OSError:
    pass

# ---------- 报告 ----------
print(f"新增移入 ALL: {len(report['moved_new'])} 个")
print(f"删除重复副本: {len(report['dup_deleted'])} 个")
print(f"备份(无数据/太短): {len(report['backup'])} 个")
print(f"备份(解析失败): {len(report['failed'])} 个")
from collections import Counter
print(f"\nALL 最终分布:")
for k, v in sorted(Counter(report['all_to']).items()):
    print(f"  {k}: {v} 个")
print(f"备份目录: {BAK}")
