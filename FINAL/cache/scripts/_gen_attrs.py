# -*- coding: utf-8 -*-
"""
生成小区属性表 data/complex_attrs.xlsx:
  - 列出全部训练集(398)+ 测试集(2)小区名;
  - 启发式初稿:城区/楼龄/容积率/绿化率从名字推断(如含"别墅"→低容积率高绿化,
    含"新村"→老小区);学区/地铁/商业/公园填空(NaN)待 AI 或人工补充;
  - 用户可以修改这个表,训练时自动读取。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from train_all import load_all_complexes
from load_data import load_complex, WORKBOOK_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "complex_attrs.xlsx")

DISTRICTS = [
    ("宜兴", "宜兴市"), ("江阴", "江阴市"),
    ("滨湖|蠡湖|马山|胡埭|太湖|山水城", "滨湖区"),
    ("锡山|鹅湖", "锡山区"),
    ("惠山|阳山|洛社|前洲|堰桥|玉祁", "惠山区"),
    ("梁溪|崇安|南长|北塘|扬名|广益", "梁溪区"),
    ("新吴|梅村|硕放|鸿山|江溪", "新吴区"),
]


def guess_district(name):
    for pat, d in DISTRICTS:
        if re.search(pat, name):
            return d
    return "待补充"


def guess_age(name):
    """楼龄初稿(年,2026 基准):新村→30, 花园→20, 别墅/院/府→12, 其他→15。"""
    if "新村" in name:
        return 30
    if "花园" in name:
        return 20
    if "别墅" in name or "院" in name or "府" in name:
        return 12
    return 15


def guess_density(name):
    """容积率初稿:别墅→0.6, 商住/公寓→3.0, 其他→2.0。"""
    if "别墅" in name:
        return 0.6
    if "商住" in name or "公寓" in name or "SOHO" in name or "公馆" in name:
        return 3.0
    return 2.0


def guess_green(name):
    """绿化率初稿(%):别墅→45, 其他→35。"""
    return 45 if "别墅" in name else 35


def main():
    complexes = load_all_complexes()
    names = [c["name"] for c in complexes]
    for f in sorted(os.listdir(WORKBOOK_DIR)):
        if f.lower().endswith((".xlsx", ".xlsm")) and "宏观" not in f:
            c = load_complex(os.path.join(WORKBOOK_DIR, f))
            if c["name"] not in names:
                names.append(c["name"])
    names = sorted(names)

    rows = []
    for n in names:
        rows.append({
            "小区名": n,
            "城区": guess_district(n),
            "学区(1/0)": "",            # 待补充:市重点学区 1,否则 0
            "楼龄(年)": guess_age(n),
            "绿化率%": guess_green(n),
            "容积率": guess_density(n),
            "距地铁km": "",             # 待补充:最近地铁站直线距离
            "周边商业(1/0)": "",        # 待补充:1km 内综合体/商业街
            "周边公园(1/0)": "",        # 待补充:1km 内公园
            "主力户型": "",             # 待补充:如 89-118㎡三房
        })
    df = pd.DataFrame(rows)
    df.to_excel(OUT, index=False)
    print(f"已生成 {OUT}: {len(df)} 个小区")
    print("注意:学区/地铁/商业/公园列为空,可让 AI 从小区信息填充,或人工补充;")
    print("     城区/楼龄/绿化率/容积率为启发式初稿,请抽查修正(准确值直接改表)。")


if __name__ == "__main__":
    main()
