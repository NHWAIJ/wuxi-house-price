# -*- coding: utf-8 -*-
"""
load_data.py — 自适应读取价格数据表,不依赖固定表头/固定列名。

设计目标:以后更换任何小区/任何形式的 Excel 价格表,只要满足以下最低要求即可自动解析:
  1. 表内有一行表头,含日期信息(「年份」+「月份」两列,或「日期」一列);
  2. 有价格列,列名含「售价/均价/价格」等字样(毛坯/精装/别墅/综合等类型自动识别);
  3. 文件名或表名能区分新房与二手房(含「新房/二手」字样即可;列名「售价/均价」也可兜底);
  4. 若存在事件年表(表名含「事件/政策/年表」),自动提取事件时间线用于事件特征。

输出统一结构(每个小区一个 dict):
  {
    "name": 小区名,
    "new": DataFrame(date, maopi, jingzhuang, bieshu, zonghe),   # 新房段
    "old": DataFrame(date, maopi, jingzhuang, bieshu, zonghe, is_actual),  # 二手房段(稀疏自动插值)
    "events": DataFrame(date, text, category, score),            # 事件时间线(score 见 features.py)
  }
"""
import os
import re

import openpyxl
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKBOOK_DIR = os.path.join(ROOT, "data", "test")   # 测试版:数据来自 CHECK/数据快照(正式版为 workbook)

# ---------------------------------------------------------------------------
# 关键词表(可自行扩展;列名匹配用)
# ---------------------------------------------------------------------------
DATE_YEAR_KW = ("年份", "年")
DATE_MONTH_KW = ("月份", "月")
DATE_SINGLE_KW = ("日期", "时间")
PRICE_KW = ("售价", "均价", "价格", "单价")
EXCLUDE_KW = ("差价", "环比", "变化", "涨跌", "价差", "%", "指数")
TYPE_KW = {
    "maopi": ("毛坯", "清水"),
    "jingzhuang": ("精装", "装修", "装标", "成品"),
    "bieshu": ("别墅", "叠墅", "独栋", "合院", "排屋"),
    "zonghe": ("综合", "总体", "全部", "平均价"),
}
MARK_KW = ("标记", "数据点", "★")
SOURCE_KW = ("来源", "出处")
EVENT_SHEET_KW = ("事件", "政策", "年表")
NEW_SHEET_KW = ("新房", "一手", "在售")
OLD_SHEET_KW = ("二手", "存量", "成交参考价")

PRICE_COLS = ("maopi", "jingzhuang", "bieshu", "zonghe")


# ---------------------------------------------------------------- 基础工具

def _clean(v):
    return "" if v is None else str(v).strip()


def _to_float(v):
    s = _clean(v).replace(",", "")
    if s in ("", "—", "-", "--", "售罄", "None", "nan", "NaN"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _rows(path, sheet_name, wb=None):
    own = wb is None
    if own:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb[sheet_name]
        rows = [r for r in ws.iter_rows(values_only=True) if any(v is not None for v in r)]
    finally:
        if own:
            wb.close()
    return rows


def _is_header_row(cells):
    """判断一行是否为表头:严格含「年份」+「月份」或「日期」;宽松模式识别单字「年」+「月」,
    但排除含时间范围(~、—、单位等)的标题行。"""
    strict_year = any("年份" in c for c in cells)
    strict_month = any("月份" in c for c in cells)
    has_date = any(any(k in c for k in DATE_SINGLE_KW) for c in cells)
    if (strict_year and strict_month) or has_date:
        return True
    # 宽松:单字「年」「月」,但该行不能是"2019.09~2021.09"式范围标题
    if any("年" in c and "月" in c for c in cells):
        joined = "|".join(cells)
        if any(sym in joined for sym in ("~", "—", "–", "单位", "元/㎡", "房价", "均价")):
            return False
        return True
    return False


def _find_header(rows):
    """找表头行(严格优先,宽松兜底)。返回(列名列表, 数据起始行)。"""
    strict = []
    for i, row in enumerate(rows):
        cells = [_clean(c) for c in row]
        if _is_header_row(cells):
            strict.append((i, cells))
    if strict:
        # 优先严格匹配(含「年份」「月份」「日期」字段名的行)
        def _strictness(cells):
            s = sum(1 for c in cells if any(k in c for k in ("年份", "月份", "日期", "时间")))
            return -s
        i, cells = min(strict, key=lambda t: (_strictness(t[1]), t[0]))
        return cells, i + 1
    raise ValueError("未找到表头行(需含「年份」+「月份」或「日期」列)")


def _to_month(date_text, year=None, month=None):
    """把日期文本/年月转成月末 datetime。"""
    if year is not None and month is not None:
        y = re.sub(r"[^\d]", "", _clean(year))     # 兼容「2019年」格式
        m = re.sub(r"[^\d]", "", _clean(month))
        if y and m:
            return pd.Timestamp(int(y), int(m), 1) + pd.offsets.MonthEnd(0)
        return pd.NaT
    d = _clean(date_text)
    m = re.match(r"(\d{4})[.\-/年](\d{1,2})", d)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1) + pd.offsets.MonthEnd(0)
    return pd.NaT


def _col_type(name):
    """列名 → 价格类型(maopi/jingzhuang/bieshu/zonghe/None)。"""
    for t, kws in TYPE_KW.items():
        if any(k in name for k in kws):
            return t
    return None


# ---------------------------------------------------------------- 单表解析

def parse_price_sheet(path, sheet_name, wb=None):
    """
    解析一张价格表 → (DataFrame(date, 各价格列), is_actual列, 新房/二手标记)。
    自动识别:表头、日期列、价格列类型、数据标记列(★=实际点)、数据来源列。
    """
    rows = _rows(path, sheet_name, wb)
    header, start = _find_header(rows)
    df = pd.DataFrame(rows[start:], columns=header)
    df = df.dropna(how="all").reset_index(drop=True)

    # --- 日期 ---
    date_col = next((c for c in df.columns if any(k in c for k in DATE_SINGLE_KW)), None)
    if date_col:
        df["date"] = [_to_month(d) for d in df[date_col]]
    else:
        y_col = next(c for c in df.columns if any(k in c for k in DATE_YEAR_KW))
        m_col = next(c for c in df.columns if any(k in c for k in DATE_MONTH_KW))
        df["date"] = [_to_month(None, y, m) for y, m in zip(df[y_col], df[m_col])]
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # --- 价格列 ---
    price_map = {}
    for c in df.columns:
        if any(k in c for k in PRICE_KW) and not any(k in c for k in EXCLUDE_KW):
            t = _col_type(c)
            if t:
                price_map[t] = c
    if not price_map:
        raise ValueError(f"{sheet_name}: 未识别到价格列(列名需含 售价/均价/价格 及 毛坯/精装/别墅/综合)")
    if "zonghe" not in price_map and len(price_map) > 1:
        # 无综合列 → 用其他类型均值当综合
        df["_zonghe_auto"] = df[[price_map[t] for t in price_map]].apply(
            lambda r: pd.to_numeric(r, errors="coerce").mean(), axis=1)
        price_map["zonghe"] = "_zonghe_auto"
    for t in PRICE_COLS:
        if t not in price_map:
            df[t] = None
        else:
            df[t] = [_to_float(v) for v in df[price_map[t]]]

    # --- 数据标记(★=实际,~=插值估算;排序前按原始行序对齐) ---
    mark_col = next((c for c in df.columns if any(k in c for k in MARK_KW)), None)
    if mark_col:
        df["is_actual"] = [("★" in _clean(v)) for v in df[mark_col]]
    else:
        df["is_actual"] = True

    df = df[["date"] + list(PRICE_COLS) + ["is_actual"]]

    # --- 新房/二手判定:表名优先,列名兜底 ---
    phase = None
    if any(k in sheet_name for k in OLD_SHEET_KW):
        phase = "old"
    elif any(k in sheet_name for k in NEW_SHEET_KW):
        phase = "new"
    if phase is None:
        # 列名兜底:「均价」更像二手
        avg_cols = [c for c in header if "均价" in c]
        phase = "old" if avg_cols else "new"
    return df, phase


def monthlyize(old_df):
    """稀疏二手房表补成逐月;is_actual 标记真实数据点;插值段线性补齐。
    空表(无有效日期)直接返回空 df。"""
    if old_df is None or len(old_df) == 0 or old_df["date"].isna().all():
        return old_df.iloc[0:0] if old_df is not None else pd.DataFrame(
            columns=["date", "zonghe", "maopi", "jingzhuang", "bieshu", "is_actual"])
    dates = pd.date_range(old_df["date"].min(), old_df["date"].max(), freq="ME")
    merged = pd.DataFrame({"date": dates}).merge(old_df, on="date", how="left")
    merged["is_actual"] = merged["zonghe"].notna()
    for c in PRICE_COLS:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").interpolate(
            method="linear", limit_direction="both")
    return merged


# ---------------------------------------------------------------- 事件表解析

def parse_event_sheet(path, sheet_name, wb=None):
    """解析事件年表 → DataFrame(date, text, category)。"""
    rows = _rows(path, sheet_name, wb)
    header, start = _find_header(rows)
    df = pd.DataFrame(rows[start:], columns=header).dropna(how="all").reset_index(drop=True)
    date_col = next((c for c in df.columns if any(k in c for k in DATE_SINGLE_KW)), None)
    if date_col is None:
        y_col = next(c for c in df.columns if any(k in c for k in DATE_YEAR_KW))
        m_col = next(c for c in df.columns if any(k in c for k in DATE_MONTH_KW))
        df["_d"] = [_to_month(None, y, m) for y, m in zip(df[y_col], df[m_col])]
    else:
        df["_d"] = [_to_month(d) for d in df[date_col]]
    df = df.dropna(subset=["_d"]).sort_values("_d").reset_index(drop=True)

    text_col = next((c for c in df.columns if "描述" in c), None)
    if text_col is None:
        text_col = next((c for c in df.columns if "事件" in c and "分类" not in c), None)
    if text_col is None:
        text_col = next(c for c in df.columns if c not in (date_col, "_d", "日期") and "影响" not in c)
    cat_col = next((c for c in df.columns if "分类" in c or "类型" in c), None)
    out = pd.DataFrame({
        "date": df["_d"],
        "text": df[text_col].apply(_clean),
        "category": df[cat_col].apply(_clean) if cat_col else "",
    })
    return out


# ---------------------------------------------------------------- 小区级加载

def load_complex(file_path):
    """解析一个小区文件(内含新房表/二手房表/事件年表)。
    修复:同一文件只打开一次 workbook(此前每个 sheet 各开一次);
    多个二手房表时合并(按日期去重)后再插值。"""
    name = os.path.splitext(os.path.basename(file_path))[0]
    wb = openpyxl.load_workbook(file_path, read_only=True)
    try:
        sheets = wb.sheetnames
        new_df = None
        old_dfs = []
        events = None
        for sh in sheets:
            if any(k in sh for k in EVENT_SHEET_KW):
                try:
                    events = parse_event_sheet(file_path, sh, wb)
                except Exception:
                    continue
                continue
            try:
                df, phase = parse_price_sheet(file_path, sh, wb)
            except Exception:
                continue
            if phase == "new" and new_df is None:
                new_df = df
            elif phase == "old":
                old_dfs.append(df)
    finally:
        wb.close()

    # 多个二手房表合并(按日期去重,先到先得)后统一插值
    old_df = None
    if old_dfs:
        merged = pd.concat(old_dfs, ignore_index=True)
        merged = merged.sort_values("date").drop_duplicates(
            subset="date", keep="first").reset_index(drop=True)
        old_df = monthlyize(merged)
        if old_df is not None and len(old_df) == 0:
            old_df = None   # 空二手房表(只有表头无数据)→ 视为无二手房
    if new_df is None and old_df is None:
        raise ValueError(f"{name}: 未解析到任何价格表")
    if old_df is None:
        raise ValueError(f"{name}: 未找到二手房表(表名需含「二手」)")
    if new_df is None:
        # 只有二手房表也允许(如未来数据只给二手)
        new_df = pd.DataFrame(columns=["date"] + list(PRICE_COLS))
    # 统一接口:price = 综合均价(特征/模型统一用 price 列)
    old_df["price"] = old_df["zonghe"]
    if len(new_df):
        new_df["price"] = new_df["zonghe"]
    return {"name": name, "new": new_df, "old": old_df, "events": events}


def discover():
    """自动发现 workbook 目录下所有 xlsx 小区文件。"""
    files = sorted(f for f in os.listdir(WORKBOOK_DIR)
                   if f.lower().endswith((".xlsx", ".xlsm")) and not f.startswith("~$"))
    return [os.path.join(WORKBOOK_DIR, f) for f in files]


if __name__ == "__main__":
    for fp in discover():
        c = load_complex(fp)
        print(f"[{c['name']}] 新房 {len(c['new'])} 行,"
              f" 二手 {len(c['old'])} 行(实际点 {c['old']['is_actual'].sum()}),"
              f" 事件 {0 if c['events'] is None else len(c['events'])} 条")
