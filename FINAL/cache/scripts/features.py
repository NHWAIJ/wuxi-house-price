# -*- coding: utf-8 -*-
"""
features.py — 特征工程 + 事件系统。

事件系统(让预测"结合当下事件"而不只是历史价格):
  1. 自动打分:从事件年表的文本里按关键词识别收紧/宽松方向(可扩展关键词表);
  2. 外部事件:可选 data/events_extra.csv(date,score,note) 追加/覆盖自动打分结果;
  3. 情景事件流:预测期事件由"最近12个月事件节奏"自动生成三种变体
     (基准=延续节奏, 乐观=利好增强, 悲观=利空增强),也可手工在配置里指定。

特征列(每行=一个月):
  t_idx / year / month_sin / month_cos        时间
  p_lag1/3/6/12, r_lag1/3/6/12                滞后价格与收益率
  ma6 / ma12 / vol12                          移动平均与波动率
  dd_ratio / months_since_peak                距峰值回撤
  ev_net6 / ev_net12 / ev_net24               事件滚动净方向
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA_EVENTS_FILE = os.path.join(ROOT, "data", "events_extra.csv")

# ---------------------------------------------------------------------------
# 事件打分关键词(文本中出现即 ±1,可自行增删)
# ---------------------------------------------------------------------------
NEG_KEYWORDS = [
    "参考价", "限购", "限售", "认房又认贷", "收紧", "严查", "上调", "提高",
    "增值税", "二套", "三道红线", "两道红线", "利率上调", "放款周期", "额度紧张",
    "停贷", "断供", "信心崩塌", "悲观", "下跌", "下行", "降温", "观望",
    "以价换量", "挂牌量", "供过于求", "库存", "承压", "折价", "让利", "抛售",
    "回落", "走低", "冰封", "腰斩", "低迷", "乏力", "城投托底", "挤泡沫",
]
POS_KEYWORDS = [
    "取消限购", "放松", "放宽", "解除", "降息", "下调", "降低", "LPR",
    "首付20%", "首套20%", "认房不认贷", "锡二十条", "锡十条", "救市", "十七条",
    "政策红利", "利好", "回暖", "反弹", "上涨", "热销", "售罄", "三开三罄",
    "土拍", "地价", "成交回升", "小阳春", "刚需释放", "改善需求", "补贴",
    "收储", "货币化", "以价换量见成效", "刚需入市", "止跌",
]
PROJECT_CATEGORY_KW = ("项目节点", "项目", "开盘", "交付")

FEATURE_COLS = [
    "t_idx", "year", "month_sin", "month_cos",
    "p_lag1", "p_lag3", "p_lag6", "p_lag12",
    "r_lag1", "r_lag3", "r_lag6", "r_lag12",
    "ma6", "ma12", "vol12", "dd_ratio", "months_since_peak",
    "ev_net6", "ev_net12", "ev_net24",
]


# ---------------------------------------------------------------- 事件打分

# 「放松/放宽/解除/取消/松绑」+ 限制类名词 = 政策宽松(正面),负面计数时需跳过这类组合
RELAX_VERBS = ("放松", "放宽", "解除", "取消", "松绑", "优化")


def score_event(text, category=""):
    """事件文本 → 方向分(截断到 ±3)。项目节点类事件(开盘/交付等)默认 0,避免干扰二手预测。
    修复:正面短语(如"放松限购")被移除后残留负面名词"限购",导致正负相抵为 0。
    现改为:负面词计数时,跳过紧跟"放松/放宽/解除/取消"等动词的负面名词。"""
    import re as _re
    if any(k in category for k in PROJECT_CATEGORY_KW):
        # 仅当文本明显含政策/金融词时仍计入
        if not any(k in text for k in ("政策", "贷款", "利率", "限购", "限售", "增值税", "首付")):
            return 0
    s = sum(1 for k in POS_KEYWORDS if k in text)
    n = 0
    for k in NEG_KEYWORDS:
        for m in _re.finditer(_re.escape(k), text):
            pre = text[max(0, m.start() - 2):m.start()]
            if any(pre.endswith(v) for v in RELAX_VERBS):
                continue    # "放松限购"类 → 政策宽松,不计负面
            n += 1
    return int(max(-3, min(3, s - n)))


def load_extra_events():
    """可选外部事件文件 data/events_extra.csv:列 date(YYYY-MM), score(±1..3), note。"""
    if not os.path.exists(EXTRA_EVENTS_FILE):
        return pd.DataFrame(columns=["date", "score", "note"])
    df = pd.read_csv(EXTRA_EVENTS_FILE, encoding="utf-8-sig", comment="#")
    df["date"] = pd.to_datetime(df["date"] + "-01") + pd.offsets.MonthEnd(0)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    return df


def build_events(complex_events, extra_events=None):
    """
    事件年表(load_data 解析的 DataFrame) + 外部事件 → 统一月度方向序列。
    返回 Series(月末 index → 当月事件净方向)。
    """
    parts = []
    if complex_events is not None and len(complex_events):
        scored = pd.DataFrame({
            "date": pd.to_datetime(complex_events["date"]),
            "score": [score_event(t, c) for t, c in
                      zip(complex_events["text"], complex_events["category"])],
        })
        scored = scored[scored["score"] != 0]
        parts.append(scored)
    if extra_events is not None and len(extra_events):
        parts.append(extra_events[["date", "score"]])
    if not parts:
        return pd.Series(dtype=float)
    all_ev = pd.concat(parts, ignore_index=True)
    s = all_ev.groupby("date")["score"].sum().sort_index()
    return s.astype(float)


# ---------------------------------------------------------------- 特征构建

def build_price_features(df):
    """价格/时间特征(不含事件;事件用 attach_events 单独加)。
    df 需含 date 与 price 列;price 未来月可为 NaN(递归预测时逐月回填)。

    重要:所有特征在 t 月只使用 ≤ t-1 的信息(无泄漏),训练与预测口径一致——
    这样递归预测时,未来月特征可用"已填的历史/预测值"直接重算。
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    p = df["price"]
    prev = p.shift(1)                     # 截至 t-1 的价格序列
    df["t_idx"] = np.arange(len(df), dtype=float)
    df["year"] = df["date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    for lag in (1, 3, 6, 12):
        df[f"p_lag{lag}"] = p.shift(lag)
        df[f"r_lag{lag}"] = p.shift(lag).pct_change(1)   # lag 月的已知收益
    df["ma6"] = prev.rolling(6, min_periods=1).mean()     # t-6..t-1 均值
    df["ma12"] = prev.rolling(12, min_periods=1).mean()
    df["vol12"] = p.pct_change().shift(1).rolling(12).std()  # t-12..t-1 波动率
    peak_sofar = prev.cummax()                            # 截至 t-1 的历史峰值
    df["dd_ratio"] = prev / peak_sofar
    ms, peak_pos = [], None
    for i in range(len(df)):
        if np.isnan(prev.values[i]):
            ms.append(ms[-1] + 1 if ms else np.nan)
            continue
        if peak_pos is None or prev.values[i] >= prev.values[peak_pos] * (1 - 1e-9):
            peak_pos = i
        ms.append(i - peak_pos)
    df["months_since_peak"] = ms
    return df


def attach_events(df, events_series):
    """为每行计算近 6/12/24 个月事件净方向(事件当月计入,窗口 (t-6, t])。"""
    df = df.sort_values("date").reset_index(drop=True)
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="ME")
    ev = events_series.reindex(idx).fillna(0.0)
    for win, col in ((6, "ev_net6"), (12, "ev_net12"), (24, "ev_net24")):
        roll = ev.rolling(win, min_periods=1).sum()
        df[col] = roll.reindex(df["date"]).values
    return df


def build_full_features(df, events_series):
    """价格特征 + 事件特征一步到位。"""
    return attach_events(build_price_features(df), events_series)


# ---------------------------------------------------------------- 小区属性特征

ATTRS_FILE = os.path.join(ROOT, "data", "complex_attrs.xlsx")
ATTR_NUM = ["楼龄(年)", "绿化率%", "容积率", "距地铁km"]
ATTR_BIN = ["学区(1/0)", "周边商业(1/0)", "周边公园(1/0)"]
DISTRICT_COLS = ["宜兴市", "江阴市", "滨湖区", "锡山区", "惠山区", "梁溪区", "新吴区"]
ATTRIBUTE_COLS = ATTR_NUM + ATTR_BIN + DISTRICT_COLS


def load_attrs(path=None):
    """
    读取小区属性表 → DataFrame(index=小区名, 列=ATTRIBUTE_COLS)。
    数值列转 float;二进制列 0/1;城区 one-hot(未知城区全 0)。
    读取失败返回 None(训练自动退化为无属性特征)。
    """
    try:
        df = pd.read_excel(path or ATTRS_FILE)
    except Exception as e:
        print(f"  ⚠ 属性表读取失败({e}),本次无属性特征")
        return None
    if df is None or len(df) == 0 or "小区名" not in df.columns:
        return None
    df = df.dropna(subset=["小区名"]).copy()
    df["小区名"] = df["小区名"].astype(str)
    df = df.set_index("小区名")
    num = df[ATTR_NUM].apply(pd.to_numeric, errors="coerce")
    bins = {c: pd.to_numeric(df[c], errors="coerce").clip(0, 1) for c in ATTR_BIN}
    dist = df["城区"].fillna("待补充").astype(str)
    oh = pd.get_dummies(dist).reindex(columns=DISTRICT_COLS, fill_value=0).astype(float)
    out = pd.concat([num, pd.DataFrame(bins), oh], axis=1)
    return out[ATTRIBUTE_COLS]


# ---------------------------------------------------------------- 宏观月度数据

MACRO_FILE = os.path.join(ROOT, "data", "macro.xlsx")
MACRO_COLS = ["lpr", "lpr_chg", "city_price", "city_ret6"]


def load_macro(path=None):
    """
    读取宏观月度数据 → {sheet: 月末索引 DataFrame}(数值列 ffill + bfill)。
    sheets: 5年期LPR(2019-08起,完整) / 全市二手均价_聚合(2023-01起,稀疏) / 各区域均价。
    缺失月份前向填充;起始月之前用最早值垫底(北控 2021 起,全市均价 2023 起)。
    读取失败返回空 dict(训练继续,自动退化为无宏观特征)。
    """
    try:
        xl = pd.read_excel(path or MACRO_FILE, sheet_name=None)
    except Exception as e:
        print(f"  ⚠ 宏观数据读取失败({e}),本次无宏观特征")
        return {}
    out = {}
    for sheet, df in xl.items():
        if "月份" not in df.columns:
            continue
        df = df.copy()
        df["月份"] = pd.to_datetime(df["月份"]) + pd.offsets.MonthEnd(0)  # 对齐月末
        df = df.set_index("月份").sort_index()
        num = df.select_dtypes(include="number")
        if num.empty:
            continue
        out[sheet] = num.ffill().bfill()
    return out


def build_macro_frame(macro, dates):
    """
    从宏观 dict 构造与 dates 对齐的宏观特征 DataFrame(行序一致):
      lpr        5年期LPR水平(预测期持平假设:未来月自动取最后已知值)
      lpr_chg    LPR 月度变化
      city_price 全市二手均价(ffill;2023 前用最早值垫底)
      city_ret6  全市均价 6 个月变化率(预测期随持平而衰减为 0)
    dates: 月末 Timestamp 序列(历史 + 未来)。
    """
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)
    out["lpr"] = np.nan
    out["lpr_chg"] = np.nan
    out["city_price"] = np.nan
    out["city_ret6"] = np.nan
    if macro:
        if "5年期LPR" in macro:
            lpr = macro["5年期LPR"]["5年期以上LPR(%)"]
            out["lpr"] = lpr.reindex(idx).ffill().bfill()
            out["lpr_chg"] = out["lpr"].diff().fillna(0.0)
        if "全市二手均价_聚合" in macro:
            cp = macro["全市二手均价_聚合"]["全市均价(元/㎡)"]
            out["city_price"] = cp.reindex(idx).ffill().bfill()
            out["city_ret6"] = out["city_price"].pct_change(6).fillna(0.0)
    return out[MACRO_COLS]


# ---------------------------------------------------------------- 情景事件流

def future_event_flow(events_series, scenario, n_months):
    """
    预测期(未来 n 个月)事件流,基于最近 12 个月的实际事件节奏生成:
      基准 = 最近12个月逐月净方向原样重复;
      乐观 = 只保留正向(负向清零),正向强度×2;
      悲观 = 只保留负向(正向清零),负向强度×2。
    返回 Series(未来月末 index → 净方向)。
    """
    last = events_series.index.max()
    if pd.isna(last):
        # 无任何事件数据:返回全 0 未来事件流(起点取当天,下游 reindex 后不影响)
        future_dates = pd.date_range(pd.Timestamp.now().normalize(),
                                     periods=n_months, freq="ME")
        return pd.Series(0.0, index=future_dates)
    recent = events_series[events_series.index > last - pd.offsets.DateOffset(months=12)]
    future_dates = pd.date_range(last + pd.offsets.MonthEnd(1), periods=n_months, freq="ME")
    # 把最近12个月的月度净方向铺到未来(逐月循环)
    base = recent.reindex(pd.date_range(recent.index.min(), last, freq="ME")).fillna(0).values
    if len(base) == 0:
        base = np.zeros(12)
    flow = np.tile(base, int(np.ceil(n_months / len(base))))[:n_months]
    if scenario == "乐观":
        # 保留并加倍正向事件、清零负向,再整体 +1:相当于政策强宽松(类2023年)
        flow = np.where(flow < 0, 0, flow * 2) + 1
    elif scenario == "悲观":
        # 保留并加倍负向事件、清零正向,再整体 -1:相当于政策持续收紧(类2021→2022)
        flow = np.where(flow > 0, 0, flow * 2) - 1
    return pd.Series(flow, index=future_dates, dtype=float)


def scenario_events(events_series, scenario, n_months):
    """历史事件 + 未来情景事件 → 完整事件序列(用于特征计算)。"""
    fut = future_event_flow(events_series, scenario, n_months)
    return pd.concat([events_series, fut]).sort_index()
