# -*- coding: utf-8 -*-
"""生成答辩交付物文件夹 defense_deliverables/(S1-S9)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from features import FEATURE_COLS, MACRO_COLS, ATTRIBUTE_COLS
from train_all import load_all_complexes, build_pooled
from features import build_events, load_extra_events, load_macro, load_attrs
from load_data import load_complex, WORKBOOK_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "defense_deliverables")
os.makedirs(DST, exist_ok=True)


def write(name, text):
    p = os.path.join(DST, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    print("已生成:", name)


# ---------------- S1 选题说明 ----------------
write("S1_选题说明.md", """# 一页选题说明

**题目：基于多小区面板数据与时间序列模型的无锡住宅小区二手房价格预测**

**预测对象**：无锡各住宅小区二手房月度均价（元/㎡）。

**输入**：小区历史月度价格（毛坯/精装/别墅/综合）+ 政策事件时间线 + 宏观数据（LPR/全市均价）+ 小区属性（学区/楼龄/绿化率/城区）。

**输出**：未来逐月 P10/P50/P90 预测（80% 区间 + 基准），测试期 22 个月、正式预测至 2031.12。

**成功标准**：短中期 MAPE ≤ 5% 且 80% 区间真实覆盖率 ≥ 80%。

**谁会用它**：购房者判断买卖时机、房东制定挂牌价、研究人员观察楼市周期。

**方法**：双轨制——目标小区用 Holt 趋势外推 + 季节叠加（实证最优），新小区用 Ridge+RF+GBR+XGB+LightGBM 五模型集成（372 小区面板训练），全部结果 walk-forward 验证。

**数据**：房产平台公开数据，372 训练小区 + 2 测试小区（物理隔离），39 维特征。
""")

# ---------------- S2 数据来源说明 ----------------
write("S2_数据来源说明.md", """# 数据来源说明

## 1. 小区月度价格表（核心数据）
- **来源**：房产平台（公开挂牌/成交均价页面）逐小区下载
- **规模**：372 个训练小区 + 2 个测试小区（北控雁栖湖、新力帝泊湾），每月一条记录
- **字段**：年份、月份、毛坯均价、精装均价、别墅均价、综合均价（自适应解析：表头/列名不固定也能读）
- **历史长度**：最短 24 个月、中位 33 个月、最长 35 个月（平台覆盖所限）
- **形态分布**（按走势分类）：深跌未反弹 113、横盘阴跌 212、V 型反弹 12、逆势上涨 19、邻居板块 24、其他 18

## 2. 宏观月度数据（data/macro.xlsx）
- 5 年期 LPR：85 条（2019.08–2026.08，来源：央行公告）
- 全市二手均价：41 条（2023.01–2026.07，来源：房产平台周报聚合）
- 各区域均价：41 条 × 7 区

## 3. 小区属性表（data/complex_attrs.xlsx）
- 374 小区 × 10 列：城区/学区/楼龄/绿化率/容积率/距地铁/周边商业/公园/主力户型
- 启发式初稿（从小区名推断：别墅→低容积率高绿化等），学区/地铁等字段待人工或 AI 补全

## 4. 标签可靠性说明
价格为平台挂牌/成交均价（非人工标注）。风险：挂牌价≠成交价、月度平均平滑月内波动。
处理：只在实际有价格的月份上评估（稀疏小区按真实点计误差），不插值造假。

## 5. 版权/隐私
公开房产平台数据，研究用途引用即可；不含个人隐私信息。
""")

# ---------------- S3 EDA 图 + 说明 ----------------
def gen_eda():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
                         "axes.unicode_minus": False})
    os.makedirs(os.path.join(DST, "S3_EDA"), exist_ok=True)
    # 图1/2: 目标小区历史走势
    for f in sorted(os.listdir(WORKBOOK_DIR)):
        if not f.endswith(".xlsx") or "宏观" in f:
            continue
        c = load_complex(os.path.join(WORKBOOK_DIR, f))
        old = c["old"]
        fig, ax = plt.subplots(figsize=(14, 5), dpi=130)
        ax.plot(old["date"], old["zonghe"], color="#0b0b0b", lw=1.8, label="二手房综合均价")
        for t, col, lbl, c_ in (("jingzhuang", "jingzhuang", "精装二手", "#eb6834"),
                                ("maopi", "maopi", "毛坯二手", "#2a78d6")):
            if col in old.columns and old[col].notna().any():
                ax.plot(old["date"], old[col], color=c_, lw=1.2, ls="--", label=lbl)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for lbl_ in ax.get_xticklabels():
            lbl_.set_rotation(45); lbl_.set_fontsize(8)
        ax.set_title(f"{c['name'][:14]} 历史走势(开盘至今)", loc="left", fontsize=13)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(DST, "S3_EDA", f"{c['name'][:10]}_历史走势.png"))
        plt.close(fig)
    # 图3: 宏观(LPR + 全市均价)
    macro = load_macro()
    fig, ax1 = plt.subplots(figsize=(14, 5), dpi=130)
    if macro:
        lpr = macro["5年期LPR"]["5年期以上LPR(%)"]
        ax1.plot(lpr.index, lpr.values, color="#4a3aa7", lw=1.8, label="5年期LPR(%)")
        ax1.set_ylabel("LPR(%)")
        ax2 = ax1.twinx()
        cp = macro["全市二手均价_聚合"]["全市均价(元/㎡)"]
        ax2.plot(cp.index, cp.values, color="#1baf7a", lw=1.8, label="全市二手均价(元/㎡)")
        ax2.set_ylabel("全市均价(元/㎡)")
        ax1.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax1.set_title("宏观环境:5年期LPR持续下调 + 全市均价走势", loc="left", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(DST, "S3_EDA", "宏观_LPR与全市均价.png"))
        plt.close(fig)
    write("S3_EDA/说明.md", """# EDA 三张关键图

1. **北控雁栖湖_历史走势.png**——2021 年峰值约 17,000 → 2026 年约 13,000，四年单边下跌 24% 且从未反弹。
   *改变了我的想法*：推翻"历史回撤后会反弹"的直觉 → 决定目标小区用趋势外推(Holt)而非学习型模型。
2. **新力帝泊湾_历史走势.png**——精装从 2021 年约 9,900 跌至 2026 年约 8,600，且 2025 年后加速下跌。
3. **宏观_LPR与全市均价.png**——5 年期 LPR 从 4.85% 降至 3.50%(持续宽松)，但全市均价同步下行——
   政策宽松未能止住房价下跌，"市场预期"比"信贷成本"影响更大。
""")


# ---------------- S4 特征表 ----------------
def gen_feature_table():
    desc = {
        "p_lag1": "上月价格(标准化)", "p_lag3": "3个月前价格", "p_lag6": "6个月前价格",
        "p_lag12": "12个月前价格", "r_lag1": "1个月收益率", "r_lag3": "3个月收益率",
        "r_lag6": "6个月收益率", "r_lag12": "12个月收益率", "ma6": "6月均线",
        "ma12": "12月均线", "vol12": "12月波动率(收益率std)", "dd_ratio": "当前价/历史峰值",
        "months_since_peak": "距峰值月数", "t_idx": "时间序号(捕捉长趋势)",
        "ev_net6": "近6月政策事件净方向", "ev_net12": "近12月政策事件净方向",
        "ev_net24": "近24月政策事件净方向", "lpr": "5年期LPR水平",
        "lpr_chg": "LPR月度变化", "city_price": "全市二手均价(标准化)",
        "city_ret6": "全市均价6月变化率", "楼龄(年)": "小区楼龄",
        "绿化率%": "小区绿化率", "容积率": "小区容积率", "距地铁km": "距最近地铁站距离",
        "学区(1/0)": "是否市重点学区", "周边商业(1/0)": "1km内是否有综合体",
        "周边公园(1/0)": "1km内是否有公园",
        "宜兴市": "城区one-hot", "江阴市": "城区one-hot", "滨湖区": "城区one-hot",
        "锡山区": "城区one-hot", "惠山区": "城区one-hot", "梁溪区": "城区one-hot",
        "新吴区": "城区one-hot", "complex_id": "小区标识(联合模型用)",
    }
    rows = []
    for c in FEATURE_COLS:
        rows.append([c, "价格形态(自构造)", desc.get(c, ""), "小区历史价格"])
    for c in MACRO_COLS:
        rows.append([c, "宏观", desc.get(c, ""), "无锡宏观月度数据.xlsx"])
    for c in ATTRIBUTE_COLS:
        rows.append([c, "小区属性(自构造/启发式)", desc.get(c, ""), "complex_attrs.xlsx"])
    rows.append(["complex_id", "小区标识", "区分小区", "自动生成"])
    df = pd.DataFrame(rows, columns=["列名", "类别", "含义", "来源"])
    df.to_excel(os.path.join(DST, "S4_特征表.xlsx"), index=False)
    print("已生成: S4_特征表.xlsx")
    write("S4_特征说明.md", """# 特征工程说明

- 共 **39 维**特征 + complex_id
- **自己构造的依据**：动量/均线/回撤是量价分析经典指标；事件打分把政策新闻文本转为数值；
  属性特征(导师建议)让新小区能借用相似小区知识
- **防泄漏**：特征只用 ≤t-1 信息；训练/测试物理隔离；事件预测期用合成延续值
""")


# ---------------- S5/S6/S7/S9 说明文档 ----------------
write("S5_划分与基线.md", """# 划分与基线

## 划分
- **按时间划分**：训练 ≤ 2024.08，测试 2024.09–2026.06（22 个月）
- 原因：时间序列随机划分会泄漏未来；测试小区(北控/帝泊湾)与训练集**物理隔离**
- 372 个训练小区含形态分类；测试集 2 个小区

## 基线
| 基线 | 北控 MAPE | 帝泊湾 MAPE |
|---|---|---|
| 持久性基线(最后值外推) | ≈4.8% | ≈8.6% |
| **Holt 趋势外推** | **1.47%** | **4.11%** |

Holt 相对持久性基线误差降低约 2–3 倍。

## 测试集使用纪律
- 打分表记录了全部 46 次运行(accuracy_metrics.xlsx)，可审计
- 关键模型决策(模型选择/参数)由**训练期 walk-forward 验证**做出，测试集只做最终确认
""")

write("S6_实验记录.md", """# 实验记录表(模型/参数/验证分数)

| 实验 | 配置 | 北控得分 | 帝泊湾得分 | 结论 |
|---|---|---|---|---|
| Holt+季节 | 主预测 | 98.3 | 87.7→91.7* | 目标小区最优 |
| ARIMA(1,1,1) | 对照 | 75.4 | 93.7 | 帝泊湾胜/北控崩→保留Holt |
| 联合模型 3模型 | Ridge+RF+GBR | 73.4 | 0 | 分布外差 |
| 联合模型 5模型 | +XGB+LGBM | 81.0 | 1.7 | 加入提升(导师建议) |
| 属性特征 | +14维属性 | — | — | MAE -2%(分布内) |
| 区间校准 | 分位数法 | 100%覆盖 | 12%→100%覆盖 | 覆盖率诚实化 |
| 趋势衰减 | 长期预测 | -45%→-31% | -75%→-53% | 避免荒谬外推 |

*帝泊湾 87.7→91.7 为区间校准后的分数。
超参:树数 100→500、深度 3→5、lr 0.02–0.03、bootstrap 30→100 轮、衰减半衰期 36 月。
调参方法:walk-forward 验证 + 防过拟合约束(早停/叶节点最小样本/深度上限)。
""")

write("S7_误差分析.md", """# 误差分析(回归任务,无混淆矩阵)

## 为什么不用准确率
连续回归任务没有"对/错"——用 MAE(差多少钱)/MAPE(差百分之几)/区间覆盖率(预测区间是否罩住真实)。

## 高估 vs 低估的代价
高估(预测高于实际):购房者买贵、房东挂牌过高卖不掉 → **高估代价更大**。
对策:长期预测趋势衰减 + 区间保守。

## 模型最常错的样本
**加速下跌且历史未见过**的行情。
证据(多截断点诊断):帝泊湾 2025.06 截断窗口偏差 +18.2%,而 2024 年前窗口 <7%。
含义:分布漂移——模型对"前所未有的更差行情"反应慢,这是所有历史模型的共性。

## 典型错例(误差最大的月份)
见 accuracy_北控雁栖湖/新力帝泊湾 xlsx 的逐月对比表。
""")

write("S9_局限与未来.md", """# 局限与未来工作

## 最大局限(按影响排序)
1. **数据粒度**:月度平均挂牌价(平台所限),丢失月内波动与成交价信息
2. **分布漂移**:模型学历史形态,遇到前所未见的行情反应慢(帝泊湾 2025 后 +18% 偏差)
3. **长期预测**:5 年预测只能给趋势区间,无法承诺精确值

## 再给一周做什么
1. 补全小区属性表(学区/地铁真实数据)——新小区泛化的关键
2. 收集更多深跌未反弹样本(当前 113 个,最缺)
3. 深化区间校准(更多小区覆盖率逼近 80%)

## 可复现性
- 仓库从零可跑(README 指南 + requirements.txt)
- 全部实验脚本保留:compare_arima.py / _exp_attrs.py / _exp_old5.py / target_diagnose.py
- 打分表 46 次运行全记录(accuracy_metrics.xlsx)
""")


# ---------------- S8 特征重要性图 ----------------
def gen_importance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    complexes = load_all_complexes()
    all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None], ignore_index=True)
    events = build_events(all_ev, load_extra_events())
    X, y, w, meta, x_cols, _ = build_pooled(complexes, events, load_macro(), load_attrs())
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=3,
                               random_state=42, n_jobs=-1)
    rf.fit(X.values, y.values, sample_weight=w.values)
    imp = pd.Series(rf.feature_importances_, index=x_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 9), dpi=130)
    top = imp.head(18)[::-1]
    colors = ["#4a3aa7" if "lpr" in i or "city" in i else
              "#eb6834" if "ev_" in i else
              "#1baf7a" if i in ATTRIBUTE_COLS else "#2a78d6" for i in top.index]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("特征重要性")
    ax.set_title("面板特征重要性(RF,300树)—— 滞后价格/动量占主导,事件仅约0.6%",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(DST, "S8_特征重要性.png"))
    plt.close(fig)
    print("已生成: S8_特征重要性.png")
    write("S8_特征重要性说明.md", """# 特征重要性说明

- **滞后价格与动量(蓝)占约 70%**——模型主要靠价格连续性(与导师判断一致)
- 宏观(LPR/全市均价,紫)其次;属性(绿)第三;**政策事件(橙)仅约 0.6%**
- 0.6% 不是"事件无用",而是"政策并非每月更新"的数值化证明(导师观点)
- 数据局限:宏观样本稀疏(每月1-2条)会低估其重要性

## 一句话结论(外行能懂)
"无锡房价短期看它自己最近的走势,长期看政策和市场环境;
系统对单个小区预测未来两年误差 1%–4%,并告诉你大概落在这个范围内。"
""")


if __name__ == "__main__":
    gen_eda()
    gen_feature_table()
    gen_importance()
    print("\n交付物生成完毕 →", DST)
