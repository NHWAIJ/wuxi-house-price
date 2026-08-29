# -*- coding: utf-8 -*-
"""
validate_data.py — 数据校验模块

对新加入的小区数据表进行格式校验,确保:
  1. 必填列存在(年份+月份 或 日期,以及价格列)
  2. 价格数值合理(无异常值、单位正确)
  3. 日期连续、无重复
  4. 历史月数 >= 最小要求

用法:
    from validate_data import validate_complex
    issues = validate_complex(complex_dict)
    if issues:
        for i in issues:
            print(f"  ⚠ {i}")
"""
import pandas as pd
import numpy as np


def validate_complex(c, min_months=12):
    """
    校验小区数据,返回问题列表(空列表 = 全部通过)。

    参数:
        c: load_complex() 返回的 dict,含 "old"(二手房 df)和 "name"
        min_months: 最小历史月数

    返回:
        [issue_str, ...]
    """
    issues = []
    name = c.get("name", "未知")
    old = c.get("old")

    # 1. 数据存在性
    if old is None or len(old) == 0:
        return [f"[{name}] 无二手房数据"]

    # 2. 必填列
    for col in ["date", "price"]:
        if col not in old.columns:
            issues.append(f"[{name}] 缺少必填列: {col}")

    if issues:
        return issues

    # 3. 日期列检查
    dates = old["date"]
    if dates.isna().any():
        issues.append(f"[{name}] 日期列存在空值")

    # 4. 日期排序 & 连续性
    date_sorted = dates.sort_values()
    date_range = (date_sorted.max() - date_sorted.min()).days / 30.44
    if date_range < min_months - 1:
        issues.append(f"[{name}] 历史跨度仅 {date_range:.0f} 个月,建议 ≥{min_months} 个月")

    # 5. 价格检查
    prices = old["price"].dropna()
    if len(prices) == 0:
        issues.append(f"[{name}] 价格列全部为空")
        return issues

    if (prices <= 0).any():
        n_neg = (prices <= 0).sum()
        issues.append(f"[{name}] 价格含 {n_neg} 个非正值")

    # 6. 异常价格检测(与中位数的偏差超过 5 倍 IQR)
    median_p = prices.median()
    iqr = np.percentile(prices, 75) - np.percentile(prices, 25)
    if iqr > 0:
        lower = median_p - 5 * iqr
        upper = median_p + 5 * iqr
        outliers = prices[(prices < lower) | (prices > upper)]
        if len(outliers) > 0:
            issues.append(f"[{name}] 发现 {len(outliers)} 个异常价格值"
                          f"(超出中位数 ±5×IQR 范围: [{lower:.0f}, {upper:.0f}])")

    # 7. 价格单位合理性(无锡房价:单价应在 3000-100000 元/㎡)
    if prices.median() < 3000:
        issues.append(f"[{name}] 价格中位数仅 {prices.median():.0f} 元/㎡,"
                      f"可能单位错误(如录入为千元而非元)")
    if prices.median() > 100000:
        issues.append(f"[{name}] 价格中位数达 {prices.median():.0f} 元/㎡,"
                      f"请确认单位正确")

    # 8. 月份重复检查
    dup = dates[dates.duplicated()]
    if len(dup) > 0:
        issues.append(f"[{name}] 存在 {len(dup)} 个重复月份: {dup.tolist()[:5]}")

    # 9. 数据点数量
    if len(prices) < min_months:
        issues.append(f"[{name}] 仅有 {len(prices)} 个有效价格数据点"
                      f"(建议 ≥{min_months})")

    return issues


def validate_all(complexes, min_months=12):
    """
    批量校验多个小区。

    返回: {小区名: [issue, ...]}
    """
    results = {}
    for c in complexes:
        issues = validate_complex(c, min_months)
        if issues:
            results[c.get("name", "未知")] = issues
    return results


def print_validation_report(results):
    """打印校验报告。"""
    if not results:
        print("✓ 所有小区数据校验通过")
        return
    print(f"\n{'='*60}")
    print(f"数据校验报告: {len(results)} 个小区存在问题")
    print(f"{'='*60}")
    for name, issues in sorted(results.items()):
        print(f"\n  [{name}]")
        for issue in issues:
            print(f"    ⚠ {issue}")
    print(f"\n{'='*60}")