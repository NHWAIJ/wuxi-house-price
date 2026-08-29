# -*- coding: utf-8 -*-
"""
validate_data 测试

运行: python -m pytest scripts/tests/test_validate.py -v
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validate_data import validate_complex


def _make_complex(name, dates, prices, **kw):
    """构造测试用小区 dict。"""
    return {
        "name": name,
        "old": pd.DataFrame({"date": pd.to_datetime(dates), "price": prices, **kw}),
    }


def test_valid_data():
    """正常数据应通过校验。"""
    dates = pd.date_range("2020-01", periods=36, freq="MS")
    prices = [10000 + i * 100 for i in range(36)]
    c = _make_complex("测试小区", dates, prices)
    issues = validate_complex(c, min_months=12)
    assert len(issues) == 0, f"正常数据不应有校验问题: {issues}"


def test_missing_price_column():
    """缺少价格列应报错。"""
    c = {
        "name": "缺价格",
        "old": pd.DataFrame({"date": pd.date_range("2020-01", periods=12, freq="MS")}),
    }
    issues = validate_complex(c)
    assert any("price" in i for i in issues), "应提示缺少价格列"


def test_too_short_history():
    """历史不足应报错。"""
    dates = pd.date_range("2024-01", periods=3, freq="MS")
    prices = [10000, 10100, 10200]
    c = _make_complex("短历史", dates, prices)
    issues = validate_complex(c, min_months=12)
    assert any("历史" in i for i in issues), "应提示历史不足"


def test_empty_price():
    """价格全部为空应报错。"""
    dates = pd.date_range("2020-01", periods=12, freq="MS")
    prices = [None] * 12
    c = _make_complex("空价格", dates, prices)
    issues = validate_complex(c)
    assert any("全部为空" in i for i in issues), "应提示价格全空"


def test_negative_price():
    """负价格应报错。"""
    dates = pd.date_range("2020-01", periods=12, freq="MS")
    prices = [10000, -1000, 10100, 10200, 10300, 10400,
              10500, 10600, 10700, 10800, 10900, 11000]
    c = _make_complex("负价格", dates, prices)
    issues = validate_complex(c)
    assert any("非正值" in i for i in issues), "应提示非正值"


def test_duplicate_dates():
    """重复日期应报错。"""
    dates = pd.to_datetime(["2020-01", "2020-01", "2020-02", "2020-03"])
    prices = [10000, 10100, 10200, 10300]
    c = _make_complex("重复日期", dates, prices)
    issues = validate_complex(c, min_months=1)
    assert any("重复" in i for i in issues), "应提示重复月份"


def test_unreasonable_unit():
    """价格单位异常应报错。"""
    dates = pd.date_range("2020-01", periods=12, freq="MS")
    prices = [500] * 12  # 仅 500 元/㎡
    c = _make_complex("单位异常", dates, prices)
    issues = validate_complex(c)
    assert any("千元" in i for i in issues), "应提示单位可能错误"