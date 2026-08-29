# -*- coding: utf-8 -*-
"""
run_all.py — 一键全流程(代码独立完成训练 + 出图 + 出表):
  自适应读表 → 共享事件构建 → walk-forward 验证(集成 vs Holt)
  → 历史走势图 → 基准情景多轮递归预测(综合/精装二手/毛坯或别墅二手,至2031.12)
  → 预测表格(预测结果/目录,整数) → 预测延伸折线图(charts/目录)。

运行:  python scripts/run_all.py
  或双击 D:\\BK\\运行房价预测.bat(自动弹出进度窗口 + 控制台)。
换小区: 把新的价格数据表放入 workbook/ 目录(自动发现),满足 load_data 的最小要求即可。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from load_data import discover, load_complex
from features import build_events, load_extra_events
from models import walk_forward, summarize
from forecast import forecast_complex, export, print_overview
from plot_forecast import plot_all
from plot_monthly import plot_all_history

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_metrics(mdf):
    """验证指标表输出为 xlsx(等线、居中、细边框,与预测表同风格)。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side
    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    for j, h in enumerate(mdf.columns, 1):
        cell = ws.cell(1, j, h)
        cell.font = Font(name="等线", size=12, bold=True)
        cell.alignment = center
        cell.border = border
    ws.row_dimensions[1].height = 18
    for i, (_, row) in enumerate(mdf.iterrows()):
        r = i + 2
        for j, v in enumerate(row, 1):
            cell = ws.cell(r, j, float(v) if isinstance(v, (int, float)) else v)
            cell.font = Font(name="等线", size=12)
            cell.alignment = center
            cell.border = border
        ws.row_dimensions[r].height = 18
    ws.column_dimensions["A"].width = 24
    for col in "BCDEFGH":
        ws.column_dimensions[col].width = 13
    wb.save(os.path.join(OUTPUT_DIR, "metrics.xlsx"))


def _make_progress():
    """创建进度弹窗;设置环境变量 BK_NO_GUI=1 或环境无桌面时回退为控制台输出。"""
    if os.environ.get("BK_NO_GUI"):
        return None
    try:
        from progress_gui import ProgressGUI
        return ProgressGUI()
    except Exception:
        return None


def main():
    # ---- 进度窗口(回退模式 pw=None 时仍用 print) ----
    pw = _make_progress()
    try:
        _run(pw)
    except Exception as e:
        if pw is not None:
            pw.update(f"运行出错: {e}", 100, log=str(e))
            pw.done(["请查看控制台完整报错信息"])
            pw.root.mainloop()
        raise


def _run(pw):
    last_pct = [0]

    def report(text, pct, log=None):
        if pw is not None:
            pw.update(text, pct, log)
        else:
            # 控制台模式也显示进度(每 5% 一行,不静默)
            if int(pct) >= last_pct[0] + 5 or pct >= 100:
                last_pct[0] = int(pct)
                print(f"  [{pct:3.0f}%] {text}", flush=True)
        if log:
            print("  " + log, flush=True)

    files = discover()
    if not files:
        report("未找到价格数据表", 0, "workbook/ 下没有 xlsx 文件")
        return
    report(f"发现 {len(files)} 个小区文件", 2,
           f"{[os.path.basename(f) for f in files]}")
    print("=" * 90)

    complexes = [load_complex(fp) for fp in files]

    # 全局事件:两小区同板块,政策/市场事件共享(如帝泊湾年表只到2020,靠北控表补足2021+)
    all_events = pd.concat([c["events"] for c in complexes
                            if c["events"] is not None], ignore_index=True)
    events = build_events(all_events, load_extra_events())
    report("事件构建完成", 5, f"共享政策/市场事件 {len(events)} 条")

    metrics_rows = []
    for i, c in enumerate(complexes):
        name = c["name"]
        report(f"[{i + 1}/{len(complexes)}] {name}:解析与验证", 6 + i * 2,
               f"新房 {len(c['new'])} 行 / 二手 {len(c['old'])} 行"
               f"(实际点 {c['old']['is_actual'].sum()})")

        # 1) walk-forward 验证:集成 vs Holt
        wf = walk_forward(c["old"], events)
        ens_s, holt_s = summarize(wf["ensemble"]), summarize(wf["holt"])
        print(f"  walk-forward 验证(预测步长1-12月):")
        print(f"    集成(Ridge+RF+GBR): MAE={ens_s['MAE']:,.0f}  RMSE={ens_s['RMSE']:,.0f}"
              f"  MAPE={ens_s['MAPE%']:.2f}%  (验证月数 {ens_s['验证月数']})")
        print(f"    Holt 基线:          MAE={holt_s['MAE']:,.0f}  RMSE={holt_s['RMSE']:,.0f}"
              f"  MAPE={holt_s['MAPE%']:.2f}%  (验证月数 {holt_s['验证月数']})")
        metrics_rows.append({
            "小区": name,
            "集成_MAE": ens_s["MAE"], "集成_RMSE": ens_s["RMSE"], "集成_MAPE%": ens_s["MAPE%"],
            "Holt_MAE": holt_s["MAE"], "Holt_RMSE": holt_s["RMSE"], "Holt_MAPE%": holt_s["MAPE%"],
            "验证月数": ens_s["验证月数"],
        })
        report(f"[{i + 1}/{len(complexes)}] {name}:验证完成", 11 + i * 2,
               f"集成 MAPE={ens_s['MAPE%']:.2f}%  vs  Holt MAPE={holt_s['MAPE%']:.2f}%")

    # 2) 历史走势折线图
    for i, c in enumerate(complexes):
        report(f"[{i + 1}/{len(complexes)}] {c['name']}:绘制历史走势图", 16 + i * 2,
               f"{c['name']}: 历史折线图(逐月X轴)")
        plot_all_history(c)

    # 3) 基准情景多轮递归预测(进度条在 20%→90% 区间随轮数推进)
    from forecast import targets_for
    results = {}
    total_rounds_all = sum(30 * len(targets_for(c["old"])) for c in complexes)
    done_rounds = [0]

    def _on_round(done, total):
        done_rounds[0] += 1
        report(f"训练预测中:{done_rounds[0]}/{total_rounds_all} 轮",
               20 + 70 * done_rounds[0] / total_rounds_all, None)

    for i, c in enumerate(complexes):
        name = c["name"]
        report(f"[{i + 1}/{len(complexes)}] {name}:多轮训练预测", 20,
               f"{name}: 30轮bootstrap × 3序列 递归预测66个月")
        result = forecast_complex(c, events, on_round=_on_round)
        results[name] = result
        xlsx_path = export(name, result)
        print_overview(name, c["old"], result)
        report(f"{name}:预测表已输出", 92, f"预测明细(Excel) → {xlsx_path}")

    # 4) 预测延伸折线图
    for i, c in enumerate(complexes):
        report(f"[{i + 1}/{len(complexes)}] {c['name']}:绘制预测延伸图", 93 + i * 3,
               f"{c['name']}: 预测延伸图(至2031.12,逐月X轴)")
        plot_all(c, results[c["name"]])

    # 汇总指标表(与预测表同为 xlsx 模板样式)
    mdf = pd.DataFrame(metrics_rows)
    export_metrics(mdf)
    print("\n" + "=" * 90)
    print("验证指标对比(集成 vs Holt),已存 预测结果/metrics.xlsx:")
    print(mdf.round(1).to_string(index=False))

    # 完成
    lines = [
        f"4 张折线图 → {os.path.join(ROOT, 'charts')}",
        f"2 张预测表 + 验证指标 → {os.path.join(ROOT, 'output')}",
        "预测范围:2026.07–2031.12(66个月,基准情景 + 80%区间)",
    ]
    report("全部完成", 100, "全部完成:4 张折线图 + 2 张预测表 + 验证指标")
    if pw is not None:
        pw.done(lines)
        pw.root.mainloop()


if __name__ == "__main__":
    main()
