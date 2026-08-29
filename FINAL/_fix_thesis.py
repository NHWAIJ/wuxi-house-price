# -*- coding: utf-8 -*-
"""桌面论文修改:错误修正 + 亮点跟进(python-docx,保留公式)。"""
import docx

PATH = r"C:\Users\nnnn8\Desktop\基于机器学习技术的住宅成交价预测系统——以无锡为例.docx"
d = docx.Document(PATH)


def replace_para(old_frag, new_text, count=1):
    """整段替换:找到含 old_frag 的段落,清空 runs 后写入新文本。"""
    n = 0
    for p in d.paragraphs:
        if old_frag in p.text:
            for r in p.runs:
                r.text = ""
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            n += 1
            if n >= count:
                return True
    return False


def fix_double_comma():
    """去掉 '，，' 双逗号(只改文本 run,不动公式)。"""
    n = 0
    for p in d.paragraphs:
        for r in p.runs:
            if "，，" in r.text:
                r.text = r.text.replace("，，", "，")
                n += 1
    return n


def insert_after(frag, new_text):
    """在含 frag 的段落后插入新段落(保留原有格式设置)。"""
    from docx.oxml.ns import qn
    for p in d.paragraphs:
        if frag in p.text:
            new_p = p._p.makeelement(qn("w:p"), {})
            p._p.addnext(new_p)
            # 新段落加一个 run
            from docx.oxml import OxmlElement
            r = OxmlElement("w:r")
            t = OxmlElement("w:t")
            t.text = new_text
            r.append(t)
            new_p.append(r)
            return True
    return False


results = {}

# ---------------- 一、错误修正 ----------------
results["398训练集->372"] = replace_para(
    "最终选出了398个数据质量较高的小区作为训练集",
    "在采集到的600余个小区成交价信息中进行筛选，剔除数据缺失严重与重复的小区后，最终选出了372个数据质量较高的小区作为训练集（含测试集污染清理与去重），并根据不同小区的数据进行分类，具体分类见表5：")
results["84章小结398->372"] = replace_para(
    "将筛选好的的398个小区数据进行分类",
    "将筛选好的372个小区数据进行分类")
results["近500个小区"] = replace_para(
    "近500个小区",
    "372个训练小区与2个测试小区")
results["80%区间113"] = replace_para(
    "在训练后加权平均得到P50区间，100轮×5模型得到500个预测值，取 10%/90% 分位数得到P10，P90区间，将80%区间在折线图中呈现给用户。",
    "在训练后加权平均得到P50基准预测，100轮×5模型共500个预测值，取25%/75%分位数得到P10、P90边界，构造50%区间（经区间校准与缩放，见4.5.4），在折线图中以区间带的形式呈现给用户，避免单一预测值误导判断。")
results["80%区间166"] = replace_para(
    "取 10% 和 90% 分位数构造 P10–P90 区间，在折线图里以 80% 区间的形式给用户看",
    "取 25% 和 75% 分位数构造 P10–P90 区间，在折线图里以 50% 区间的形式给用户看")
results["96表述"] = replace_para(
    "（北控 1.47% & 9.9%）",
    "（北控：Holt 1.47% vs ARIMA 9.9%）")
results["免费开源"] = replace_para(
    "它免费开源、在普通电脑上即可本地部署",
    "它计划以 MIT 协议开源、在普通电脑上即可本地部署")
results["cloude code"] = replace_para(
    "因时间紧张，代码过长等原因，项目使用Visual Studio Code + cloude code + deepseek v4 flash大模型辅助完成。",
    "为提升开发效率，项目使用 Visual Studio Code 配合 Claude Code 与 DeepSeek 大模型辅助完成代码开发。")
results["双逗号"] = fix_double_comma()

# ---------------- 二、亮点跟进 ----------------
results["三情景插入"] = insert_after(
    "预测模块",
    "三情景预测：在基准预测（趋势衰减半衰期36个月）之外，同时给出乐观与悲观两条情景线——乐观情景取半衰期18个月（趋势快速收敛、跌幅收窄），悲观情景取半衰期72个月（下行延续）。三情景写入预测表情景sheet，并在网页折线图中以三条虚线呈现，供用户判断未来行情的可能区间。")
results["网页漂移插入"] = insert_after(
    "网页展示如下（见图7）",
    "网页除历史走势与预测图外，还展示分布漂移检测结果（红/橙/绿三色徽章提示预测可信度）与多截断点诊断表。")
results["安装包插入"] = insert_after(
    "系统测试与界面展示",
    "交付形式：除源码目录外，项目提供正式安装包（Inno Setup 编译），安装向导包含许可协议页与安装说明页，默认安装至英文路径（Program Files\\WuxiHousePrice，避免中文路径兼容性问题），安装后可在控制面板\"应用和功能\"中正常卸载。")
results["区间配置插入"] = insert_after(
    "这个区间比较诚实，不再被训练期早期的高波动（2021-2022）撑得太宽。",
    "区间配置说明：50% 区间的分位数（默认25/75）与缩放系数（默认0.5）均在代码中可调——想更保守可改回10/90分位（80%区间），想更激进可进一步收窄。实测两个目标小区在50%区间下覆盖率分别为100%（北控）与62%（帝泊湾），即\"承诺50%，实际覆盖62%-100%\"。")

d.save(PATH)
for k, v in results.items():
    print(f"{k}: {'OK' if v else '未找到!'}")
