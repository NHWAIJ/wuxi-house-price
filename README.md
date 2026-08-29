# 无锡住宅房价预测系统 —— 交付说明

> 本文件夹（D:\BK\FINAL）即**全部交付物**：代码 + 数据 + 模型 + 全部产物。
> 换一台电脑，把本文件夹**整体复制**过去，按本说明安装依赖后即可运行。

---

## 一、目录结构

```text
D:\BK\
└── FINAL\                              ★ 全部交付(本文件夹)
    ├── README.md                       本说明(从零开始运行指南)
    ├── cache\                          全部交付物(数据/代码/模型/产物/安装包)
    │   ├── data\                       数据
    │   │   ├── train\
    │   │   │   ├── 01_deep_fall\       深跌未反弹
    │   │   │   ├── 02_v_rebound\       V 型反弹
    │   │   │   ├── 03_flat_decline\    横盘阴跌
    │   │   │   ├── 04_rising\          逆势上涨
    │   │   │   └── 06_regular\         其他(常规)
    │   │   ├── test\                   测试集(北控雁栖湖、新力帝泊湾,与训练集物理隔离)
    │   │   ├── complex_attrs.xlsx      小区属性表(学区/楼龄/绿化率/城区等,可补充)
    │   │   └── macro.xlsx              宏观月度数据(LPR/全市均价/区域均价)
    │   ├── config.yaml                 集中配置(所有可调参数,修改后重跑即生效)
    │   ├── scripts\                    全部代码
    │   │   ├── config_loader.py        配置加载器(读取 config.yaml,自动回退默认值)
    │   │   ├── experiment_tracker.py   实验追踪(每次训练/预测记录到 experiments.csv)
    │   │   ├── validate_data.py        数据校验(检查新小区表格格式/价格合理性)
    │   │   └── tests\                  单元测试
    │   │       ├── test_config.py      配置测试
    │   │       └── test_validate.py    数据校验测试
    │   ├── web\                        网页应用(Flask 后端 + 前端页面)
    │   │   ├── templates\
    │   │   │   └── index.html
    │   │   └── app.py
    │   ├── models\                     联合模型(ensemble_all.joblib,约 2-3GB,训练产物)
    │   ├── charts\                     历史走势图(自动清理)
    │   ├── comparison\{1,2,3,…}\       每次预测:对比图 + 预测延伸图(编号递增保留)
    │   ├── output\                     预测表 + 打分表 + 对比表(XLSX)
    │   ├── logs\                       运行日志
    │   ├── defense_deliverables\       答辩交付物
    │   ├── 04_code\                    正式运行包(脚本+数据副本)
    │   │   ├── data\                   数据副本(同 data 结构)
    │   │   ├── scripts\                代码副本
    │   │   ├── requirements.txt
    │   │   └── run.bat
    │   ├── dist_setup\                 安装包目录
    │   ├── _deleted\                   已移走文件备份(确认后手动删除)
    │   │   └── 房价预测系统_APP副本\
    │   ├── INSTALL_INFO.txt
    │   ├── LICENSE.txt
    │   └── answer.md
    ├── train.bat                       训练 + 自动预测(10-12 分钟)
    ├── predict.bat                     直接预测(1-2 分钟)
    ├── diagnose.bat                    目标小区诊断(多截断点)+ 正式预测
    ├── test.bat                        随机小区测试
    ├── web.bat                         网页演示(自动打开浏览器)
    ├── 编译安装包.bat                  编译安装包(Inno Setup)
    ├── Dockerfile                      容器化(一行命令构建 Docker 镜像)
    ├── requirements.txt                依赖清单(根目录,供 Docker/新环境使用)
    └── _fix_thesis.py                  论文辅助脚本
```

---

## 二、新电脑环境搭建(首次使用)

### 0. 安装(正式安装包)或直接github部署整个项目
双击 `房价预测系统_安装包.exe` —— 完整应用安装引导:
欢迎页 → 许可协议(MIT)→ 安装说明 → **选择安装路径**(默认
`C:\Program Files\WuxiHousePrice`,英文路径,避免中文路径兼容性问题)
→ 快捷方式 → 安装完成。可在**控制面板应用和功能中正常卸载**(带卸载向导)。

> 所有代码路径均为自动检测,整个文件夹换位置可直接运行。

### 1. 安装 Python

- 下载 **Python 3.9 及以上(64 位)**:https://www.python.org/downloads/
- 安装时务必勾选 **"Add Python to PATH"**

**验证 Python 是否安装成功**(任一命令能输出版本号即成功):

```PowerShell
python --version
```

或(Windows 多版本场景):

```PowerShell
py --version
```

能显示类似 `Python 3.13.2` 即安装成功;若提示"不是内部或外部命令",说明未勾选 PATH 或未安装。

### 2. 安装依赖(一条命令,不依赖任何文件)

打开命令行(CMD),直接执行(无需 requirements.txt 文件):

```PowerShell
pip install pandas numpy scikit-learn statsmodels openpyxl matplotlib joblib lightgbm xgboost flask
```

安装内容:pandas / numpy / scikit-learn / statsmodels / openpyxl / matplotlib / joblib / lightgbm / xgboost / flask。
**不需要 GPU、不需要 CUDA**(训练自动多核并行,8 核约 10-12 分钟)。

### 3. 验证环境(依赖是否装好)

```PowerShell
python -c "import pandas, sklearn, statsmodels, openpyxl, matplotlib, lightgbm, xgboost, flask; print('OK')"
```

看到 `OK` 即可开始使用;若报 ModuleNotFoundError,说明对应依赖未装成功,重新执行第 2 步。

> 注:首次双击 bat 时若 Windows 弹"受保护"提示,点"更多信息 → 仍要运行"。

---

## 三、怎么运行

| 操作            | 方式                      | 耗时       | 产出                                |
| --------------- | ------------------------- | ---------- | ----------------------------------- |
| 训练 + 自动预测 | 双击`train.bat`         | 10-12 分钟 | 新模型 + 打分表追加 + 对比图/预测图 |
| 只预测(不重训)  | 双击`predict.bat`       | 1-2 分钟   | 打分表追加 + 对比图/预测图          |
| 目标小区诊断    | 双击`diagnose.bat`      | 2-3 分钟   | 多截断点诊断(控制台)+ 预测表        |
| 随机小区测试    | 双击`test.bat`          | 5-10 分钟  | 随机 5 个小区打分                   |
| 网页演示(答辩)  | 双击`web.bat`           | 秒级启动   | 浏览器:选小区→历史/预测/诊断/漂移  |
| 同步正式包      | 双击`collect_final.bat` | 秒级       | 04_code/ 更新为最新 scripts + data  |

每次训练/预测都会弹出**进度窗口**(带进度条,不会"未响应"),完成后:

- 打分追加到 `output\accuracy_metrics.xlsx`(主表,含"特征占比"sheet)与 `output\打分表.xlsx`
- **对比折线图 + 预测延伸折线图**(至 2031.12,50% 区间 + 三情景)存入 `comparison\{新编号}\`(编号递增,永久保留)
- 预测表(含情景 sheet)存入 `output\predictions_{小区}.xlsx`

### 首次运行建议顺序

1. 双击 `train.bat` —— 用现有 372 个小区 + 宏观/属性数据训练联合模型,并自动对北控/帝泊湾做测试预测
2. 双击 `web.bat` —— 网页浏览预测结果(答辩演示)
3. 之后日常:有新数据就按第四/五节操作,再训练

---

## 四、加入训练集(新增小区参与训练)

### 1. 准备小区表格(Excel)

格式要求(系统自动识别,无需固定模板):

- 表内有一行表头,含**日期信息**:「年份」+「月份」两列,或「日期」一列;
- 有**价格列**,列名含「售价/均价/价格」等(毛坯/精装/别墅/综合自动识别);
- 表名含「**二手**」→ 识别为二手房数据;含「**新房**」→ 新房数据;
- (可选)含「事件/政策/年表」工作表 → 自动提取政策事件特征;
- 参考现有文件:`data\train\01_deep_fall\` 下的任意一个。

### 2. 放入 incoming 文件夹(按形态分类)

在 `data\incoming\` 下按小区走势形态放入(分类名是训练多样性关键):

```
data\incoming\
├── 分类_深跌未反弹\     ← 从高点跌 30%+ 且未反弹(最缺这类!)
├── 分类_V型反弹\        ← 跌后快速收复
├── 分类_横盘阴跌\       ← 长期缓慢下滑
├── 分类_逆势上涨\       ← 大盘跌它涨
└── 分类_邻居板块_胡埭马山太湖\   ← 滨湖区一带(目标小区邻居)
```

> 分类不准确也没关系,运行归位脚本时会按实际走势自动核对。
> 历史不足 12 个月的表会被自动剔除;12 个月以上均可参与。

### 3. 自动去重、分类归位

```cmd
python scripts\_organize_data.py
```

自动完成:删除与现有重复的文件 → 按分类移到 `data\train\对应文件夹` → 无用的移入 `_deleted\`。

### 4. 重新训练

双击 `train.bat`。

---

## 五、加入测试集(验证预测准不准的小区)

### 1. 准备小区表格

格式同第四节,但该小区**必须有 2024.09 之后的真实价格数据**——系统会把 2024.08 前的数据作为训练窗口,用 2024.09 起的真实价格来打分。

### 2. 放入测试集

把表格复制到 `data\test\`:

```cmd
copy 某小区_价格数据表.xlsx data\test\
```

> 测试集与训练集**物理隔离**:放在 `data\test\` 的小区永远不会参与联合模型训练,
> 打分结果才是"模型对陌生小区的真实能力"。

### 3. 运行预测

双击 `predict.bat`(或 `diagnose.bat` 先做多截断点诊断)—— 自动对 `data\test\` 下**所有**小区:

- 截断到 2024.08 → Holt 趋势外推 + 季节叠加 + 50% 区间 → 预测 2024.09 起 22 个月;
- 与真实价格对比打分(0-100),追加到 `output\accuracy_metrics.xlsx`;
- **对比图 + 预测延伸图(至 2031.12,含三情景)** 存入 `comparison\{新编号}\`;
- 自动输出**分布漂移检测**(当前行情是否在训练分布内,降级提示)。

> 若某小区 2024.09 后没有真实价格,会输出预测但**无法打分**(没有可对比的真实值),属正常现象。

---

## 六、模型与数据说明

| 项                    | 说明                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------- |
| 目标小区(北控/帝泊湾) | **Holt 趋势外推 + 季节叠加 + 趋势衰减**(实证最优:北控 98.3 分 / 帝泊湾 88.0 分) |
| ALL 新小区            | **联合模型**(Ridge+RF+GBR+XGB+LightGBM 五模型等权,100 轮 × 500 树,并行训练)    |
| 训练集                | 372 个唯一小区 × 中位 33 个月历史,按形态分 6 类                                      |
| 测试集                | 北控/帝泊湾(2024.09 起真实数据对比,不参与训练)                                        |
| 特征                  | 价格形态 21 维 + 政策事件 3 维 + 宏观 4 维(LPR/全市均价)+ 小区属性 14 维              |
| 预测区间              | **50% 区间**(P10-P90,数据驱动分位数,可配置)                                     |
| 新增能力              | 分布漂移检测(可信度降级提示)· 三情景预测(基准/乐观/悲观)· 多截断点诊断 · 网页演示  |
| 打分                  | 0-100 分(MAE/RMSE/MAPE/覆盖率综合),每次训练追加                                       |

---

## 七、常见问题

**Q: 训练要多久?**
A: 8 核 CPU 约 10-12 分钟(100 轮 × 5 模型并行);核心越多越快。数据更多时可用记事本打开
`scripts\train_all.py`,把 `TRAIN_ROUNDS = 100` 调成 60 可减半。

**Q: 预测要多久?**
A: 1-2 分钟——主要耗时是解析 372 个训练小区构建漂移检测分布;Holt 预测本身为秒级。

**Q: 模型文件太大(2-3GB)?**
A: 100 轮 × 500 棵树的产物。磁盘紧张可删 `models\ensemble_all.joblib`
(下次 `train.bat` 会重新生成),或调低 `TRAIN_ROUNDS`。

**Q: 弹窗显示"未响应"?**
A: 不会。进度窗口运行在独立线程,计算在主线程外进行,窗口始终可操作。

**Q: 报错 ModuleNotFoundError?**
A: 依赖没装全,执行: `pip install pandas numpy scikit-learn statsmodels openpyxl matplotlib joblib lightgbm xgboost flask`

**Q: 区间怎么调整?**
A: 打开 `scripts\predict.py` 顶部,`INTERVAL_QUANTILES = (25, 75)` 为 50% 区间
(改 `(10, 90)` 为 80% 区间),`INTERVAL_SCALE = 0.5` 为缩放系数。
