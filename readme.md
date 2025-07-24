# 转债数据处理系统

一个专门用于处理可转换债券（转债）数据的Python工具，提供数据清洗、合并、指标计算和分析功能。

## 🚀 特性

- **数据清洗**: 自动处理多个Excel工作表的数据清洗
- **数据合并**: 智能合并历史数据和新数据，支持去重
- **指标计算**: 自动计算转债相关技术指标
- **数据分析**: 生成详细的分析报告和统计信息
- **模块化设计**: 清晰的步骤划分，便于维护和扩展
- **配置管理**: 集中的配置文件管理所有路径和参数

## 📁 项目结构

```
cb_weekly_update/
├── py/                     # Python源代码
│   ├── config.py          # 配置文件
│   ├── main.py            # 主程序入口
│   ├── 1_data_cleaning.py       # 步骤1：数据清洗
│   ├── 2_data_merging.py        # 步骤2：数据合并
│   ├── 3_metrics_calculation.py # 步骤3：指标计算
│   ├── 4_data_analysis.py       # 步骤4：数据分析
│   └── readme.md          # 详细技术文档
├── date/                   # 输入数据目录
├── output/                 # 输出结果目录
├── .gitignore             # Git忽略文件
└── README.md              # 项目说明文档
```

## 🛠️ 系统要求

- Python 3.7+
- Windows 操作系统
- 必需的Python包：
  - pandas
  - openpyxl
  - datetime (内置)
  - os (内置)

## 📦 安装

1. 克隆仓库：
```bash
git clone https://github.com/your-username/cb_weekly_update.git
cd cb_weekly_update
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据文件：
   - 将输入数据文件放入 `date/` 目录
   - 确保文件名与配置文件中的设置一致

## 🚀 使用方法

### 快速开始

```bash
cd py
python main.py
```

### 分步执行

```bash
# 步骤1：数据清洗
python 1_data_cleaning.py

# 步骤2：数据合并
python 2_data_merging.py

# 步骤3：指标计算
python 3_metrics_calculation.py

# 步骤4：数据分析
python 4_data_analysis.py
```

## 📊 数据处理流程

1. **数据清洗** (`1_data_cleaning.py`)
   - 读取原始Excel文件的多个工作表
   - 清洗和标准化数据格式
   - 输出清洗后的数据文件

2. **数据合并** (`2_data_merging.py`)
   - 合并新数据和历史数据
   - 执行数据去重操作
   - 生成合并后的完整数据集

3. **指标计算** (`3_metrics_calculation.py`)
   - 计算转债相关技术指标
   - 生成衍生数据字段
   - 输出包含指标的数据文件

4. **数据分析** (`4_data_analysis.py`)
   - 执行统计分析
   - 计算分位数和变化率
   - 生成分析报告

## ⚙️ 配置说明

主要配置在 `py/config.py` 文件中：

```python
# 输入数据目录
DATA_ROOT = os.path.join(PROJECT_ROOT, 'date')

# 输出数据目录
PROCESS_DIR = os.path.join(PROJECT_ROOT, 'output')

# 分析参数
CHANGE_PERIOD_DAYS = 5
TRADING_DAYS_SINCE_2018 = 1825
TRADING_DAYS_SINCE_2021 = 1094
```

## 📋 输入文件要求

程序需要以下输入文件（放置在 `date/` 目录）：

1. **原始数据文件**：
   - `1.1 日度数据-清洗前转债基础数据-813支-20250711-20250718-无链接.xlsx`
   - 包含多个工作表的转债基础数据

2. **历史数据文件**：
   - `2.1 日度数据-清洗后转债基础数据-去重-20180102-20250711(1).xlsx`
   - 用于数据合并的历史数据

## 📤 输出文件

程序在 `output/` 目录生成以下文件：

- `1_cleaned_data_YYYYMMDD.xlsx` - 清洗后的数据
- `2_merged_data_YYYYMMDD.xlsx` - 合并后的数据
- `3_calculated_metrics_YYYYMMDD.xlsx` - 计算指标后的数据
- `4_analysis_report_YYYYMMDD.xlsx` - 最终分析报告

## 🔧 故障排除

### 常见问题

1. **文件路径错误**
   - 检查 `config.py` 中的路径配置
   - 确保输入文件存在于 `date/` 目录

2. **依赖包缺失**
   ```bash
   pip install -r requirements.txt
   ```

3. **权限问题**
   - 确保对 `output/` 目录有写入权限
   - 检查Excel文件是否被其他程序占用

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系

如有问题或建议，请通过以下方式联系：
- 创建Issue
- 发送邮件

---

**注意**: 请确保在运行程序前备份重要数据。