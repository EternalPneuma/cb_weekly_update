# -*- coding: utf-8 -*-
"""
配置文件 - 转债数据处理系统
生成时间: 2025-07-24 14:37:19
"""

import os
from datetime import datetime

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录
DATA_ROOT = os.path.join(PROJECT_ROOT, 'date')  # 输入目录：根目录下的date文件夹
PROCESS_DIR = os.path.join(PROJECT_ROOT, 'output')  # 输出目录：根目录下的output文件夹
OUTPUT_DIR = PROCESS_DIR  # 统一输出目录
INPUT_DATA_DIR = DATA_ROOT

# 确保输出目录存在
os.makedirs(PROCESS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 文件路径配置 - 使用Windows兼容的路径格式
STEP1_INPUT_FILE = os.path.join(DATA_ROOT, "1.1 日度数据-清洗前转债基础数据-813支-20250711-20250718-无链接.xlsx")
STEP2_INPUT_FILE_A = os.path.join(DATA_ROOT, "2.1 日度数据-清洗后转债基础数据-去重-20180102-20250711(1).xlsx")

# 输出文件路径
today = datetime.now().strftime('%Y%m%d')
STEP1_OUTPUT_FILE = os.path.join(PROCESS_DIR, f'1_cleaned_data_{today}.xlsx')
STEP2_OUTPUT_FILE = os.path.join(PROCESS_DIR, f'2_merged_data_{today}.xlsx')
STEP3_OUTPUT_FILE = os.path.join(PROCESS_DIR, f'3_calculated_metrics_{today}.xlsx')
STEP4_OUTPUT_FILE = os.path.join(PROCESS_DIR, f'4_analysis_report_{today}.xlsx')

# 步骤2需要的额外配置
STEP2_INPUT_A = STEP2_INPUT_FILE_A  # 历史数据文件
STEP2_INPUT_B = STEP1_OUTPUT_FILE   # 步骤1的输出作为步骤2的输入B
STEP2_OUTPUT_MERGED = os.path.join(PROCESS_DIR, f'2_merged_data_{today}.xlsx')
STEP2_OUTPUT_DEDUP = STEP2_OUTPUT_FILE  # 去重后的最终输出

# 步骤3需要的配置
STEP3_INPUT_FILE = STEP2_OUTPUT_FILE  # 步骤2的输出作为步骤3的输入

# 分析参数配置
CHANGE_PERIOD_DAYS = 5
TRADING_DAYS_SINCE_2018 = 1825
TRADING_DAYS_SINCE_2021 = 1094

def calculate_trading_days_from_data(data_file_path):
    """
    根据数据文件的行数动态计算交易日数
    """
    try:
        import pandas as pd
        # 尝试读取Excel文件的第一个工作表
        df = pd.read_excel(data_file_path, sheet_name=0, nrows=0)  # 只读取列名
        df_full = pd.read_excel(data_file_path, sheet_name=0)  # 读取完整数据
        
        # 计算行数（减去表头）
        total_rows = len(df_full) - 1 if len(df_full) > 0 else 0
        
        # 根据行数估算交易日数
        # 假设每年约250个交易日
        years_since_2018 = (datetime.now().year - 2018) + (datetime.now().month / 12.0)
        years_since_2021 = (datetime.now().year - 2021) + (datetime.now().month / 12.0)
        
        estimated_2018 = int(years_since_2018 * 250)
        estimated_2021 = int(years_since_2021 * 250)
        
        return estimated_2018, estimated_2021
    except Exception as e:
        print(f"计算交易日数时出错: {e}")
        return TRADING_DAYS_SINCE_2018, TRADING_DAYS_SINCE_2021

# 分析参数
ANALYSIS_PARAMS = {
    'change_period_days': CHANGE_PERIOD_DAYS,
    'trading_days_since_2018': TRADING_DAYS_SINCE_2018,
    'trading_days_since_2021': TRADING_DAYS_SINCE_2021
}

def get_step4_analysis_config():
    """
    获取步骤4的分析配置
    """
    return {
        'file_paths': {
            'input_excel_for_analysis': STEP3_OUTPUT_FILE,
            'output_excel_analysis': STEP4_OUTPUT_FILE
        },
        'analysis_parameters': {
            'change_period_days': CHANGE_PERIOD_DAYS,
            'trading_days_since_2018': TRADING_DAYS_SINCE_2018,
            'trading_days_since_2021': TRADING_DAYS_SINCE_2021
        },
        'result_labels': {
            'quantile_a': '2018年至今分位数',
            'quantile_b': '2021年至今分位数',
            'difference': '本周变化额',
            'change_rate': '本周变化率',
            'closing_value': '本周收盘值'
        }
    }