# -*- coding: utf-8 -*-
"""
第四步 - 数据分析与指标计算
计算五个关键参数：2018/2021窗口分位点、本周变化额/率、本周收盘值
"""

import sys
import pandas as pd
import numpy as np
import datetime
import os
from tqdm import tqdm
from scipy.stats import percentileofscore

# 导入config模块 - 处理相对路径问题
try:
    import config
except ImportError:
    # 如果直接导入失败，尝试从当前目录导入
    sys.path.append(os.path.dirname(__file__))
    import config

def _calculate_key_metrics(df, change_period_days, trading_days_since_2018, trading_days_since_2021, result_labels):
    """计算每列最后一行数据的五个关键参数"""
    if df.empty:
        return pd.DataFrame(index=list(result_labels.values()))

    # 获取最后一行数据
    last_row = df.iloc[-1]
    
    # 筛选数值列
    valid_columns = []
    for col in df.columns:
        try:
            # 尝试将整列转换为数值
            pd.to_numeric(df[col], errors='raise')
            valid_columns.append(col)
        except (ValueError, TypeError):
            continue
    
    if not valid_columns:
        return pd.DataFrame(index=list(result_labels.values()))
    
    # 初始化结果字典
    analysis_results = {label: {} for label in result_labels.values()}

    for col_name in valid_columns:
        # 转换为数值序列
        series = pd.to_numeric(df[col_name], errors='coerce')
        current_value = series.iloc[-1]
        
        # 初始化所有指标为NaN
        for label in result_labels.values():
            analysis_results[label][col_name] = np.nan

        # 1. 计算2018年至今的百分位数
        window_2018 = series.iloc[-trading_days_since_2018:] if len(series) >= trading_days_since_2018 else series
        cleaned_window_2018 = window_2018.dropna()
        if len(cleaned_window_2018) > 0:
            analysis_results[result_labels["quantile_a"]][col_name] = percentileofscore(cleaned_window_2018, current_value, kind='rank') / 100.0

        # 2. 计算2021年至今的百分位数
        window_2021 = series.iloc[-trading_days_since_2021:] if len(series) >= trading_days_since_2021 else series
        cleaned_window_2021 = window_2021.dropna()
        if len(cleaned_window_2021) > 0:
            analysis_results[result_labels["quantile_b"]][col_name] = percentileofscore(cleaned_window_2021, current_value, kind='rank') / 100.0

        # 3. 计算与倒数第六行的变化额和变化率
        if len(series) >= change_period_days + 1:
            previous_value = series.iloc[-(change_period_days + 1)]
            if pd.notna(previous_value) and pd.notna(current_value):
                # 变化额
                difference = current_value - previous_value
                analysis_results[result_labels["difference"]][col_name] = difference
                
                # 变化率
                if previous_value != 0:
                    analysis_results[result_labels["change_rate"]][col_name] = difference / previous_value
                else:
                    analysis_results[result_labels["change_rate"]][col_name] = np.inf if difference != 0 else 0

        # 4. 记录本周收盘值
        analysis_results[result_labels["closing_value"]][col_name] = current_value

    # 转换为DataFrame
    analysis_df = pd.DataFrame.from_dict(analysis_results, orient='index', columns=valid_columns)
    analysis_df = analysis_df.reindex(index=list(result_labels.values()))

    return analysis_df



def _run_analysis(input_data_path, output_path, change_period_days, trading_days_since_2018, trading_days_since_2021, result_labels):
    """主分析流程：读取数据，计算关键参数，输出七列长表格"""

    try:
        # 读取Excel文件
        excel_file = pd.ExcelFile(input_data_path)
        sheet_names = excel_file.sheet_names

    except FileNotFoundError as e:
        raise
    except Exception as e:
        raise

    processed_count = 0
    all_results = []  # 存储所有结果的列表
    
    # 使用tqdm显示进度
    for sheet_name in tqdm(sheet_names, desc="分析工作表"):
        try:
            # 读取工作表数据
            df = excel_file.parse(sheet_name, index_col=0)
            
            if df.empty:
                continue
            
            # 计算关键参数
            analysis_df = _calculate_key_metrics(
                df, change_period_days, trading_days_since_2018, 
                trading_days_since_2021, result_labels
            )
            
            if not analysis_df.empty:
                # 将结果转换为长表格格式
                for param_name in analysis_df.columns:
                    row_data = {
                        '工作表名': sheet_name,
                        '参数名': param_name,
                        result_labels['quantile_a']: analysis_df.loc[result_labels['quantile_a'], param_name],
                        result_labels['quantile_b']: analysis_df.loc[result_labels['quantile_b'], param_name],
                        result_labels['difference']: analysis_df.loc[result_labels['difference'], param_name],
                        result_labels['change_rate']: analysis_df.loc[result_labels['change_rate'], param_name],
                        result_labels['closing_value']: analysis_df.loc[result_labels['closing_value'], param_name]
                    }
                    all_results.append(row_data)
                
                pass
            
            processed_count += 1
            
        except Exception as e:
            continue

    # 将所有结果合并为一个DataFrame并保存
    if all_results:
        final_df = pd.DataFrame(all_results)
        # 重新排列列的顺序
        column_order = ['工作表名', '参数名', result_labels['quantile_a'], result_labels['quantile_b'], 
                       result_labels['difference'], result_labels['change_rate'], result_labels['closing_value']]
        final_df = final_df[column_order]
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            final_df.to_excel(writer, sheet_name='关键数据汇总', index=False)

def calculate_trading_days_from_input_file(input_file_path):
    """根据输入文件动态计算交易日参数"""
    try:
        # 读取第一个工作表来获取数据行数
        excel_file = pd.ExcelFile(input_file_path)
        first_sheet = excel_file.sheet_names[0]
        df = pd.read_excel(input_file_path, sheet_name=first_sheet, index_col=0)
        
        total_rows = len(df)
        
        # 2018年第一天是第2行，2021年第一天是第732行
        # 计算从2018年至今的交易日数（从第2行到最后一行）
        trading_days_since_2018 = total_rows - 1  # 减1因为第2行对应索引1
        
        # 计算从2021年至今的交易日数（从第732行到最后一行）
        trading_days_since_2021 = total_rows - 731  # 减731因为第732行对应索引731
        
        print(f"动态计算的交易日参数:")
        print(f"  数据总行数: {total_rows}")
        print(f"  2018年至今交易日数: {trading_days_since_2018}")
        print(f"  2021年至今交易日数: {trading_days_since_2021}")
        
        return trading_days_since_2018, trading_days_since_2021
    except Exception as e:
        print(f"计算交易日参数时出错: {e}，使用默认值")
        # 如果计算失败，使用默认值
        return 1825, 1094

def run_data_analysis(input_path, output_path):
    """
    执行数据分析步骤
    
    Args:
        input_path (str): 输入Excel文件路径
        output_path (str): 输出Excel文件路径
    
    Returns:
        bool: True表示成功，False表示失败
    """
    try:
        print("开始执行数据分析...")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"错误：输入文件不存在 - {input_path}")
            return False
        
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        
        # 获取配置
        fp_config = config.get_step4_analysis_config()
        
        # 动态计算交易日参数
        print("正在计算交易日参数...")
        trading_days_since_2018, trading_days_since_2021 = calculate_trading_days_from_input_file(input_path)
        
        # 执行分析
        print("正在执行数据分析...")
        _run_analysis(
            input_data_path=input_path,
            output_path=output_path,
            change_period_days=fp_config['analysis_parameters']['change_period_days'],
            trading_days_since_2018=trading_days_since_2018,
            trading_days_since_2021=trading_days_since_2021,
            result_labels=fp_config['result_labels']
        )
        
        print(f"数据分析完成！")
        print(f"输出文件: {output_path}")
        return True
        
    except Exception as e:
        print(f"数据分析失败: {e}")
        return False

def main():
    """主函数，加载配置并执行分析"""
    try:
        # 获取配置
        fp_config = config.get_step4_analysis_config()
        
        # 调用新的模块化函数
        success = run_data_analysis(
            input_path=fp_config['file_paths']['input_excel_for_analysis'],
            output_path=fp_config['file_paths']['output_excel_analysis']
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"数据分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()