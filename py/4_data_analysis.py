# -*- coding: utf-8 -*-
"""
第四步 - 数据分析与指标计算
计算五个关键参数：2018/2021窗口分位点、本周变化额/率、本周收盘值
"""

import logging
import sys
import pandas as pd
import numpy as np
import datetime
import os
from tqdm import tqdm
from scipy.stats import percentileofscore
import config

def setup_logging(log_path):
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def _calculate_key_metrics(df, change_period_days, trading_days_since_2018, trading_days_since_2021, result_labels):
    """计算每列最后一行数据的五个关键参数"""
    if df.empty:
        logging.warning("数据为空，无法进行计算。")
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
        logging.warning("没有找到有效的数值列，无法计算分析指标。")
        return pd.DataFrame(index=list(result_labels.values()))

    logging.info(f"将为以下 {len(valid_columns)} 列计算关键参数: {valid_columns[:5]}{'...' if len(valid_columns) > 5 else ''}")
    
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



def run_analysis(input_data_path, output_path, change_period_days, trading_days_since_2018, trading_days_since_2021, result_labels):
    """主分析流程：读取数据，计算关键参数，输出七列长表格"""
    logging.info("=== 数据分析模块启动 ===")
    logging.info(f"输入文件: {input_data_path}")
    logging.info(f"输出文件: {output_path}")

    try:
        # 读取Excel文件
        excel_file = pd.ExcelFile(input_data_path)
        sheet_names = excel_file.sheet_names
        logging.info(f"发现 {len(sheet_names)} 个工作表: {sheet_names[:3]}{'...' if len(sheet_names) > 3 else ''}")

    except FileNotFoundError as e:
        logging.error(f"输入文件未找到: {e}")
        raise
    except Exception as e:
        logging.error(f"读取Excel文件时发生错误: {e}")
        raise

    processed_count = 0
    all_results = []  # 存储所有结果的列表
    
    # 使用tqdm显示进度
    for sheet_name in tqdm(sheet_names, desc="分析工作表"):
        try:
            # 读取工作表数据
            df = excel_file.parse(sheet_name, index_col=0)
            
            if df.empty:
                logging.warning(f"工作表 '{sheet_name}' 为空，跳过处理。")
                continue
            
            logging.info(f"处理工作表 '{sheet_name}': {df.shape[0]} 行 × {df.shape[1]} 列")
            
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
                
                logging.info(f"工作表 '{sheet_name}' 处理完成，关键参数已计算")
            else:
                logging.warning(f"工作表 '{sheet_name}' 无法计算关键参数")
            
            processed_count += 1
            
        except Exception as e:
            logging.error(f"处理工作表 '{sheet_name}' 时发生错误: {e}", exc_info=True)
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
        
        logging.info(f"成功处理 {processed_count} 个工作表，共生成 {len(all_results)} 行数据")
    else:
        logging.warning("未生成任何有效数据")
    
    logging.info(f"=== 数据分析完成 === 结果已保存至: {output_path}")

def main():
    """主函数，加载配置并执行分析"""
    log_file_path = os.path.join(config.BASE_DIR, 'run_log_4_analysis.txt')
    setup_logging(log_file_path)

    try:
        # 动态获取配置，确保交易日参数基于实际数据计算
        fp_config = config.get_step4_analysis_config()
        
        # 从配置中获取参数
        analysis_params = fp_config['analysis_parameters']
        
        logging.info(f"使用动态计算的交易日参数:")
        logging.info(f"  2018年至今交易日数: {analysis_params['trading_days_since_2018']}")
        logging.info(f"  2021年至今交易日数: {analysis_params['trading_days_since_2021']}")
        
        run_analysis(
            input_data_path=fp_config['file_paths']['input_excel_for_analysis'],
            output_path=fp_config['file_paths']['output_excel_analysis'],
            change_period_days=analysis_params['change_period_days'],
            trading_days_since_2018=analysis_params['trading_days_since_2018'],
            trading_days_since_2021=analysis_params['trading_days_since_2021'],
            result_labels=fp_config['result_labels']
        )
        
        print(f"数据分析完成！")
        print(f"输出文件: {fp_config['file_paths']['output_excel_analysis']}")
        
    except Exception as e:
        logging.error(f"数据分析过程中发生错误: {e}", exc_info=True)
        print(f"数据分析失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()