# -*- coding: utf-8 -*-
"""
步骤2：数据合并

1.  合并两个Excel文件（历史数据和本期新数据）。
2.  对合并后的数据按日期进行去重，保留最新的记录。
3.  将最终结果保存到新的Excel文件中。
"""
import pandas as pd
import os
import sys
import config
from tqdm import tqdm
import concurrent.futures
import threading

def run_data_merging(file_a, file_b, output_file, dedup_file, log_callback):
    """
    执行数据合并步骤
    
    参数:
        file_a (str): 输入文件A路径（历史数据）
        file_b (str): 输入文件B路径（新数据）
        output_file (str): 合并后输出文件路径
        dedup_file (str): 去重后输出文件路径
        log_callback (function): 日志回调函数
    
    返回:
        bool: 成功返回True，失败返回False
    """
    try:
        MAX_WORKERS = os.cpu_count() or 1  # 设置最大进程数
        
        # --- 内部函数定义 --- #
        def merge_sheet(sheet_name):
            """读取并合并单个工作表"""
            try:
                df_a = pd.read_excel(file_a, sheet_name=sheet_name)
                if sheet_name == '8、2017Q3后存续的813支转债标的':
                    df_b = pd.read_excel(file_b, sheet_name=sheet_name)
                    return sheet_name, df_b
                
                df_b = pd.read_excel(file_b, sheet_name=sheet_name)
                
                combined = pd.concat([df_a, df_b], ignore_index=True)
                
                b_cols = df_b.columns.tolist()
                a_cols = df_a.columns.tolist()
                final_cols = b_cols + [col for col in a_cols if col not in b_cols]
                
                return sheet_name, combined[final_cols]
            except ValueError:
                log_callback(f"工作表 '{sheet_name}' 在文件B中不存在，将只使用文件A的数据。")
                df_a = pd.read_excel(file_a, sheet_name=sheet_name)
                return sheet_name, df_a
            except Exception as e:
                log_callback(f"处理工作表 '{sheet_name}' 时出错: {e}")
                return sheet_name, None
        
        def dedup_sheet(sheet_data):
            """对单个工作表数据进行去重"""
            sheet_name, df = sheet_data
            if df is not None and not df.empty:
                first_col = df.columns[0]
                df_dedup = df.drop_duplicates(subset=[first_col], keep='last')
                return sheet_name, df_dedup
            return sheet_name, df  # 返回原始df（可能为None或空）

        # --- 主逻辑 --- #
        # 检查输入文件是否存在
        for f in [file_a, file_b]:
            if not os.path.exists(f):
                log_callback(f"错误：输入文件不存在 '{f}'。请检查路径配置或确保上一步已成功运行。")
                return False
        
        log_callback(f"开始合并文件:\n  A: {file_a}\n  B: {file_b}\n输出至: {output_file}")

        # 1. 并行合并文件
        merged_data = {}
        try:
            xls_a = pd.ExcelFile(file_a)
            a_sheets_names = xls_a.sheet_names
            xls_a.close()  # 关闭文件句柄
        except Exception as e:
            log_callback(f"错误：读取Excel文件 '{file_a}' 的工作表列表时出错: {e}")
            return False
        
        if not a_sheets_names:
            log_callback(f"警告：输入文件 '{file_a}' 不包含任何工作表。")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # 使用tqdm显示进度
                future_to_sheet = {executor.submit(merge_sheet, name): name for name in a_sheets_names}
                for future in tqdm(concurrent.futures.as_completed(future_to_sheet), total=len(a_sheets_names), desc="合并工作表"):
                    sheet_name, df = future.result()
                    if df is not None and not df.empty:
                        merged_data[sheet_name] = df

        # 串行写入合并后的文件
        if not merged_data:
            log_callback("警告：没有合并任何数据。跳过写入合并文件。")
        else:
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                for sheet_name, df in merged_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            log_callback(f"文件合并完成，已保存到 {output_file}")

        # 2. 并行对合并后的文件按日期去重
        log_callback(f"开始对 {output_file} 进行去重，结果保存到 {dedup_file}")
        deduped_data = {}
        if not merged_data:
            log_callback("警告：没有数据可供去重。")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_dedup = {executor.submit(dedup_sheet, item): item[0] for item in merged_data.items()}
                for future in tqdm(concurrent.futures.as_completed(future_to_dedup), total=len(merged_data), desc="去重工作表"):
                    sheet_name, df = future.result()
                    if df is not None and not df.empty:
                        deduped_data[sheet_name] = df

        # 串行写入去重后的文件
        if not deduped_data:
            log_callback("警告：去重后没有剩余数据。跳过写入去重文件。")
        else:
            # 定义工作表重命名映射
            sheet_name_map = {
                '1、转债收盘价': '1、转债收盘价',
                '收盘价': '1、转债收盘价',
                '2、纯债到期收益率': '2、转债YTM',
                '纯债到期收益率': '2、转债YTM',
                '3、纯债价值': '3、纯债价值',
                '纯债价值': '3、纯债价值',
                '4、转债余额': '4、转债余额',
                '转债余额': '4、转债余额',
                '5、纯债溢价率': '5、纯债溢价率',
                '纯债溢价率': '5、纯债溢价率',
                '6、转换价值': '6、转换价值',
                '转换价值': '6、转换价值',
                '7、转股溢价率': '7、转股溢价率',
                '转股溢价率': '7、转股溢价率',
                '8、开盘价': '8、开盘价',
                '开盘价': '8、开盘价',
                '9、平底溢价率': '9、债股性',
                '平底溢价率': '9、债股性',
                '10、债项评级': '10、债项评级',
                '债项评级': '10、债项评级',
                '12、成交额': '12、成交额',
                '成交额': '12、成交额',
                '13、正股收盘价': '13、正股收盘价',
                '正股收盘价': '13、正股收盘价',
                '14、60日均线': '14、60日均线',
                '60日均线': '14、60日均线',
                '15、120日均线': '15、120日均线',
                '120日均线': '15、120日均线',
                '转债数量': '转债数量',
                '8、2017Q3后存续的813支转债标的': '8、2017Q3后存续的813支转债标的'
            }
            
            renamed_data = {}
            for old_name, df in deduped_data.items():
                new_name = sheet_name_map.get(old_name, old_name)
                renamed_data[new_name] = df
            
            with pd.ExcelWriter(dedup_file, engine='xlsxwriter') as writer:
                for sheet_name, df in renamed_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            log_callback(f"去重完成，最终数据已保存到 {dedup_file}")
        
        return True
        
    except Exception as e:
        log_callback(f"数据合并过程中发生错误: {str(e)}")
        log_callback(f"错误类型: {type(e).__name__}")
        return False

if __name__ == '__main__':
    # 在Windows上，需要为多进程设置启动方法
    # 'spawn' a new process rather than 'fork' is safer
    multiprocessing.set_start_method('spawn', force=True)
    
    # 独立测试运行
    def print_log(message):
        print(message)
    
    file_a = config.STEP2_INPUT_A
    file_b = config.STEP2_INPUT_B
    output_file = config.STEP2_OUTPUT_MERGED
    dedup_file = config.STEP2_OUTPUT_DEDUP
    
    success = run_data_merging(file_a, file_b, output_file, dedup_file, print_log)
    if not success:
        sys.exit(1)
