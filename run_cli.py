# -*- coding: utf-8 -*-
"""
CLI入口脚本 - 转债数据处理系统命令行工具

使用方法:
python run_cli.py --history-file "path/to/history.xlsx" --new-data "path/to/new_data.xlsx" --output-dir "path/to/output"

或者使用简短参数:
python run_cli.py -h "path/to/history.xlsx" -n "path/to/new_data.xlsx" -o "path/to/output"
"""

import argparse
import os
import sys
from datetime import datetime

# 添加py目录到路径以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'py'))

# 导入所有处理函数
try:
    import importlib.util
    
    # 动态导入模块 (文件名包含数字)
    def import_module_from_file(file_path, module_name):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块 {module_name} 从文件 {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    # 导入各个处理模块
    py_dir = os.path.join(os.path.dirname(__file__), 'py')
    data_cleaning = import_module_from_file(os.path.join(py_dir, '1_data_cleaning.py'), 'data_cleaning')
    data_merging = import_module_from_file(os.path.join(py_dir, '2_data_merging.py'), 'data_merging')
    metrics_calculation = import_module_from_file(os.path.join(py_dir, '3_metrics_calculation.py'), 'metrics_calculation')
    data_analysis = import_module_from_file(os.path.join(py_dir, '4_data_analysis.py'), 'data_analysis')
    
    # 获取函数引用
    run_data_cleaning = data_cleaning.run_data_cleaning
    run_data_merging = data_merging.run_data_merging
    run_metrics_calculation = metrics_calculation.run_metrics_calculation
    run_data_analysis = data_analysis.run_data_analysis
    
except ImportError as e:
    print(f"错误: 无法导入处理模块 - {e}")
    print("请确保py/目录存在且包含所有必需的Python文件")
    sys.exit(1)
except Exception as e:
    print(f"错误: 导入模块时发生问题 - {e}")
    print("请检查py/目录中的Python文件是否完整")
    sys.exit(1)

def validate_file_exists(file_path, description):
    """验证文件是否存在"""
    if not os.path.exists(file_path):
        print(f"错误: {description}文件不存在: {file_path}")
        return False
    return True

def create_output_files(output_dir, timestamp=None):
    """创建输出文件路径"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    return {
        'step1_output': os.path.join(output_dir, f'1_cleaned_data_{timestamp}.xlsx'),
        'step2_merged': os.path.join(output_dir, f'2_merged_temp_{timestamp}.xlsx'),
        'step2_output': os.path.join(output_dir, f'2_merged_data_{timestamp}.xlsx'),
        'step3_output': os.path.join(output_dir, f'3_calculated_metrics_{timestamp}.xlsx'),
        'step4_output': os.path.join(output_dir, f'4_analysis_report_{timestamp}.xlsx')
    }

def run_step1(new_data_file, output_files):
    """执行步骤1: 数据清洗"""
    print("\n" + "="*60)
    print("正在执行步骤1: 数据清洗...")
    print("="*60)
    
    success = run_data_cleaning(new_data_file, output_files['step1_output'])
    
    if success:
        print("✅ 步骤1: 数据清洗 - 完成")
        return True
    else:
        print("❌ 步骤1: 数据清洗 - 失败")
        return False

def run_step2(history_file, output_files):
    """执行步骤2: 数据合并"""
    print("\n" + "="*60)
    print("正在执行步骤2: 数据合并...")
    print("="*60)
    
    success = run_data_merging(
        history_file, 
        output_files['step1_output'],
        output_files['step2_merged'],
        output_files['step2_output']
    )
    
    if success:
        print("✅ 步骤2: 数据合并 - 完成")
        return True
    else:
        print("❌ 步骤2: 数据合并 - 失败")
        return False

def run_step3(output_files):
    """执行步骤3: 指标计算"""
    print("\n" + "="*60)
    print("正在执行步骤3: 指标计算...")
    print("="*60)
    
    success = run_metrics_calculation(
        output_files['step2_output'], 
        output_files['step3_output']
    )
    
    if success:
        print("✅ 步骤3: 指标计算 - 完成")
        return True
    else:
        print("❌ 步骤3: 指标计算 - 失败")
        return False

def run_step4(output_files):
    """执行步骤4: 数据分析"""
    print("\n" + "="*60)
    print("正在执行步骤4: 数据分析...")
    print("="*60)
    
    success = run_data_analysis(
        output_files['step3_output'], 
        output_files['step4_output']
    )
    
    if success:
        print("✅ 步骤4: 数据分析 - 完成")
        return True
    else:
        print("❌ 步骤4: 数据分析 - 失败")
        return False

def cleanup_temp_files(output_files):
    """清理临时文件"""
    temp_files = [output_files['step2_merged']]  # 临时合并文件
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"已清理临时文件: {temp_file}")
        except Exception as e:
            print(f"清理临时文件失败 {temp_file}: {e}")

def main():
    """主函数 - 解析命令行参数并执行处理流程"""
    parser = argparse.ArgumentParser(
        description='转债数据处理系统命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_cli.py --history-file "data/history.xlsx" --new-data "data/new.xlsx" --output-dir "output"
  python run_cli.py -h "data/history.xlsx" -n "data/new.xlsx" -o "output" --keep-temp
        """
    )
    
    # 必需参数
    parser.add_argument(
        '--history-file', '-f',
        required=True,
        help='历史数据Excel文件路径 (必需)'
    )
    
    parser.add_argument(
        '--new-data', '-n', 
        required=True,
        help='新数据Excel文件路径 (必需)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='输出目录路径 (必需)'
    )
    
    # 可选参数
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='保留临时文件 (默认: 删除临时文件)'
    )
    
    parser.add_argument(
        '--timestamp',
        help='自定义时间戳 (格式: YYYYMMDD_HHMMSS, 默认: 当前时间)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='转债数据处理系统 CLI v1.0.0'
    )
    
    # 解析参数
    try:
        args = parser.parse_args()
    except SystemExit:
        return 1
    
    # 开始处理
    print("="*80)
    print("转债数据处理系统 - 命令行工具")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"历史数据文件: {args.history_file}")
    print(f"新数据文件: {args.new_data}")
    print(f"输出目录: {args.output_dir}")
    
    # 验证输入文件
    if not validate_file_exists(args.history_file, "历史数据"):
        return 1
    if not validate_file_exists(args.new_data, "新数据"):
        return 1
    
    # 创建输出文件路径
    try:
        output_files = create_output_files(args.output_dir, args.timestamp)
        print(f"输出文件时间戳: {args.timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')}")
    except Exception as e:
        print(f"错误: 创建输出目录失败 - {e}")
        return 1
    
    # 执行处理流程
    steps_completed = 0
    total_steps = 4
    
    try:
        # 步骤1: 数据清洗
        if run_step1(args.new_data, output_files):
            steps_completed += 1
        else:
            return 1
        
        # 步骤2: 数据合并  
        if run_step2(args.history_file, output_files):
            steps_completed += 1
        else:
            return 1
        
        # 步骤3: 指标计算
        if run_step3(output_files):
            steps_completed += 1
        else:
            return 1
        
        # 步骤4: 数据分析
        if run_step4(output_files):
            steps_completed += 1
        else:
            return 1
        
        # 清理临时文件
        if not args.keep_temp:
            cleanup_temp_files(output_files)
        
        # 成功完成
        print("\n" + "="*80)
        print("🎉 所有处理步骤完成！")
        print("="*80)
        print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"已完成步骤: {steps_completed}/{total_steps}")
        print("\n生成的文件:")
        for key, path in output_files.items():
            if key != 'step2_merged' and os.path.exists(path):  # 跳过临时文件
                file_size = os.path.getsize(path) / (1024*1024)  # MB
                print(f"  {path} ({file_size:.1f} MB)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理...")
        return 1
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        print(f"已完成步骤: {steps_completed}/{total_steps}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)