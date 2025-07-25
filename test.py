# -*- coding: utf-8 -*-
"""
测试脚本 - 验证模块化后的函数功能

此脚本用于测试所有模块化的函数，确保它们按正确顺序运行并产生预期的输出。
"""

import os
import sys
import tempfile
from datetime import datetime

# 导入所有模块化的函数
sys.path.append(os.path.join(os.path.dirname(__file__), 'py'))

# 使用动态导入来避免文件名问题
import importlib.util

def import_module_from_file(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 导入各个处理模块
py_dir = os.path.join(os.path.dirname(__file__), 'py')
data_cleaning = import_module_from_file(os.path.join(py_dir, '1_data_cleaning.py'), 'data_cleaning')
data_merging = import_module_from_file(os.path.join(py_dir, '2_data_merging.py'), 'data_merging')
metrics_calculation = import_module_from_file(os.path.join(py_dir, '3_metrics_calculation.py'), 'metrics_calculation')
data_analysis = import_module_from_file(os.path.join(py_dir, '4_data_analysis.py'), 'data_analysis')
config = import_module_from_file(os.path.join(py_dir, 'config.py'), 'config')

# 获取函数引用
run_data_cleaning = data_cleaning.run_data_cleaning
run_data_merging = data_merging.run_data_merging
run_metrics_calculation = metrics_calculation.run_metrics_calculation
run_data_analysis = data_analysis.run_data_analysis

def create_test_paths():
    """创建测试用的文件路径"""
    # 创建测试输出目录
    test_output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 生成唯一的时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 测试文件路径
    test_paths = {
        'step1_output': os.path.join(test_output_dir, f'test_1_cleaned_data_{timestamp}.xlsx'),
        'step2_merged': os.path.join(test_output_dir, f'test_2_merged_data_{timestamp}.xlsx'),
        'step2_dedup': os.path.join(test_output_dir, f'test_2_dedup_data_{timestamp}.xlsx'),
        'step3_output': os.path.join(test_output_dir, f'test_3_metrics_data_{timestamp}.xlsx'),
        'step4_output': os.path.join(test_output_dir, f'test_4_analysis_data_{timestamp}.xlsx')
    }
    
    return test_paths

def test_step1_data_cleaning(test_paths):
    """测试步骤1: 数据清洗"""
    print("\n=== 测试步骤1: 数据清洗 ===")
    
    # 使用配置中的输入文件
    input_file = config.STEP1_INPUT_FILE
    output_file = test_paths['step1_output']
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    # 执行数据清洗
    success = run_data_cleaning(input_file, output_file)
    
    if success and os.path.exists(output_file):
        print("✅ 步骤1测试成功")
        return True
    else:
        print("❌ 步骤1测试失败")
        return False

def test_step2_data_merging(test_paths):
    """测试步骤2: 数据合并"""
    print("\n=== 测试步骤2: 数据合并 ===")
    
    # 使用历史数据文件和步骤1的输出
    file_a = config.STEP2_INPUT_A  # 历史数据
    file_b = test_paths['step1_output']  # 步骤1的输出
    output_merged = test_paths['step2_merged']
    output_dedup = test_paths['step2_dedup']
    
    print(f"历史数据文件: {file_a}")
    print(f"新数据文件: {file_b}")
    print(f"合并文件: {output_merged}")
    print(f"去重文件: {output_dedup}")
    
    # 检查输入文件是否存在
    if not os.path.exists(file_a):
        print(f"❌ 历史数据文件不存在: {file_a}")
        return False
    if not os.path.exists(file_b):
        print(f"❌ 新数据文件不存在: {file_b}")
        return False
    
    # 执行数据合并
    success = run_data_merging(file_a, file_b, output_merged, output_dedup)
    
    if success and os.path.exists(output_dedup):
        print("✅ 步骤2测试成功")
        return True
    else:
        print("❌ 步骤2测试失败")
        return False

def test_step3_metrics_calculation(test_paths):
    """测试步骤3: 指标计算"""
    print("\n=== 测试步骤3: 指标计算 ===")
    
    input_file = test_paths['step2_dedup']  # 步骤2的输出
    output_file = test_paths['step3_output']
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    # 执行指标计算
    success = run_metrics_calculation(input_file, output_file)
    
    if success and os.path.exists(output_file):
        print("✅ 步骤3测试成功")
        return True
    else:
        print("❌ 步骤3测试失败")
        return False

def test_step4_data_analysis(test_paths):
    """测试步骤4: 数据分析"""
    print("\n=== 测试步骤4: 数据分析 ===")
    
    input_file = test_paths['step3_output']  # 步骤3的输出
    output_file = test_paths['step4_output']
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    # 执行数据分析
    success = run_data_analysis(input_file, output_file)
    
    if success and os.path.exists(output_file):
        print("✅ 步骤4测试成功")
        return True
    else:
        print("❌ 步骤4测试失败")
        return False

def main():
    """主测试函数"""
    print("开始测试模块化后的函数...")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建测试路径
    test_paths = create_test_paths()
    
    # 依次执行各步骤的测试
    test_results = []
    
    # 测试步骤1
    result1 = test_step1_data_cleaning(test_paths)
    test_results.append(("步骤1: 数据清洗", result1))
    
    # 只有前一步成功才继续下一步
    if result1:
        result2 = test_step2_data_merging(test_paths)
        test_results.append(("步骤2: 数据合并", result2))
        
        if result2:
            result3 = test_step3_metrics_calculation(test_paths)
            test_results.append(("步骤3: 指标计算", result3))
            
            if result3:
                result4 = test_step4_data_analysis(test_paths)
                test_results.append(("步骤4: 数据分析", result4))
    
    # 输出测试总结
    print("\n" + "="*50)
    print("测试结果总结:")
    print("="*50)
    
    all_passed = True
    for step_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{step_name}: {status}")
        if not result:
            all_passed = False
    
    # 输出测试文件位置
    print(f"\n测试输出文件位置:")
    for key, path in test_paths.items():
        if os.path.exists(path):
            print(f"  {key}: {path}")
    
    if all_passed:
        print("\n🎉 所有测试通过！模块化成功！")
        return True
    else:
        print("\n❌ 部分测试失败，请检查问题。")
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        print(f"错误类型: {type(e).__name__}")
        sys.exit(1)