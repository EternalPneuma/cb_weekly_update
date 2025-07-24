# -*- coding: utf-8 -*-
"""
主执行文件，用于按顺序调用数据处理流程中的各个脚本。
自动使用config.py中的配置运行四个处理步骤。
"""
import subprocess
import sys
import os
from datetime import datetime
import config

def check_input_files():
    """检查输入文件是否存在"""
    missing_files = []
    
    if not os.path.exists(config.STEP1_INPUT_FILE):
        missing_files.append(f"步骤1输入文件: {config.STEP1_INPUT_FILE}")
    
    if not os.path.exists(config.STEP2_INPUT_FILE_A):
        missing_files.append(f"步骤2输入文件A: {config.STEP2_INPUT_FILE_A}")
    
    if missing_files:
        print("错误：以下输入文件不存在：")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True

def run_script(script_name):
    """运行指定的Python脚本并实时显示输出。"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print("\n" + "="*20 + f"  开始运行: {script_name}  " + "="*20)
    try:
        # 使用Popen启动子进程
        # 注意：为了避免编码问题，最好不在这里指定encoding，让communicate处理原始字节流
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # 让communicate返回文本而不是字节
            encoding='gbk', # 指定正确的编码
            errors='replace' # 如果遇到无法解码的字符，用'?'替换
        )

        # 使用 communicate() 来安全地获取所有输出，避免死锁
        # communicate 会读取所有stdout和stderr数据，然后等待进程结束
        stdout_output, stderr_output = process.communicate()

        # 实时打印获取到的输出
        if stdout_output:
            print("--- 标准输出 ---")
            print(stdout_output)
        
        # 检查返回码
        if process.returncode != 0:
            if stderr_output:
                print("--- 标准错误 ---", file=sys.stderr)
                print(stderr_output, file=sys.stderr)
            print("="*20 + f"  {script_name} 运行失败 (返回码: {process.returncode})  " + "="*20)
            return False

        print("\n" + "="*20 + f"  {script_name} 运行成功  " + "="*20)
        return True

    except FileNotFoundError:
        print(f"--- 错误: 找不到脚本 {script_path} ---")
        return False
    except Exception as e:
        print(f"--- 运行 {script_name} 时发生意外错误: {e} ---")
        return False


def main():
    """主函数，按顺序执行所有数据处理脚本。"""
    try:
        print("\n" + "="*60)
        print("         转债数据处理系统")
        print("="*60)
        
        # 检查输入文件是否存在
        if not check_input_files():
            print("\n请确保输入文件存在后重新运行程序。")
            return
        
        # 显示配置信息
        print(f"\n配置信息:")
        print(f"  步骤1输入文件: {config.STEP1_INPUT_FILE}")
        print(f"  步骤2输入文件A: {config.STEP2_INPUT_FILE_A}")
        print(f"  输出目录: {config.PROCESS_DIR}")
        
        print("\n" + "="*60)
        print("         开始执行数据处理流程")
        print("="*60)
        
        scripts_to_run = [
            "1_data_cleaning.py",
            "2_data_merging.py",
            "3_metrics_calculation.py",
            "4_data_analysis.py"
        ]
        
        success_count = 0
        for i, script in enumerate(scripts_to_run, 1):
            print(f"\n[{i}/{len(scripts_to_run)}] 正在执行: {script}")
            if run_script(script):
                success_count += 1
            else:
                print(f"\n{'!'*20} 由于 {script} 运行失败，整个流程已中止。 {'!'*20}")
                break
        
        if success_count == len(scripts_to_run):
            print(f"\n{'*'*20} 所有脚本均已成功运行 {'*'*20}")
            print(f"\n输出文件保存在: {config.PROCESS_DIR}")
            print(f"\n生成的文件:")
            print(f"  - 步骤1输出: {config.STEP1_OUTPUT_FILE}")
            print(f"  - 步骤2输出: {config.STEP2_OUTPUT_FILE}")
            print(f"  - 步骤3输出: {config.STEP3_OUTPUT_FILE}")
            print(f"  - 步骤4输出: {config.STEP4_OUTPUT_FILE}")
        else:
            print(f"\n完成了 {success_count}/{len(scripts_to_run)} 个步骤")
            
    except KeyboardInterrupt:
        print("\n\n用户中断了程序执行。")
    except Exception as e:
        print(f"\n\n程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 在脚本末尾添加，以便在从IDE运行时也能看到最终结果
    if sys.stdout.isatty():
        input("\n按回车键退出...")

if __name__ == "__main__":
    main()