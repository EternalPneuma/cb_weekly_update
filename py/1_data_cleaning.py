# -*- coding: utf-8 -*-
"""
步骤1：数据清洗

1.  读取原始Excel文件中的多个工作表。
2.  将每个工作表的第一列设置为日期索引。
3.  根据存续日期（DATE1, DATE2）筛选掉不在存续期内的数据。
4.  计算平底溢价率和纯债溢价率。
5.  统计每日有效的转债数量。
6.  将清洗和计算后的结果输出到新的Excel文件中。
"""
import pandas as pd
import config
import os
import sys

# 设置pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("开始执行数据清洗步骤...")

input_path = config.STEP1_INPUT_FILE
output_path = config.STEP1_OUTPUT_FILE

print(f"输入文件路径: {input_path}")
print(f"输出文件路径: {output_path}")

# 检查输入文件是否存在
if not os.path.exists(input_path):
    print(f"错误：输入文件不存在 - {input_path}")
    sys.exit(1)

# 1. 读取Excel中的指定工作表
sheets_to_read = [
    '1、转债收盘价',
    '2、纯债到期收益率',
    '3、纯债价值',
    '4、转债余额',
    '6、转换价值',
    '7、转股溢价率',
    '8、开盘价',
    '8、2017Q3后存续的813支转债标的',
    '10、债项评级',
    '12、成交额',
    '正股收盘价',
    '60日均线',
    '120日均线'
]

# 使用字典推导式，高效读取所有需要的工作表
print("开始读取Excel工作表...")
try:
    # 尝试读取Excel文件，使用不同的引擎和编码方式
    dfs = {}
    for sheet in sheets_to_read:
        try:
            print(f"正在读取工作表: {sheet}")
            # 尝试使用openpyxl引擎读取
            df = pd.read_excel(input_path, sheet_name=sheet, engine='openpyxl')
            dfs[sheet] = df
            print(f"成功读取工作表: {sheet}, 形状: {df.shape}")
        except Exception as e:
            print(f"读取工作表 {sheet} 时出错: {e}")
            # 尝试使用xlrd引擎
            try:
                df = pd.read_excel(input_path, sheet_name=sheet, engine='xlrd')
                dfs[sheet] = df
                print(f"使用xlrd引擎成功读取工作表: {sheet}, 形状: {df.shape}")
            except Exception as e2:
                print(f"使用xlrd引擎读取工作表 {sheet} 也失败: {e2}")
                raise e2
except Exception as e:
    print(f"读取Excel文件时发生严重错误: {e}")
    print(f"错误类型: {type(e).__name__}")
    sys.exit(1)

print(f"成功读取 {len(dfs)} 个工作表")


# 2. 将各工作表的第一列设为索引，并统一日期格式
# 这种方式比硬编码列名更灵活
df1_raw = dfs['1、转债收盘价']
df1 = df1_raw.set_index(df1_raw.columns[0])
df1.index = pd.to_datetime(df1.index).strftime("%Y-%m-%d")

df2_raw = dfs['2、纯债到期收益率']
df2 = df2_raw.set_index(df2_raw.columns[0])
df2.index = pd.to_datetime(df2.index).strftime("%Y-%m-%d")

df3_raw = dfs['3、纯债价值']
df3 = df3_raw.set_index(df3_raw.columns[0])
df3.index = pd.to_datetime(df3.index).strftime("%Y-%m-%d")

df4_raw = dfs['4、转债余额']
df4 = df4_raw.set_index(df4_raw.columns[0])
df4.index = pd.to_datetime(df4.index).strftime("%Y-%m-%d")

df6_raw = dfs['6、转换价值']
df6 = df6_raw.set_index(df6_raw.columns[0])
df6.index = pd.to_datetime(df6.index).strftime("%Y-%m-%d")

df7_raw = dfs['7、转股溢价率']
df7 = df7_raw.set_index(df7_raw.columns[0])
df7.index = pd.to_datetime(df7.index).strftime("%Y-%m-%d")

df8_open_raw = dfs['8、开盘价']  # 重命名避免与df8冲突
df8_open = df8_open_raw.set_index(df8_open_raw.columns[0])
df8_open.index = pd.to_datetime(df8_open.index).strftime("%Y-%m-%d")

df10_raw = dfs['10、债项评级']
df10 = df10_raw.set_index(df10_raw.columns[0])
df10.index = pd.to_datetime(df10.index).strftime("%Y-%m-%d")

df12_raw = dfs['12、成交额']
df12 = df12_raw.set_index(df12_raw.columns[0])
df12.index = pd.to_datetime(df12.index).strftime("%Y-%m-%d")

df13_raw = dfs['正股收盘价']
df13 = df13_raw.set_index(df13_raw.columns[0])
df13.index = pd.to_datetime(df13.index).strftime("%Y-%m-%d")

df14_raw = dfs['60日均线']
df14 = df14_raw.set_index(df14_raw.columns[0])
df14.index = pd.to_datetime(df14.index).strftime("%Y-%m-%d")

df15_raw = dfs['120日均线']
df15 = df15_raw.set_index(df15_raw.columns[0])
df15.index = pd.to_datetime(df15.index).strftime("%Y-%m-%d")


# 3. 清洗转债存续日期数据
df8 = dfs['8、2017Q3后存续的813支转债标的'].set_index('转债代码')
df8 = df8[['DATE1', 'DATE2', '到期日', '强赎提示日']]

# 将无效日期字符串替换为None，然后转换为datetime对象
for col in ['DATE1', 'DATE2', '到期日', '强赎提示日']:
    df8[col] = df8[col].replace('1900-01-00', None)
    df8[col] = pd.to_datetime(df8[col], errors='coerce')
    df8[col] = df8[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else None)


# 4. 根据存续日期筛选各表数据
def apply_date_filter(df, bond_code, start_date, end_date):
    """将指定债券在存续期外的数据置为None"""
    if start_date and end_date:
        df.loc[(df.index < start_date) | (df.index > end_date), bond_code] = None

data_frames_to_filter = [df1, df2, df3, df4, df6, df7, df8_open, df10, df12, df13, df14, df15]

for bond_code, row in df8.iterrows():
    for df in data_frames_to_filter:
        if bond_code in df.columns:
            apply_date_filter(df, bond_code, row['DATE1'], row['DATE2'])


# 5. 计算衍生指标：平底溢价率和纯债溢价率
df9 = ((df6 / df3) - 1) * 100  # 平底溢价率
df5 = ((df1 / df3) - 1) * 100  # 纯债溢价率
df9 = df9.round(2)
df5 = df5.round(2)


# 6. 统计每日有效转债数量
dfx = pd.DataFrame(index=df1.index)
dfx['转债数量-收盘价'] = df1.count(axis=1)
dfx['转债数量-纯债到期收益率'] = df2.count(axis=1)
dfx['转债数量-纯债价值'] = df3.count(axis=1)
dfx['转债数量-转债余额'] = df4.count(axis=1)
dfx['转债数量-转换价值'] = df6.count(axis=1)
dfx['转债数量-转股溢价率'] = df7.count(axis=1)
dfx['转债数量-债项评级'] = df10.count(axis=1)
dfx['转债数量-纯债溢价率'] = df5.count(axis=1)
dfx['转债数量-平底溢价率'] = df9.count(axis=1)
dfx['转债数量-开盘价'] = df8_open.count(axis=1)
dfx['转债数量-成交额'] = df12.count(axis=1)
dfx['转债数量-正股收盘价'] = df13.count(axis=1)
dfx['转债数量-60日均线'] = df14.count(axis=1)
dfx['转债数量-120日均线'] = df15.count(axis=1)


# 7. 将结果写入新的Excel文件
with pd.ExcelWriter(output_path) as writer:
    df1.to_excel(writer, sheet_name='1、转债收盘价')
    df2.to_excel(writer, sheet_name='2、纯债到期收益率')
    df3.to_excel(writer, sheet_name='3、纯债价值')
    df4.to_excel(writer, sheet_name='4、转债余额')
    df5.to_excel(writer, sheet_name='5、纯债溢价率')
    df6.to_excel(writer, sheet_name='6、转换价值')
    df7.to_excel(writer, sheet_name='7、转股溢价率')
    df8_open.to_excel(writer, sheet_name='8、开盘价')
    df9.to_excel(writer, sheet_name='9、平底溢价率')
    df10.to_excel(writer, sheet_name='10、债项评级')
    df12.to_excel(writer, sheet_name='12、成交额')
    df13.to_excel(writer, sheet_name='13、正股收盘价')
    df14.to_excel(writer, sheet_name='14、60日均线')
    df15.to_excel(writer, sheet_name='15、120日均线')
    dfx.to_excel(writer, sheet_name='转债数量')
    df8.to_excel(writer, sheet_name='8、2017Q3后存续的813支转债标的')

print(f"数据清洗完成，结果已保存至 {output_path}")
print("步骤1：数据清洗 - 执行成功")