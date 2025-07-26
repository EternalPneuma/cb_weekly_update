import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
import sys
import traceback
from typing import Union

# 导入config模块，处理相对路径问题
try:
    import config
except ImportError:
    # 如果直接导入失败，尝试从当前目录导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config

# --- 模型定义 ---

def predict_for_row_1(row_a: pd.Series, row_b: pd.Series, t: int) -> tuple[float, float]:
    """二次多项式回归 y = β₀ + β₁x + β₂x²"""
    if len(row_a) < 6:
        return np.nan, np.nan  # 数据点不足则返回NaN
    
    x = row_a.to_numpy().reshape(-1, 1)
    y = row_b.to_numpy().reshape(-1, 1)

    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    
    y_predict = model.predict(poly.transform(np.array([[t]])))[0, 0]
    residuals = y - model.predict(x_poly)
    residuals_sum = np.sum(residuals ** 2)
    return y_predict, residuals_sum


def predict_for_row_2(row_a: pd.Series, row_b: pd.Series, t: int) -> tuple[float, float]:
    """倒数函数回归-1: y*x = β₀ + β₁x"""
    if len(row_a) < 6:
        return np.nan, np.nan

    x = row_a.to_numpy().reshape(-1, 1)
    y = row_b.to_numpy().reshape(-1, 1)

    y_transformed = y * x  # 转换目标变量
    model = LinearRegression()
    model.fit(x, y_transformed)
    
    y_x_predict = model.predict(np.array([[t]]))[0, 0]
    y_predict = y_x_predict / t  # 从 y*x 的预测值反算 y
    
    y_pred_orig = model.predict(x) / x.ravel()
    residuals = y.ravel() - y_pred_orig
    residuals_sum = np.sum(residuals ** 2)
    return y_predict, residuals_sum


def predict_for_row_3(row_a: pd.Series, row_b: pd.Series, t: int) -> tuple[float, float]:
    """倒数函数回归-2: y = β₀ + β₁(1/x)"""
    if len(row_a) < 6:
        return np.nan, np.nan

    x = row_a.to_numpy().reshape(-1, 1)
    y = row_b.to_numpy().reshape(-1, 1)

    z = 1 / x  # 转换特征
    model = LinearRegression()
    model.fit(z, y)
    
    y_predict = model.predict(np.array([[1 / t]]))[0, 0]
    residuals = y - model.predict(z)
    residuals_sum = np.sum(residuals ** 2)
    return y_predict, residuals_sum


def predict_for_row_4(row_a: pd.Series, row_b: pd.Series, t: int) -> tuple[float, float]:
    """倒数函数回归-3: y = β₀ + β₁x + β₂(1/x)"""
    if len(row_a) < 6:
        return np.nan, np.nan

    x = row_a.to_numpy().reshape(-1, 1)
    y = row_b.to_numpy().reshape(-1, 1)

    z = 1 / x
    features = np.concatenate((x, z), axis=1) # 组合特征
    model = LinearRegression()
    model.fit(features, y)
    
    y_predict = model.predict(np.array([[t, 1 / t]]))[0, 0]
    residuals = y - model.predict(features)
    residuals_sum = np.sum(residuals ** 2)
    return y_predict, residuals_sum

# --- 指标计算函数 ---

def most_10_percent_mean(row_a: pd.Series, row_b: pd.Series) -> pd.Series:
    """计算row_a中最大和最小10%的均值，并返回对应在row_b中的值的均值"""
    non_null_values = row_a.dropna()
    if len(non_null_values) == 0:
        return pd.Series([np.nan] * 5, index=['数量', '低价转债均价', '高价转债均价', '低价转债平价均值', '高价转债平价均值'])

    n = max(1, int(len(non_null_values) * 0.1))
    
    min_10_percent_mean = non_null_values.nsmallest(n).mean()
    max_10_percent_mean = non_null_values.nlargest(n).mean()
    
    min_valid_indices = non_null_values.nsmallest(n).index
    max_valid_indices = non_null_values.nlargest(n).index
    
    return pd.Series([n, min_10_percent_mean, max_10_percent_mean, row_b.reindex(min_valid_indices).mean(), row_b.reindex(max_valid_indices).mean()],
                     index=['数量', '低价转债均价', '高价转债均价', '低价转债平价均值', '高价转债平价均值'])


def conversion_premium(row_a: pd.Series, row_b: pd.Series) -> pd.Series:
    """根据row_a的特定分位数，获取row_b中对应位置的值"""
    non_null_values = row_a.dropna()
    len_non_null = len(non_null_values)
    if len_non_null == 0:
        return pd.Series([np.nan] * 10, index=['数量1', '数量2', '数量3', '数量4', '数量5', '5%溢价率', '25%溢价率', '50%溢价率', '75%溢价率', '95%溢价率'])

    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    indices_n = [max(1, int(len_non_null * q)) for q in quantiles]
    
    min_indices = [non_null_values.nsmallest(min(n, len_non_null)).index[-1] for n in indices_n]

    results = [
        indices_n[0], indices_n[1], indices_n[2], indices_n[3], indices_n[4],
        row_b.get(min_indices[0], np.nan),
        row_b.get(min_indices[1], np.nan),
        row_b.get(min_indices[2], np.nan),
        row_b.get(min_indices[3], np.nan),
        row_b.get(min_indices[4], np.nan)
    ]
    return pd.Series(results, index=['数量1', '数量2', '数量3', '数量4', '数量5', '5%溢价率', '25%溢价率', '50%溢价率', '75%溢价率', '95%溢价率'])

# 定义全局评级分类，供 rate_group 函数使用
bad = ['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C']
good = ['AAA', 'AAA-']

def rate_group(rate: str) -> Union[str, float]:
    """评级分组"""
    if pd.isna(rate):
        return np.nan
    if rate in bad:
        return '评级低于A'
    elif rate in good:
        return 'AAA'
    else:
        return rate


def balance_group(balance: float) -> Union[str, float]:
    """余额分组"""
    if pd.isna(balance):
        return np.nan
    elif balance < 3:
        return '剩余规模<3亿'
    elif balance < 10:
        return '剩余规模3-10亿'
    elif balance < 50:
        return '剩余规模10-50亿'
    else:
        return '剩余规模>50亿'


def trimmed_mean(row: pd.Series) -> float:
    """去除每行数据排序后头尾各3个极端值，再计算平均数"""
    sorted_row = np.sort(row.dropna())
    if len(sorted_row) < 7:
        return float(np.mean(sorted_row)) if len(sorted_row) > 0 else np.nan
    else:
        return float(sorted_row[3:-3].mean())


def trimmed_median(row: pd.Series) -> float:
    """计算去除NaN后的中位数"""
    sorted_row = np.sort(row.dropna())
    return float(np.median(sorted_row)) if len(sorted_row) > 0 else np.nan


def calculate_regression_metrics(data_x: pd.DataFrame, data_y: pd.DataFrame) -> pd.DataFrame:
    """
    为给定的数据集(data_x, data_y)进行回归分析。
    对每一行数据，尝试四种回归模型，选择最优模型，并计算多情景下的预测值。
    """
    predicted_values_best = []
    residuals_list_best = []
    model_names_best = []
    predictions_model4 = {t: [] for t in [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150]}

    for i in range(len(data_x)):
        row_x, row_y = data_x.iloc[i].dropna(), data_y.iloc[i].dropna()
        common_index = row_x.index.intersection(list(row_y.index))
        row_x_aligned, row_y_aligned = row_x.loc[common_index], row_y.loc[common_index]

        models = [predict_for_row_1, predict_for_row_2, predict_for_row_3, predict_for_row_4]
        predictions = [model(row_x_aligned, row_y_aligned, 100) for model in models]
        rss_values = [p[1] for p in predictions]
        valid_rss = [(idx, rss) for idx, rss in enumerate(rss_values) if pd.notna(rss)]

        if not valid_rss:
            predicted_values_best.append(np.nan)
            residuals_list_best.append(np.nan)
            model_names_best.append("无有效模型")
        else:
            best_model_idx, min_rss = min(valid_rss, key=lambda item: item[1])
            predicted_values_best.append(predictions[best_model_idx][0])
            residuals_list_best.append(min_rss)
            model_names_best.append(["二次多项式", "倒数函数-1", "倒数函数-2", "倒数函数-3"][best_model_idx])

        for t_val in predictions_model4.keys():
            pred, _ = predict_for_row_4(row_x_aligned, row_y_aligned, t_val)
            predictions_model4[t_val].append(pred)

    results_df = pd.DataFrame(index=data_x.index)
    results_df['百元平价溢价率-最佳模型'] = predicted_values_best
    results_df['残差平方和-最佳模型'] = residuals_list_best
    results_df['最佳模型名称'] = model_names_best
    for t_val, preds in predictions_model4.items():
        results_df[f'平价溢价率-混合模型-{t_val}'] = preds
        
    return results_df

# --- 主流程函数 ---

def run_metrics_calculation(input_path: str, output_path: str) -> bool:
    """
    执行所有指标计算步骤，从读取数据到最终写入Excel。
    """
    try:
        print("开始执行指标计算步骤...")

        print("读取输入文件...")
        sheets_to_read = ['1、转债收盘价', '2、转债YTM', '3、纯债价值', '4、转债余额', 
                          '5、纯债溢价率', '6、转换价值', '7、转股溢价率', '9、债股性', '10、债项评级']
        dfs = {sheet: pd.read_excel(input_path, sheet_name=sheet, parse_dates=['Date']).set_index('Date') for sheet in sheets_to_read}

        df1, df2, df3, df4, df5, df6, df7, df9, df10 = [dfs[s] for s in sheets_to_read]

        df11 = df1 * df4 / 100

        print("处理数据并计算基础指标...")
        mask_EDH = (df1 > 130) & (df7 > 30)
        df1_EDH, df6_EDH, df7_EDH = df1.mask(mask_EDH), df6.mask(mask_EDH), df7.mask(mask_EDH)

        mask_BC = (df4 > 15) & (df10.isin(['AAA', 'AA+', 'AA']))
        df1_EDH_BC, df6_EDH_BC, df7_EDH_BC = df1_EDH.where(mask_BC), df6_EDH.where(mask_BC), df7_EDH.where(mask_BC)
        df1_BC, df3_BC, df4_BC, df5_BC, df6_BC, df7_BC = [d.where(mask_BC) for d in [df1, df3, df4, df5, df6, df7]]

        print("计算各类指标...")
        df_LP = pd.DataFrame(index=df1.index)
        df_LP['转债数量'] = df1.count(axis=1)
        df_LP['低于面值转债数量'] = (df1 < 100).sum(axis=1)
        df_LP[['数量', '低价转债均价', '高价转债均价', '低价转债平价均值', '高价转债平价均值']] = df1.apply(
            lambda row: most_10_percent_mean(row, df6.loc[row.name]), axis=1)

        df_AP_S = pd.DataFrame(index=df1.index)
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        for q in quantiles:
            df_AP_S[f'价格{int(q*100)}%分位数'] = df1.quantile(q, axis=1).round(2)
            df_AP_S[f'平价{int(q*100)}%分位数'] = df6.quantile(q, axis=1).round(2)
        df_AP_S[['数量1', '数量2', '数量3', '数量4', '数量5', '5%溢价率', '25%溢价率', '50%溢价率', '75%溢价率', '95%溢价率']] = df6.apply(
            lambda row: conversion_premium(row, df7.loc[row.name]), axis=1)
        df_AP_S['跌破债底转债数量'] = (df1 < df3).sum(axis=1)

        print("计算平价溢价率回归模型...")
        df_regression_EDH = calculate_regression_metrics(df6_EDH, df7_EDH)
        df_regression_EDH_BC = calculate_regression_metrics(df6_EDH_BC, df7_EDH_BC)

        df_EDH_S = pd.DataFrame(index=df6_EDH.index)
        df_EDH_S['小于30'] = df7_EDH[df6_EDH < 30].mean(axis=1)
        df_EDH_S['大于130'] = df7_EDH[df6_EDH >= 130].mean(axis=1)
        df_EDH_S = df_EDH_S.join(df_regression_EDH).round(2)

        df_EDH_BC = pd.DataFrame(index=df6_EDH_BC.index)
        df_EDH_BC['小于70'] = df7_EDH_BC[df6_EDH_BC < 70].mean(axis=1)
        df_EDH_BC['大于130'] = df7_EDH_BC[df6_EDH_BC >= 130].mean(axis=1)
        df_EDH_BC = df_EDH_BC.join(df_regression_EDH_BC).round(2)
        
        df_Essence = pd.DataFrame(index=df1.index)
        df_Essence['转债价格-偏债型'] = df1[df9 < -20].mean(axis=1).round(2)
        df_Essence['转债价格-偏股型'] = df1[df9 > 20].mean(axis=1).round(2)
        df_Essence['转债价格-平衡型'] = df1[(df9 >= -20) & (df9 <= 20)].mean(axis=1).round(2)
        
        # 分解链式调用以帮助类型检查器
        ytm_filtered_df = df2[df9 < -20]
        ytm_median_series = ytm_filtered_df.apply(trimmed_median, axis=1)
        df_Essence['YTM-偏债型-中位数'] = ytm_median_series.round(2)

        df12 = df10.map(rate_group)
        df13 = df4.map(balance_group)

        df_Rate = pd.DataFrame(index=df1.index)
        ratings = ['评级低于A', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA']
        for r in ratings:
            mask = (df12 == r)
            df_Rate[f'价格- {r}'] = df1[mask].mean(axis=1).round(2)
            df_Rate[f'转股溢价率- {r}'] = df7[mask].mean(axis=1).round(2)
            df_Rate[f'支数- {r}'] = mask.sum(axis=1)
        df_Rate['支数- 总计'] = df12.count(axis=1)

        df_Balance = pd.DataFrame(index=df1.index)
        balances = ['剩余规模<3亿', '剩余规模3-10亿', '剩余规模10-50亿', '剩余规模>50亿']
        for b in balances:
            mask = (df13 == b)
            df_Balance[f'价格-{b}'] = df1[mask].mean(axis=1).round(2)
            df_Balance[f'转股溢价率-{b}'] = df7[mask].mean(axis=1).round(2)
            df_Balance[f'支数- {b}'] = mask.sum(axis=1)
        df_Balance['支数- 总计'] = df13.count(axis=1)

        print("保存计算结果到Excel文件...")
        with pd.ExcelWriter(input_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df11.to_excel(writer, sheet_name='11、存量规模')
            df12.to_excel(writer, sheet_name='12、债项评级分类')
            df13.to_excel(writer, sheet_name='13、转债余额分类')
            
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_LP.to_excel(writer, sheet_name='价格')
            df_AP_S.to_excel(writer, sheet_name='绝对-细分')
            df_EDH_S.to_excel(writer, sheet_name='平溢价')
            df_EDH_BC.to_excel(writer, sheet_name='平溢价-大盘')
            df_Essence.to_excel(writer, sheet_name='偏好')
            df_Balance.to_excel(writer, sheet_name='规模')
            df_Rate.to_excel(writer, sheet_name='评级')
            
        print("指标计算完成！")
        return True
    
    except Exception as e:
        print(f"指标计算过程中出现错误: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 确保配置文件路径正确
    if 'config' in sys.modules:
        input_path = config.STEP3_INPUT_FILE
        output_path = config.STEP3_OUTPUT_FILE
        success = run_metrics_calculation(input_path, output_path)
        if success:
            print("指标计算步骤执行成功")
        else:
            print("指标计算步骤执行失败")
    else:
        print("错误: 无法加载配置文件 'config.py'")