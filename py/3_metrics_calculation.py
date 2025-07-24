import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import config

input_path = config.STEP3_INPUT_FILE
output_path = config.STEP3_OUTPUT_FILE

sheets_to_read = ['1、转债收盘价',
                  '2、转债YTM',
                  '3、纯债价值',
                  '4、转债余额',
                  '5、纯债溢价率',
                  '6、转换价值',
                  '7、转股溢价率',
                  '9、债股性',
                  '10、债项评级'
                  ]

dfs = {sheet: pd.read_excel(input_path, sheet_name=sheet, parse_dates=['Date'])
       for sheet in sheets_to_read}


df1 = dfs['1、转债收盘价'].set_index('Date')
df2 = dfs['2、转债YTM'].set_index('Date')
df3 = dfs['3、纯债价值'].set_index('Date')
df4 = dfs['4、转债余额'].set_index('Date')
df5 = dfs['5、纯债溢价率'].set_index('Date')
df6 = dfs['6、转换价值'].set_index('Date')
df7 = dfs['7、转股溢价率'].set_index('Date')
df9 = dfs['9、债股性'].set_index('Date')
df10 = dfs['10、债项评级'].set_index('Date')

# df11(存量规模) = 转债收盘价 * 转债余额 / 100
# 由于df1.index已经是DatetimeIndex，df4在set_index时也会正确对齐
df11 = df1 * df4.set_index(df1.index) / 100

# 将df11的索引转换为DatetimeIndex
df11.index = pd.to_datetime(df11.index)

# df4.set_index(df1.index) 确保索引一致性，如果df4的Date列与df1不同的话
# 更安全的做法是确保df4也使用其自身的（已解析的）Date索引，然后进行合并或对齐操作。
# 但如果df1.index是所有操作的基准，当前写法在df1.index是DatetimeIndex后问题不大。
# 为确保df4的索引也是日期类型，可以这样做：
# df4_indexed = dfs['4、转债余额'].set_index('Date') # df4_indexed.index 现在也是 DatetimeIndex
# df11 = df1 * df4_indexed.reindex(df1.index) / 100 # 使用reindex确保对齐和填充

bad = ['A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'CC', 'C']
good = ['AAA', 'AAA-']


# 构建函数 predict_for_row_1 - 二次多项式回归 y = β₀ + β₁x + β₂x^2 - 输出 x=100 的预测值
def predict_for_row_1(row_a, row_b, t):
    x = row_a.values.reshape(-1, 1)  # 将一维数组重塑为列向量，-1表示自动计算维度的大小，1表示每个元素视为一行。
    y = row_b.values.reshape(-1, 1)
    if len(x) < 6:
        return np.nan, np.nan  # 如果数据点不足，返回两个NaN值
    else:
        poly = PolynomialFeatures(degree=2)  # 创建多项式特征转换器，degree=2 表示使用二次多项式
        x_poly = poly.fit_transform(x)  # 将原始特征 x 转换为多项式特征。例如，如果x是[a]，则转换后变成[1, a, a²]
        model = LinearRegression()  # 创建线性回归模型
        model.fit(x_poly, y)  # 使用多项式特征和目标值训练模型
        # np.array([[100]])：创建一个包含值100的2D数组
        # poly.transform()：将值100转换为多项式特征[1, 100, 10000]
        # model.predict()：使用训练好的模型预测对应的y值
        # [0, 0]：获取结果数组中的第一个值
        y_predict = model.predict(poly.transform(np.array([[t]])))[0, 0]  # 当 x=100 时，y 的值是多少
        # 计算残差平方和
        residuals = y - model.predict(x_poly)
        residuals_sum = np.sum(residuals ** 2)
        return y_predict, residuals_sum


# 构建函数 predict_for_row_2 - 倒数函数-1 y * x = β₀ + β₁x - 输出 x=100 的预测值的 1%
def predict_for_row_2(row_a, row_b, t):
    x = row_a.values.reshape(-1, 1)
    y = row_b.values.reshape(-1, 1)
    if len(x) < 6:
        return np.nan, np.nan
    else:
        # 转换为 y*x = β₀ + β₁*x 形式
        y_transformed = y * x  # 计算 y*x 作为新的目标变量
        model = LinearRegression()
        model.fit(x, y_transformed)  # 拟合模型 y*x = β₀ + β₁*x
        # 预测 x=t 时的值
        y_x_predict = model.predict(np.array([[t]]))[0, 0]
        # 从 y*x 预测值计算 y 值：y = (β₀ + β₁*x)/x
        y_predict = y_x_predict / t
        # 计算原始数据的预测值
        y_pred = model.predict(x) / x.ravel()
        # 计算残差
        residuals = y.ravel() - y_pred
        # 计算残差平方和
        residuals_sum = np.sum(residuals ** 2)
        return y_predict, residuals_sum


# 构建函数 predict_for_row_3 - 倒数函数-2 y = β₀ + β₁ ( 1 / x ) - 输出 x=100 的预测值
def predict_for_row_3(row_a, row_b, t):
    x = row_a.values.reshape(-1, 1)
    y = row_b.values.reshape(-1, 1)
    if len(x) < 6:
        return np.nan, np.nan
    else:
        z = 1 / x
        model = LinearRegression()
        model.fit(z, y)
        y_predict = model.predict(np.array([[1 / t]]))[0, 0]
        residuals = y - model.predict(z)
        residuals_sum = np.sum(residuals ** 2)
        return y_predict, residuals_sum


# 构建函数 predict_for_row_4 - 倒数函数-3 y = β₀ + β₁x + β₂ ( 1 / x ) - 输出 x=100 的预测值
def predict_for_row_4(row_a, row_b, t):
    x = row_a.values.reshape(-1, 1)
    y = row_b.values.reshape(-1, 1)
    if len(x) < 6:
        return np.nan, np.nan
    else:
        z = 1 / x
        features = np.concatenate((x, z), axis=1)
        model = LinearRegression()
        model.fit(features, y)
        y_predict = model.predict(np.array([[t, 1 / t]]))[0, 0]
        residuals = y - model.predict(features)
        residuals_sum = np.sum(residuals ** 2)
        return y_predict, residuals_sum


# 构建函数 most_10_percent_mean
def most_10_percent_mean(row_a, row_b):
    # 删除row中的所有空值(NA/NaN)，返回一个新的Series
    # .dropna() 的主要功能是删除包含缺失值(NaN, None, NA)的行或列
    non_null_values = row_a.dropna()
    # 计算10%的数量，至少为1
    n = max(1, int(len(non_null_values) * 0.1))
    if len(non_null_values) == 0:  # 处理全空行
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # nsmallest(n) - 找出序列中最小的 n 个值
    # nlargest(n) - 找出序列中最大的 n 个值
    # mean() 是计算序列的算术平均值的方法。
    min_10_percent_mean = non_null_values.nsmallest(n).mean()
    max_10_percent_mean = non_null_values.nlargest(n).mean()
    min_valid_indices = non_null_values.nsmallest(n).index
    max_valid_indices = non_null_values.nlargest(n).index
    return (n,
            min_10_percent_mean,
            max_10_percent_mean,
            row_b.reindex(min_valid_indices).mean(),  # 使用reindex确保索引对齐
            row_b.reindex(max_valid_indices).mean())


# 构建函数 conversion_premium (转换溢价)
def conversion_premium(row_a, row_b):
    non_null_values = row_a.dropna()
    len_non_null = len(non_null_values)
    if len_non_null == 0:  # 处理全空行
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    i = max(1, int(len_non_null * 0.05))
    j = max(1, int(len_non_null * 0.25))
    m = max(1, int(len_non_null * 0.50))
    k = max(1, int(len_non_null * 0.75))
    l = max(1, int(len_non_null * 0.95))

    # 获取从小到大的第i, j, m, k, l个值的索引
    # .nsmallest(x).index[-1] 会给出第x小的项的索引 (如果x不大于len_non_null)
    # idxmax() on nsmallest(x) gives the index of the largest among the x smallest, which is the x-th smallest value's original index.
    min_5_index = non_null_values.nsmallest(min(i, len_non_null)).index[-1]
    min_25_index = non_null_values.nsmallest(min(j, len_non_null)).index[-1]
    min_50_index = non_null_values.nsmallest(min(m, len_non_null)).index[-1]
    min_75_index = non_null_values.nsmallest(min(k, len_non_null)).index[-1]
    min_95_index = non_null_values.nsmallest(min(l, len_non_null)).index[-1]

    return (i,
            j,
            m,
            k,
            l,
            row_b.get(min_5_index, np.nan),  # 使用.get获取值，避免KeyError
            row_b.get(min_25_index, np.nan),
            row_b.get(min_50_index, np.nan),
            row_b.get(min_75_index, np.nan),
            row_b.get(min_95_index, np.nan))


# 构建函数 parity_price (平价价格)
# 这个函数与 conversion_premium 结构非常相似，可能需要确认其具体逻辑是否与 conversion_premium 不同
# 假设它是用来取特定分位数上的 row_b 值的，其索引由 row_a 的分位数决定
def parity_price(row_a, row_b):
    non_null_values = row_a.dropna()
    len_non_null = len(non_null_values)
    if len_non_null == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    i = max(1, int(len_non_null * 0.05))
    j = max(1, int(len_non_null * 0.25))
    m = max(1, int(len_non_null * 0.50))
    k = max(1, int(len_non_null * 0.75))
    l = max(1, int(len_non_null * 0.95))

    min_5_index = non_null_values.nsmallest(min(i, len_non_null)).index[-1]
    min_25_index = non_null_values.nsmallest(min(j, len_non_null)).index[-1]
    min_50_index = non_null_values.nsmallest(min(m, len_non_null)).index[-1]
    min_75_index = non_null_values.nsmallest(min(k, len_non_null)).index[-1]
    min_95_index = non_null_values.nsmallest(min(l, len_non_null)).index[-1]

    return (row_b.get(min_5_index, np.nan),
            row_b.get(min_25_index, np.nan),
            row_b.get(min_50_index, np.nan),
            row_b.get(min_75_index, np.nan),
            row_b.get(min_95_index, np.nan))


# 构建函数 rate_group (评级分组)
def rate_group(rate):
    if pd.isna(rate):  # 处理可能的 NaN 值
        return np.nan
    if rate in bad:
        return '评级低于A'
    elif rate in good:
        return 'AAA'
    else:
        return rate


# 构建函数 balance_group (余额分组)
def balance_group(balance):
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


# 构建函数 trimmed_mean (去除极端值后的平均数)
def trimmed_mean(row):
    # Ensure row is a Series for dropna()
    if not isinstance(row, pd.Series):
        row = pd.Series(row)
    sorted_row = np.sort(row.dropna())  # 对行进行排序
    if len(sorted_row) < 7:
        # return np.mean(row) # np.mean on original row might include NaNs if not handled
        return np.mean(sorted_row) if len(sorted_row) > 0 else np.nan
    else:
        return sorted_row[3:-3].mean()


# 构建函数 trimmed_median (去除NAN后的中位数)
def trimmed_median(row):
    # Ensure row is a Series for dropna()
    if not isinstance(row, pd.Series):
        row = pd.Series(row)
    sorted_row = np.sort(row.dropna())  # 对行进行排序
    return np.median(sorted_row) if len(sorted_row) > 0 else np.nan


# 基于df6的值范围，将df7分为5个不同的区间
# 对每个区间内的 df7 值计算行平均值
# 创建一个新的DataFrame，其中每列代表一个区间内的平均值
# df6.index 已经是 DatetimeIndex
df = pd.DataFrame(index=df6.index)
df['转股溢价率-小于70'] = df7[df6 < 70].apply(np.mean, axis=1)
df['转股溢价率-大于70，小于90'] = df7[(df6 >= 70) & (df6 < 90)].apply(np.mean, axis=1)
df['转股溢价率-大于90，小于110'] = df7[(df6 >= 90) & (df6 < 110)].apply(np.mean, axis=1)
df['转股溢价率-大于110，小于130'] = df7[(df6 >= 110) & (df6 < 130)].apply(np.mean, axis=1)
df['转股溢价率-大于130'] = df7[df6 >= 130].apply(np.mean, axis=1)

# 剔除条件：A > 130 且 B > 30
# _EDH : _Eliminate_double_highs (剔除双高)
# 创建一个布尔掩码，用于标识需要剔除的单元格
mask_EDH = (df1 > 130) & (df7 > 30)
# 使用掩码将符合条件的单元格替换为 NaN
# .mask preserves index type
df1_EDH = df1.mask(mask_EDH, np.nan)
df6_EDH = df6.mask(mask_EDH, np.nan)
df7_EDH = df7.mask(mask_EDH, np.nan)
print(df1_EDH.count(axis=1), "\n", df7_EDH.count(axis=1))

# 创建布尔掩码，用于标识需要保留的单元格
# 保留条件：大盘
# _EDH : _Eliminate_double_highs (剔除双高)
# _BC : _big_cap (大盘)
mask_BC = (df4 > 15) & (df10.isin(['AAA', 'AA+', 'AA']))
# .where preserves index type
df1_EDH_BC = df1_EDH.where(mask_BC, np.nan)
df6_EDH_BC = df6_EDH.where(mask_BC, np.nan)
df7_EDH_BC = df7_EDH.where(mask_BC, np.nan)
df1_BC = df1.where(mask_BC, np.nan)  # np.nan - 即 NumPy 中的 NaN 值
df3_BC = df3.where(mask_BC, np.nan)
df4_BC = df4.where(mask_BC, np.nan)
df5_BC = df5.where(mask_BC, np.nan)
df6_BC = df6.where(mask_BC, np.nan)
df7_BC = df7.where(mask_BC, np.nan)

# _EDH : _Eliminate_double_highs (剔除双高)
# df6_EDH.index is DatetimeIndex
df_EDH = pd.DataFrame(index=df6_EDH.index)
df_EDH['小于70'] = df7_EDH[df6_EDH < 70].apply(np.mean, axis=1)
df_EDH['大于70，小于90'] = df7_EDH[(df6_EDH >= 70) & (df6_EDH < 90)].apply(np.mean, axis=1)
df_EDH['大于90，小于110'] = df7_EDH[(df6_EDH >= 90) & (df6_EDH < 110)].apply(np.mean, axis=1)
df_EDH['大于110，小于130'] = df7_EDH[(df6_EDH >= 110) & (df6_EDH < 130)].apply(np.mean, axis=1)
df_EDH['大于130'] = df7_EDH[df6_EDH >= 130].apply(np.mean, axis=1)

# _EDH : _Eliminate_double_highs (剔除双高)
# _BC : _big_cap (大盘)
# df6_EDH_BC.index is DatetimeIndex
df_EDH_BC = pd.DataFrame(index=df6_EDH_BC.index)
df_EDH_BC['小于70'] = df7_EDH_BC[df6_EDH_BC < 70].apply(np.mean, axis=1)
df_EDH_BC['大于70，小于90'] = df7_EDH_BC[(df6_EDH_BC >= 70) & (df6_EDH_BC < 90)].apply(np.mean, axis=1)
df_EDH_BC['大于90，小于110'] = df7_EDH_BC[(df6_EDH_BC >= 90) & (df6_EDH_BC < 110)].apply(np.mean, axis=1)
df_EDH_BC['大于110，小于130'] = df7_EDH_BC[(df6_EDH_BC >= 110) & (df6_EDH_BC < 130)].apply(np.mean, axis=1)
df_EDH_BC['大于130'] = df7_EDH_BC[df6_EDH_BC >= 130].apply(np.mean, axis=1)

# _EDH : _Eliminate_double_highs (剔除双高)
# _S : _Segmentation (细分)
# df6_EDH.index is DatetimeIndex
df_EDH_S = pd.DataFrame(index=df6_EDH.index)
df_EDH_S['小于30'] = df7_EDH[df6_EDH < 30].apply(np.mean, axis=1)
df_EDH_S['大于30，小于50'] = df7_EDH[(df6_EDH >= 30) & (df6_EDH < 50)].apply(np.mean, axis=1)
df_EDH_S['大于50，小于70'] = df7_EDH[(df6_EDH >= 50) & (df6_EDH < 70)].apply(np.mean, axis=1)
df_EDH_S['大于70，小于90'] = df7_EDH[(df6_EDH >= 70) & (df6_EDH < 90)].apply(np.mean, axis=1)
df_EDH_S['大于90，小于110'] = df7_EDH[(df6_EDH >= 90) & (df6_EDH < 110)].apply(np.mean, axis=1)
df_EDH_S['大于110，小于130'] = df7_EDH[(df6_EDH >= 110) & (df6_EDH < 130)].apply(np.mean, axis=1)
df_EDH_S['大于130'] = df7_EDH[df6_EDH >= 130].apply(np.mean, axis=1)

# 生成 DateFrame '低价转债指标'
# _LP : _Low_price(低价)
# df1.index is DatetimeIndex
df_LP = pd.DataFrame(index=df1.index)
df_LP['转债数量'] = df1.count(axis=1)  # 计算 df1 每行非空值的元素数量
df_LP['低于面值转债数量'] = df1.apply(lambda row: sum(row < 100), axis=1)  # 计算 df1 每行小于 100 的元素数量
df_LP['小于90元转债数量'] = df1.apply(lambda row: sum(row < 90), axis=1)  # 计算 df1 每行小于 90 的元素数量
df_LP['小于80转债数量'] = df1.apply(lambda row: sum(row < 80), axis=1)  # 计算 df1 每行小于 80 的元素数量
df_LP['小于60转债数量'] = df1.apply(lambda row: sum(row < 60), axis=1)  # 计算 df1 每行小于 60 的元素数量
df_LP[['数量',
       '低价转债均价',
       '高价转债均价',
       '低价转债平价均值',
       '高价转债平价均值']] = df1.apply(
    lambda row: most_10_percent_mean(row, df6.loc[row.name]),
    axis=1,
    result_type='expand')
print(df_LP.head(10))

# 生成 DateFrame '估值指标-绝对价位比较'
# _AP : _Absolute_price(绝对价格)
# df1.index is DatetimeIndex
df_AP = pd.DataFrame(index=df1.index)
df_AP['价格中位数'] = df1.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['纯债价值中位数'] = df3.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['转股价值中位数'] = df6.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['纯债溢价率中位数'] = df5.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['转股溢价率中位数'] = df7.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['价格均值'] = df1.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['纯债价值均值'] = df3.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['转股价值均值'] = df6.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['纯债溢价率均值'] = df5.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['转股溢价率均值'] = df7.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP['转债数量1'] = df1.count(axis=1)
df_AP['转债数量2'] = df6.count(axis=1)
df_AP['跌破债底转债数量'] = (df1 < df3).sum(axis=1)
df_AP['转债余额'] = df4.sum(axis=1)
# df_AP['转债市值合计'] = df11.sum(axis=1)
# The following line is fine; if index is already datetime, it's a no-op or standardizes.
# If format="%Y-%m-%d" is crucial for parsing string dates, it should be used when `pd.to_datetime` is first applied.
# Given parse_dates in read_excel, df_AP.index is already DatetimeIndex.
df_AP.index = pd.to_datetime(df_AP.index, format="%Y-%m-%d", errors='ignore')  # errors='ignore' if already datetime
print(df_AP.head(10))

# 生成 DateFrame '估值指标-绝对价位比较-大盘转债'
# _AP : _Absolute_price(绝对价格)
# _BC : _big_cap (大盘)
# df1.index is DatetimeIndex (hence df1_BC.index is also)
df_AP_BC = pd.DataFrame(
    index=df1_BC.index)  # Changed from df1.index to df1_BC.index for consistency if df1_BC could have a subset of dates
df_AP_BC['价格中位数'] = df1_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['纯债价值中位数'] = df3_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['转股价值中位数'] = df6_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['纯债溢价率中位数'] = df5_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['转股溢价率中位数'] = df7_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['价格均值'] = df1_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['纯债价值均值'] = df3_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['转股价值均值'] = df6_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['纯债溢价率均值'] = df5_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['转股溢价率均值'] = df7_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC['转债数量1'] = df1_BC.count(axis=1)
df_AP_BC['转债数量2'] = df6_BC.count(axis=1)
df_AP_BC['转债余额'] = df4_BC.sum(axis=1)
df_AP_BC.index = pd.to_datetime(df_AP_BC.index, format="%Y-%m-%d", errors='ignore')
print(df_AP_BC.head(10))

# 生成 DateFrame '估值指标-绝对价位比较-细分'
# _AP : _Absolute_price(绝对价格)
# _S : _Segmentation (细分)
df_AP_S = pd.DataFrame(index=df1.index)
# 在 df1 '转债收盘价' 中取分位数点
df_AP_S['价格5%分位数'] = df1.quantile(0.05, axis=1, numeric_only=True).round(2)
df_AP_S['价格25%分位数'] = df1.quantile(0.25, axis=1, numeric_only=True).round(2)
df_AP_S['价格中位数'] = df1.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['价格75%分位数'] = df1.quantile(0.75, axis=1, numeric_only=True).round(2)
df_AP_S['价格95%分位数'] = df1.quantile(0.95, axis=1, numeric_only=True).round(2)
# 在 df6 '转换价值' 中取分位数点
df_AP_S['平价5%分位数'] = df6.quantile(0.05, axis=1, numeric_only=True).round(2)
df_AP_S['平价25%分位数'] = df6.quantile(0.25, axis=1, numeric_only=True).round(2)
df_AP_S['平价中位数'] = df6.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['平价75%分位数'] = df6.quantile(0.75, axis=1, numeric_only=True).round(2)
df_AP_S['平价95%分位数'] = df6.quantile(0.95, axis=1, numeric_only=True).round(2)
# 根据 df6 '转换价值' 中的分位数点，取 df7 '转股溢价率' 的值
df_AP_S[['数量1',
         '数量2',
         '数量3',
         '数量4',
         '数量5',
         '5%溢价率',
         '25%溢价率',
         '50%溢价率',
         '75%溢价率',
         '95%溢价率']] = df6.apply(
    lambda row: conversion_premium(row, df7.loc[row.name]),
    axis=1,
    result_type='expand')
# 在 df3 '纯债价值' df5 '纯债溢价率' df7 '转股溢价率' 中取中位数点
df_AP_S['债底中位数'] = df3.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['纯债溢价率中位数'] = df5.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['转股溢价率中位数'] = df7.median(axis=1, skipna=True, numeric_only=True).round(2)
# 对所有 DateFrame 求均值
df_AP_S['价格均值'] = df1.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['纯债价值均值'] = df3.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['转股价值均值'] = df6.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['纯债溢价率均值'] = df5.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_S['转股溢价率均值'] = df7.mean(axis=1, skipna=True, numeric_only=True).round(2)
# 求 df1 '转债收盘价' df6 '转换价值' 中的数量
df_AP_S['转债数量1'] = df1.count(axis=1)
df_AP_S['转债数量2'] = df6.count(axis=1)
df_AP_S['跌破债底转债数量'] = (df1 < df3).sum(axis=1)
# df_AP_S[['5%平价价格',
#          '25%平价价格',
#          '50%平价价格',
#          '75%平价价格',
#          '95%平价价格']] = df6.apply(
#     lambda row: parity_price(row, df1.loc[row.name]),
#     axis=1,
#     result_type='expand')
df_AP_S.index = pd.to_datetime(df_AP_S.index, format="%Y-%m-%d", errors='ignore')
print(df_AP_S.head(10))

# 生成 DateFrame '估值指标-绝对价位比较-细分-大盘转债'
# _AP : _Absolute_price(绝对价格)
# _BC : _big_cap (大盘)
# _S : _Segmentation (细分)
df_AP_BC_S = pd.DataFrame(index=df1_BC.index)
df_AP_BC_S['价格5%分位数'] = df1_BC.quantile(0.05, axis=1, numeric_only=True).round(2)
df_AP_BC_S['价格25%分位数'] = df1_BC.quantile(0.25, axis=1, numeric_only=True).round(2)
df_AP_BC_S['价格中位数'] = df1_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['价格75%分位数'] = df1_BC.quantile(0.75, axis=1, numeric_only=True).round(2)
df_AP_BC_S['价格95%分位数'] = df1_BC.quantile(0.95, axis=1, numeric_only=True).round(2)
df_AP_BC_S['平价5%分位数'] = df6_BC.quantile(0.05, axis=1, numeric_only=True).round(2)
df_AP_BC_S['平价25%分位数'] = df6_BC.quantile(0.25, axis=1, numeric_only=True).round(2)
df_AP_BC_S['平价中位数'] = df6_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['平价75%分位数'] = df6_BC.quantile(0.75, axis=1, numeric_only=True).round(2)
df_AP_BC_S['平价95%分位数'] = df6_BC.quantile(0.95, axis=1, numeric_only=True).round(2)
df_AP_BC_S[['数量1',
            '数量2',
            '数量3',
            '数量4',
            '数量5',
            '5%溢价率',
            '25%溢价率',
            '50%溢价率',
            '75%溢价率',
            '95%溢价率']] = df6_BC.apply(
    lambda row: conversion_premium(row, df7_BC.loc[row.name]),  # Changed df7 to df7_BC for consistency
    axis=1,
    result_type='expand')
df_AP_BC_S['债底中位数'] = df3_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['纯债溢价率中位数'] = df5_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['转股溢价率中位数'] = df7_BC.median(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['价格均值'] = df1_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['纯债价值均值'] = df3_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['转股价值均值'] = df6_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['纯债溢价率均值'] = df5_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['转股溢价率均值'] = df7_BC.mean(axis=1, skipna=True, numeric_only=True).round(2)
df_AP_BC_S['转债数量1'] = df1_BC.count(axis=1)
df_AP_BC_S['转债数量2'] = df6_BC.count(axis=1)
df_AP_BC_S['跌破债底转债数量'] = (df1_BC < df3_BC).sum(axis=1)
# df_AP_BC_S[['5%平价价格',
#             '25%平价价格',
#             '50%平价价格',
#             '75%平价价格',
#             '95%平价价格']] = df6_BC.apply(
#     lambda row: parity_price(row, df1_BC.loc[row.name]),
#     axis=1,
#     result_type='expand')
df_AP_BC_S.index = pd.to_datetime(df_AP_BC_S.index, format="%Y-%m-%d", errors='ignore')
print(df_AP_BC_S.head(10))

# 使用 apply 函数对每一行进行迭代操作
# _EDH : _Eliminate_double_highs (剔除双高)
# _CMM : _contain_target_model (包含混合模型)
# _BC : _big_cap (大盘)
# 三模型最佳
predicted_values_best_model = []
residuals_list_best_model = []
predicted_values_best_model_EDH = []
residuals_list_best_model_EDH = []
predicted_values_best_model_EDH_BC = []
residuals_list_best_model_EDH_BC = []
# 目标模型
predicted_values_target_model = []
residuals_list_target_model = []
predicted_values_target_model_EDH = []
residuals_list_target_model_EDH = []
predicted_values_target_model_EDH_BC = []
residuals_list_target_model_EDH_BC = []
# 混合模型不同取值
predicted_values_target_model_EDH_40 = []
predicted_values_target_model_EDH_50 = []
predicted_values_target_model_EDH_60 = []
predicted_values_target_model_EDH_70 = []
predicted_values_target_model_EDH_80 = []
predicted_values_target_model_EDH_90 = []
predicted_values_target_model_EDH_100 = []
predicted_values_target_model_EDH_110 = []
predicted_values_target_model_EDH_120 = []
predicted_values_target_model_EDH_130 = []
predicted_values_target_model_EDH_150 = []
predicted_values_target_model_EDH_BC_40 = []
predicted_values_target_model_EDH_BC_50 = []
predicted_values_target_model_EDH_BC_60 = []
predicted_values_target_model_EDH_BC_70 = []
predicted_values_target_model_EDH_BC_80 = []
predicted_values_target_model_EDH_BC_90 = []
predicted_values_target_model_EDH_BC_100 = []
predicted_values_target_model_EDH_BC_110 = []
predicted_values_target_model_EDH_BC_120 = []
predicted_values_target_model_EDH_BC_130 = []
predicted_values_target_model_EDH_BC_150 = []
# 三模型&混合模型最佳
target_model = []
residuals_list_best_model_EDH_CMM = []
predicted_values_best_model_EDH_BC_CMM = []
residuals_list_best_model_EDH_BC_CMM = []

# Note: The model_input variable is determined by the last row's best model in the original loops.
# This might not be ideal if different rows have different best models.
# The column names like '百元平价溢价率-{}'.format(model_input) will use the last model_input.
# This part of the logic is kept as is, focusing on the index type.

model_input = "未知模型"  # Default value
model_input_EDH = "未知模型"
model_input_EDH_BC = "未知模型"

for i in range(len(df6)):  # df6.index is DatetimeIndex
    # .iloc[i] gives a Series. .dropna() is fine.
    # Common index for row_df6 and row_df7 is bond codes.
    row_df6 = df6.iloc[i].dropna()
    row_df7 = df7.iloc[i].dropna()
    # Align data before passing to prediction functions
    common_index = row_df6.index.intersection(row_df7.index)
    row_df6_aligned = row_df6.loc[common_index]
    row_df7_aligned = row_df7.loc[common_index]

    Y_predict3, RSS3 = predict_for_row_3(row_df6_aligned, row_df7_aligned, 100)
    Y_predict2, RSS2 = predict_for_row_2(row_df6_aligned, row_df7_aligned, 100)
    Y_predict1, RSS1 = predict_for_row_1(row_df6_aligned, row_df7_aligned, 100)
    Y_predict4, RSS4 = predict_for_row_4(row_df6_aligned, row_df7_aligned, 100)

    # print(
    #     f"Row {df6.index[i]} RSS值: 二次多项式={RSS1 if pd.notna(RSS1) else np.nan:.4f} ,倒数函数-1={RSS2 if pd.notna(RSS2) else np.nan:.4f}, 倒数函数-2={RSS3 if pd.notna(RSS3) else np.nan:.4f}, 倒数函数-3={RSS4 if pd.notna(RSS4) else np.nan:.4f}")

    # Handle cases where all RSS values might be NaN
    rss_values = [RSS1, RSS2, RSS3, RSS4]
    valid_rss_values = [r for r in rss_values if pd.notna(r)]
    predicted_values_target_model.append(Y_predict4)
    residuals_list_target_model.append(RSS4)

    if not valid_rss_values:  # All RSS are NaN
        predicted_values_best_model.append(np.nan)
        residuals_list_best_model.append(np.nan)
        model_input = "无有效模型"  # Or keep previous model_input
    else:
        min_RSS = min(valid_rss_values)
        if min_RSS == RSS1:
            predicted_values_best_model.append(Y_predict1)
            residuals_list_best_model.append(RSS1)
            model_input = "二次多项式"
        elif min_RSS == RSS2:  # Check if RSS2 is part of valid_rss_values and equals min_RSS
            predicted_values_best_model.append(Y_predict2)
            residuals_list_best_model.append(RSS2)
            model_input = "倒数函数-1"
        elif min_RSS == RSS3:  # Check if RSS3 is part of valid_rss_values and equals min_RSS
            predicted_values_best_model.append(Y_predict3)
            residuals_list_best_model.append(RSS3)
            model_input = "倒数函数-2"
        elif min_RSS == RSS4:
            predicted_values_best_model.append(Y_predict4)
            residuals_list_best_model.append(RSS4)
            model_input = "倒数函数-3"
        # Fallback if min_RSS matches a NaN value (should not happen with valid_rss_values logic)
        # Or if multiple models give the same min_RSS, priority is as above.


# _EDH : _Eliminate_double_highs (剔除双高)
for i in range(len(df6_EDH)):  # df6_EDH.index is DatetimeIndex
    row_df6_EDH = df6_EDH.iloc[i].dropna()
    row_df7_EDH = df7_EDH.iloc[i].dropna()
    common_index_EDH = row_df6_EDH.index.intersection(row_df7_EDH.index)
    row_df6_EDH_aligned = row_df6_EDH.loc[common_index_EDH]
    row_df7_EDH_aligned = row_df7_EDH.loc[common_index_EDH]

    Y_predict3_EDH, RSS3_EDH = predict_for_row_3(row_df6_EDH_aligned, row_df7_EDH_aligned, 100)
    Y_predict2_EDH, RSS2_EDH = predict_for_row_2(row_df6_EDH_aligned, row_df7_EDH_aligned, 100)
    Y_predict1_EDH, RSS1_EDH = predict_for_row_1(row_df6_EDH_aligned, row_df7_EDH_aligned, 100)
    Y_predict4_EDH, RSS4_EDH = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 100)
    # print(
    #     f"Row {df6_EDH.index[i]} (EDH) RSS值: 二次多项式={RSS1_EDH if pd.notna(RSS1_EDH) else np.nan:.4f} ,倒数函数-1={RSS2_EDH if pd.notna(RSS2_EDH) else np.nan:.4f}, 倒数函数-2={RSS3_EDH if pd.notna(RSS3_EDH) else np.nan:.4f}, 倒数函数-3={RSS4_EDH if pd.notna(RSS4_EDH) else np.nan:.4f}")

    rss_values_EDH = [RSS1_EDH, RSS2_EDH, RSS3_EDH, RSS4_EDH]
    valid_rss_values_EDH = [r for r in rss_values_EDH if pd.notna(r)]

    if not valid_rss_values_EDH:
        predicted_values_best_model_EDH.append(np.nan)
        residuals_list_best_model_EDH.append(np.nan)
        model_input_EDH = "无有效模型"
    else:
        min_RSS_EDH = min(valid_rss_values_EDH)
        if min_RSS_EDH == RSS1_EDH:
            predicted_values_best_model_EDH.append(Y_predict1_EDH)
            residuals_list_best_model_EDH.append(RSS1_EDH)
            model_input_EDH = "二次多项式"
        elif min_RSS_EDH == RSS2_EDH:
            predicted_values_best_model_EDH.append(Y_predict2_EDH)
            residuals_list_best_model_EDH.append(RSS2_EDH)
            model_input_EDH = "倒数函数-1"
        elif min_RSS_EDH == RSS3_EDH:
            predicted_values_best_model_EDH.append(Y_predict3_EDH)
            residuals_list_best_model_EDH.append(RSS3_EDH)
            model_input_EDH = "倒数函数-2"
        elif min_RSS_EDH == RSS4_EDH:
            predicted_values_best_model_EDH.append(Y_predict4_EDH)
            residuals_list_best_model_EDH.append(RSS4_EDH)
            model_input_EDH = "倒数函数-3"

# ... (Similar alignment logic for other loops if necessary) ...

# _EDH : _Eliminate_double_highs (剔除双高)
for i in range(len(df6_EDH)):
    row_df6_EDH = df6_EDH.iloc[i].dropna()
    row_df7_EDH = df7_EDH.iloc[i].dropna()
    common_index_EDH = row_df6_EDH.index.intersection(row_df7_EDH.index)
    row_df6_EDH_aligned = row_df6_EDH.loc[common_index_EDH]
    row_df7_EDH_aligned = row_df7_EDH.loc[common_index_EDH]

    Y_predict4_EDH, RSS4_EDH = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 100)
    Y_predict4_EDH_40, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 40)
    Y_predict4_EDH_50, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 50)
    Y_predict4_EDH_60, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 60)
    Y_predict4_EDH_70, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 70)
    Y_predict4_EDH_80, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 80)
    Y_predict4_EDH_90, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 90)
    Y_predict4_EDH_100, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 100)
    Y_predict4_EDH_110, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 110)
    Y_predict4_EDH_120, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 120)
    Y_predict4_EDH_130, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 130)
    Y_predict4_EDH_150, _ = predict_for_row_4(row_df6_EDH_aligned, row_df7_EDH_aligned, 150)

    predicted_values_target_model_EDH.append(Y_predict4_EDH)
    residuals_list_target_model_EDH.append(RSS4_EDH)
    predicted_values_target_model_EDH_40.append(Y_predict4_EDH_40)
    predicted_values_target_model_EDH_50.append(Y_predict4_EDH_50)
    predicted_values_target_model_EDH_60.append(Y_predict4_EDH_60)
    predicted_values_target_model_EDH_70.append(Y_predict4_EDH_70)
    predicted_values_target_model_EDH_80.append(Y_predict4_EDH_80)
    predicted_values_target_model_EDH_90.append(Y_predict4_EDH_90)
    predicted_values_target_model_EDH_100.append(Y_predict4_EDH_100)
    predicted_values_target_model_EDH_110.append(Y_predict4_EDH_110)
    predicted_values_target_model_EDH_120.append(Y_predict4_EDH_120)
    predicted_values_target_model_EDH_130.append(Y_predict4_EDH_130)
    predicted_values_target_model_EDH_150.append(Y_predict4_EDH_150)

# _EDH : _Eliminate_double_highs (剔除双高)
# _CMM : _contain_target_model (包含混合模型)
print(len(df_EDH),  # df_EDH.index is DatetimeIndex
      len(predicted_values_best_model_EDH),
      len(predicted_values_target_model_EDH),
      )

# _EDH : _Eliminate_double_highs (剔除双高)
# _BC : _big_cap (大盘)
for i in range(len(df6_EDH_BC)):  # df6_EDH_BC.index is DatetimeIndex
    row_df6_EDH_BC = df6_EDH_BC.iloc[i].dropna()
    row_df7_EDH_BC = df7_EDH_BC.iloc[i].dropna()
    common_index_EDH_BC = row_df6_EDH_BC.index.intersection(row_df7_EDH_BC.index)
    row_df6_EDH_BC_aligned = row_df6_EDH_BC.loc[common_index_EDH_BC]
    row_df7_EDH_BC_aligned = row_df7_EDH_BC.loc[common_index_EDH_BC]

    Y_predict3_EDH_BC, RSS3_EDH_BC = predict_for_row_3(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 100)
    Y_predict2_EDH_BC, RSS2_EDH_BC = predict_for_row_2(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 100)
    Y_predict1_EDH_BC, RSS1_EDH_BC = predict_for_row_1(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 100)
    Y_predict4_EDH_BC, RSS4_EDH_BC = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 100)
    # print(
    #     f"Row {df6_EDH_BC.index[i]} (EDH_BC) RSS值: 二次多项式={RSS1_EDH_BC if pd.notna(RSS1_EDH_BC) else np.nan:.4f} ,倒数函数-1={RSS2_EDH_BC if pd.notna(RSS2_EDH_BC) else np.nan:.4f}, 倒数函数-2={RSS3_EDH_BC if pd.notna(RSS3_EDH_BC) else np.nan:.4f}, 倒数函数-3={RSS4_EDH_BC if pd.notna(RSS4_EDH_BC) else np.nan:.4f}")

    rss_values_EDH_BC = [RSS1_EDH_BC, RSS2_EDH_BC, RSS3_EDH_BC, RSS4_EDH_BC]
    valid_rss_values_EDH_BC = [r for r in rss_values_EDH_BC if pd.notna(r)]

    if not valid_rss_values_EDH_BC:
        predicted_values_best_model_EDH_BC.append(np.nan)
        residuals_list_best_model_EDH_BC.append(np.nan)
        model_input_EDH_BC = "无有效模型"
    else:
        min_RSS_EDH_BC = min(valid_rss_values_EDH_BC)
        if min_RSS_EDH_BC == RSS1_EDH_BC:
            predicted_values_best_model_EDH_BC.append(Y_predict1_EDH_BC)
            residuals_list_best_model_EDH_BC.append(RSS1_EDH_BC)
            model_input_EDH_BC = "二次多项式"
        elif min_RSS_EDH_BC == RSS2_EDH_BC:
            predicted_values_best_model_EDH_BC.append(Y_predict2_EDH_BC)
            residuals_list_best_model_EDH_BC.append(RSS2_EDH_BC)
            model_input_EDH_BC = "倒数函数-1"
        elif min_RSS_EDH_BC == RSS3_EDH_BC:
            predicted_values_best_model_EDH_BC.append(Y_predict3_EDH_BC)
            residuals_list_best_model_EDH_BC.append(RSS3_EDH_BC)
            model_input_EDH_BC = "倒数函数-2"
        elif min_RSS_EDH_BC == RSS4_EDH_BC:
            predicted_values_best_model_EDH_BC.append(Y_predict4_EDH_BC)
            residuals_list_best_model_EDH_BC.append(RSS4_EDH_BC)
            model_input_EDH_BC = "倒数函数-3"

# _EDH : _Eliminate_double_highs (剔除双高)
# _BC : _big_cap (大盘)
for i in range(len(df6_EDH_BC)):  # df6_EDH_BC used here, ensure its length matches lists
    row_df6_EDH_BC = df6_EDH_BC.iloc[i].dropna()
    row_df7_EDH_BC = df7_EDH_BC.iloc[i].dropna()
    common_index_EDH_BC = row_df6_EDH_BC.index.intersection(row_df7_EDH_BC.index)
    row_df6_EDH_BC_aligned = row_df6_EDH_BC.loc[common_index_EDH_BC]
    row_df7_EDH_BC_aligned = row_df7_EDH_BC.loc[common_index_EDH_BC]

    Y_predict4_EDH_BC, RSS4_EDH_BC = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 100)
    Y_predict4_EDH_BC_40, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 40)
    Y_predict4_EDH_BC_50, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 50)
    Y_predict4_EDH_BC_60, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 60)
    Y_predict4_EDH_BC_70, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 70)
    Y_predict4_EDH_BC_80, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 80)
    Y_predict4_EDH_BC_90, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 90)
    Y_predict4_EDH_BC_100, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 100)
    Y_predict4_EDH_BC_110, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 110)
    Y_predict4_EDH_BC_120, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 120)
    Y_predict4_EDH_BC_130, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 130)
    Y_predict4_EDH_BC_150, _ = predict_for_row_4(row_df6_EDH_BC_aligned, row_df7_EDH_BC_aligned, 150)

    predicted_values_target_model_EDH_BC.append(Y_predict4_EDH_BC)
    residuals_list_target_model_EDH_BC.append(RSS4_EDH_BC)
    predicted_values_target_model_EDH_BC_40.append(Y_predict4_EDH_BC_40)
    predicted_values_target_model_EDH_BC_50.append(Y_predict4_EDH_BC_50)
    predicted_values_target_model_EDH_BC_60.append(Y_predict4_EDH_BC_60)
    predicted_values_target_model_EDH_BC_70.append(Y_predict4_EDH_BC_70)
    predicted_values_target_model_EDH_BC_80.append(Y_predict4_EDH_BC_80)
    predicted_values_target_model_EDH_BC_90.append(Y_predict4_EDH_BC_90)
    predicted_values_target_model_EDH_BC_100.append(Y_predict4_EDH_BC_100)
    predicted_values_target_model_EDH_BC_110.append(Y_predict4_EDH_BC_110)
    predicted_values_target_model_EDH_BC_120.append(Y_predict4_EDH_BC_120)
    predicted_values_target_model_EDH_BC_130.append(Y_predict4_EDH_BC_130)
    predicted_values_target_model_EDH_BC_150.append(Y_predict4_EDH_BC_150)

# _EDH : _Eliminate_double_highs (剔除双高)
# _CMM : _contain_target_model (包含混合模型)
# _BC : _big_cap (大盘)
print(len(df_EDH_BC),  # df_EDH_BC.index is DatetimeIndex
      len(predicted_values_best_model_EDH_BC),
      len(predicted_values_target_model_EDH_BC),
      )

# df.index is DatetimeIndex
df['百元平价溢价率-{}'.format(model_input)] = predicted_values_best_model
df['百元平价溢价率-目标模型'] = predicted_values_target_model
print(df['百元平价溢价率-{}'.format(model_input)])
print(df['百元平价溢价率-目标模型'])
df['残差平方和-{}'.format(model_input)] = residuals_list_best_model
df['残差平方和-目标模型'] = residuals_list_target_model
print(df['残差平方和-{}'.format(model_input)])
print(df['残差平方和-目标模型'])
sum_residual_best_model = pd.Series(residuals_list_best_model).sum()  # Ensure Series for sum()
sum_residual_target_model = pd.Series(residuals_list_target_model).sum()  # Ensure Series for sum()
print(sum_residual_best_model, sum_residual_target_model)
df['转债数量1'] = df6.count(axis=1)
df['转债数量2'] = df7.count(axis=1)
df['小于70支数'] = df6.apply(lambda row: sum(row < 70), axis=1)
df['大于70，小于90支数'] = df6.apply(lambda row: sum((row >= 70) & (row < 90)), axis=1)
df['大于90，小于110支数'] = df6.apply(lambda row: sum((row >= 90) & (row < 110)), axis=1)
df['大于110，小于130支数'] = df6.apply(lambda row: sum((row >= 110) & (row < 130)), axis=1)
df['大于130支数'] = df6.apply(lambda row: sum(row >= 130), axis=1)
df['价格-小于70'] = df1[df6 < 70].apply(np.mean, axis=1)
df['价格-大于70，小于90'] = df1[(df6 >= 70) & (df6 < 90)].apply(np.mean, axis=1)
df['价格-大于90，小于110'] = df1[(df6 >= 90) & (df6 < 110)].apply(np.mean, axis=1)
df['价格-大于110，小于130'] = df1[(df6 >= 110) & (df6 < 130)].apply(np.mean, axis=1)
df['价格-大于130'] = df1[df6 >= 130].apply(np.mean, axis=1)
df['平价-小于70'] = df6[df6 < 70].apply(np.mean, axis=1)
df['平价-大于70，小于90'] = df6[(df6 >= 70) & (df6 < 90)].apply(np.mean, axis=1)
df['平价-大于90，小于110'] = df6[(df6 >= 90) & (df6 < 110)].apply(np.mean, axis=1)
df['平价-大于110，小于130'] = df6[(df6 >= 110) & (df6 < 130)].apply(np.mean, axis=1)
df['平价-大于130'] = df6[df6 >= 130].apply(np.mean, axis=1)
df = df.round(2)
df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors='ignore')
print(df.head(10))

# _EDH : _Eliminate_double_highs (剔除双高)
# _CMM : _contain_target_model (包含混合模型)
# df_EDH.index is DatetimeIndex
# Check lengths before assigning to prevent errors if lists are shorter than index
df_EDH_len = len(df_EDH.index)
df_EDH['百元平价溢价率-{}'.format(model_input_EDH)] = pd.Series(predicted_values_best_model_EDH, index=df_EDH.index[
                                                                                                       :len(
                                                                                                           predicted_values_best_model_EDH)])
df_EDH['百元平价溢价率-目标模型'] = pd.Series(predicted_values_target_model_EDH,
                                              index=df_EDH.index[:len(predicted_values_target_model_EDH)])
print(df_EDH[['百元平价溢价率-{}'.format(model_input_EDH),
              '百元平价溢价率-目标模型']])

df_EDH['残差平方和-{}'.format(model_input_EDH)] = pd.Series(residuals_list_best_model_EDH,
                                                            index=df_EDH.index[:len(residuals_list_best_model_EDH)])
df_EDH['残差平方和-混合模型'] = pd.Series(residuals_list_target_model_EDH,
                                          index=df_EDH.index[:len(residuals_list_target_model_EDH)])
print(df_EDH[['残差平方和-{}'.format(model_input_EDH),
              '残差平方和-混合模型']])
df_EDH['转债数量1'] = df6.count(axis=1)  # Should be df6_EDH? Original code uses df6.
df_EDH['转债数量2'] = df7.count(axis=1)  # Should be df7_EDH?
df_EDH['转债数量1-剔除双高'] = df6_EDH.count(axis=1)
df_EDH['转债数量2-剔除双高'] = df7_EDH.count(axis=1)
df_EDH['小于70支数'] = df6_EDH.apply(lambda row: sum(row < 70), axis=1)
df_EDH['大于70，小于90支数'] = df6_EDH.apply(lambda row: sum((row >= 70) & (row < 90)), axis=1)
df_EDH['大于90，小于110支数'] = df6_EDH.apply(lambda row: sum((row >= 90) & (row < 110)), axis=1)
df_EDH['大于110，小于130支数'] = df6_EDH.apply(lambda row: sum((row >= 110) & (row < 130)), axis=1)
df_EDH['大于130支数'] = df6_EDH.apply(lambda row: sum(row >= 130), axis=1)
df_EDH['价格-小于70'] = df1_EDH[df6_EDH < 70].apply(np.mean, axis=1)
df_EDH['价格-大于70，小于90'] = df1_EDH[(df6_EDH >= 70) & (df6_EDH < 90)].apply(np.mean, axis=1)
df_EDH['价格-大于90，小于110'] = df1_EDH[(df6_EDH >= 90) & (df6_EDH < 110)].apply(np.mean, axis=1)
df_EDH['价格-大于110，小于130'] = df1_EDH[(df6_EDH >= 110) & (df6_EDH < 130)].apply(np.mean, axis=1)
df_EDH['价格-大于130'] = df1_EDH[df6_EDH >= 130].apply(np.mean, axis=1)
df_EDH['平价-小于70'] = df6_EDH[df6_EDH < 70].apply(np.mean, axis=1)
df_EDH['平价-大于70，小于90'] = df6_EDH[(df6_EDH >= 70) & (df6_EDH < 90)].apply(np.mean, axis=1)
df_EDH['平价-大于90，小于110'] = df6_EDH[(df6_EDH >= 90) & (df6_EDH < 110)].apply(np.mean, axis=1)
df_EDH['平价-大于110，小于130'] = df6_EDH[(df6_EDH >= 110) & (df6_EDH < 130)].apply(np.mean, axis=1)
df_EDH['平价-大于130'] = df6_EDH[df6_EDH >= 130].apply(np.mean, axis=1)

df_EDH['平价溢价率-_40'] = pd.Series(predicted_values_target_model_EDH_40,
                                     index=df_EDH.index[:len(predicted_values_target_model_EDH_40)])
df_EDH['平价溢价率-_50'] = pd.Series(predicted_values_target_model_EDH_50,
                                     index=df_EDH.index[:len(predicted_values_target_model_EDH_50)])
df_EDH['平价溢价率-_60'] = pd.Series(predicted_values_target_model_EDH_60,
                                     index=df_EDH.index[:len(predicted_values_target_model_EDH_60)])
df_EDH['平价溢价率-_70'] = pd.Series(predicted_values_target_model_EDH_70,
                                     index=df_EDH.index[:len(predicted_values_target_model_EDH_70)])
df_EDH['平价溢价率-_80'] = pd.Series(predicted_values_target_model_EDH_80,
                                     index=df_EDH.index[:len(predicted_values_target_model_EDH_80)])
df_EDH['平价溢价率-_90'] = pd.Series(predicted_values_target_model_EDH_90,
                                     index=df_EDH.index[:len(predicted_values_target_model_EDH_90)])
df_EDH['平价溢价率-_100'] = pd.Series(predicted_values_target_model_EDH_100,
                                      index=df_EDH.index[:len(predicted_values_target_model_EDH_100)])
df_EDH['平价溢价率-_110'] = pd.Series(predicted_values_target_model_EDH_110,
                                      index=df_EDH.index[:len(predicted_values_target_model_EDH_110)])
df_EDH['平价溢价率-_120'] = pd.Series(predicted_values_target_model_EDH_120,
                                      index=df_EDH.index[:len(predicted_values_target_model_EDH_120)])
df_EDH['平价溢价率-_130'] = pd.Series(predicted_values_target_model_EDH_130,
                                      index=df_EDH.index[:len(predicted_values_target_model_EDH_130)])
df_EDH['平价溢价率-_150'] = pd.Series(predicted_values_target_model_EDH_150,
                                      index=df_EDH.index[:len(predicted_values_target_model_EDH_150)])

df_EDH = df_EDH.round(2)
df_EDH.index = pd.to_datetime(df_EDH.index, format="%Y-%m-%d", errors='ignore')
print(df_EDH.head(10))

# _EDH : _Eliminate_double_highs (剔除双高)
# _CMM : _contain_target_model (包含混合模型)
# _BC : _big_cap (大盘)
# df_EDH_BC.index is DatetimeIndex
df_EDH_BC_len = len(df_EDH_BC.index)
df_EDH_BC['百元平价溢价率-{}'.format(model_input_EDH_BC)] = pd.Series(predicted_values_best_model_EDH_BC,
                                                                      index=df_EDH_BC.index[
                                                                            :len(predicted_values_best_model_EDH_BC)])
df_EDH_BC['百元平价溢价率-混合模型'] = pd.Series(predicted_values_target_model_EDH_BC,
                                                 index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC)])

print(df_EDH_BC[['百元平价溢价率-{}'.format(model_input_EDH_BC),
                 '百元平价溢价率-混合模型']])

df_EDH_BC['残差平方和-{}'.format(model_input_EDH_BC)] = pd.Series(residuals_list_best_model_EDH_BC,
                                                                  index=df_EDH_BC.index[
                                                                        :len(residuals_list_best_model_EDH_BC)])
df_EDH_BC['残差平方和-混合模型'] = pd.Series(residuals_list_target_model_EDH_BC,
                                             index=df_EDH_BC.index[:len(residuals_list_target_model_EDH_BC)])

print(df_EDH_BC[['残差平方和-{}'.format(model_input_EDH_BC),
                 '残差平方和-混合模型']])
df_EDH_BC['转债数量1'] = df6_BC.count(axis=1)  # Original used df6_BC
df_EDH_BC['转债数量2'] = df7_BC.count(axis=1)  # Original used df7_BC
df_EDH_BC['转债数量1-剔除双高'] = df6_EDH_BC.count(axis=1)
df_EDH_BC['转债数量2-剔除双高'] = df7_EDH_BC.count(axis=1)
df_EDH_BC['小于70支数'] = df6_EDH_BC.apply(lambda row: sum(row < 70), axis=1)
df_EDH_BC['大于70，小于90支数'] = df6_EDH_BC.apply(lambda row: sum((row >= 70) & (row < 90)), axis=1)
df_EDH_BC['大于90，小于110支数'] = df6_EDH_BC.apply(lambda row: sum((row >= 90) & (row < 110)), axis=1)
df_EDH_BC['大于110，小于130支数'] = df6_EDH_BC.apply(lambda row: sum((row >= 110) & (row < 130)), axis=1)
df_EDH_BC['大于130支数'] = df6_EDH_BC.apply(lambda row: sum(row >= 130), axis=1)
df_EDH_BC['价格-小于70'] = df1_EDH_BC[df6_EDH_BC < 70].apply(np.mean, axis=1)
df_EDH_BC['价格-大于70，小于90'] = df1_EDH_BC[(df6_EDH_BC >= 70) & (df6_EDH_BC < 90)].apply(np.mean, axis=1)
df_EDH_BC['价格-大于90，小于110'] = df1_EDH_BC[(df6_EDH_BC >= 90) & (df6_EDH_BC < 110)].apply(np.mean, axis=1)
df_EDH_BC['价格-大于110，小于130'] = df1_EDH_BC[(df6_EDH_BC >= 110) & (df6_EDH_BC < 130)].apply(np.mean, axis=1)
df_EDH_BC['价格-大于130'] = df1_EDH_BC[df6_EDH_BC >= 130].apply(np.mean, axis=1)
df_EDH_BC['平价-小于70'] = df6_EDH_BC[df6_EDH_BC < 70].apply(np.mean, axis=1)
df_EDH_BC['平价-大于70，小于90'] = df6_EDH_BC[(df6_EDH_BC >= 70) & (df6_EDH_BC < 90)].apply(np.mean, axis=1)
df_EDH_BC['平价-大于90，小于110'] = df6_EDH_BC[(df6_EDH_BC >= 90) & (df6_EDH_BC < 110)].apply(np.mean, axis=1)
df_EDH_BC['平价-大于110，小于130'] = df6_EDH_BC[(df6_EDH_BC >= 110) & (df6_EDH_BC < 130)].apply(np.mean, axis=1)
df_EDH_BC['平价-大于130'] = df6_EDH_BC[df6_EDH_BC >= 130].apply(np.mean, axis=1)

df_EDH_BC['平价溢价率-_40'] = pd.Series(predicted_values_target_model_EDH_BC_40,
                                        index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_40)])
df_EDH_BC['平价溢价率-_50'] = pd.Series(predicted_values_target_model_EDH_BC_50,
                                        index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_50)])
df_EDH_BC['平价溢价率-_60'] = pd.Series(predicted_values_target_model_EDH_BC_60,
                                        index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_60)])
df_EDH_BC['平价溢价率-_70'] = pd.Series(predicted_values_target_model_EDH_BC_70,
                                        index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_70)])
df_EDH_BC['平价溢价率-_80'] = pd.Series(predicted_values_target_model_EDH_BC_80,
                                        index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_80)])
df_EDH_BC['平价溢价率-_90'] = pd.Series(predicted_values_target_model_EDH_BC_90,
                                        index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_90)])
df_EDH_BC['平价溢价率-_100'] = pd.Series(predicted_values_target_model_EDH_BC_100,
                                         index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_100)])
df_EDH_BC['平价溢价率-_110'] = pd.Series(predicted_values_target_model_EDH_BC_110,
                                         index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_110)])
df_EDH_BC['平价溢价率-_120'] = pd.Series(predicted_values_target_model_EDH_BC_120,
                                         index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_120)])
df_EDH_BC['平价溢价率-_130'] = pd.Series(predicted_values_target_model_EDH_BC_130,
                                         index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_130)])
df_EDH_BC['平价溢价率-_150'] = pd.Series(predicted_values_target_model_EDH_BC_150,
                                         index=df_EDH_BC.index[:len(predicted_values_target_model_EDH_BC_150)])

df_EDH_BC = df_EDH_BC.round(2)
df_EDH_BC.index = pd.to_datetime(df_EDH_BC.index, format="%Y-%m-%d", errors='ignore')
print(df_EDH_BC.head(10))

# _EDH : _Eliminate_double_highs (剔除双高)
# _CMM : _contain_target_model (包含混合模型)
# _S : _Segmentation (细分)
# df_EDH_S.index is DatetimeIndex
df_EDH_S_len = len(df_EDH_S.index)
df_EDH_S['百元平价溢价率-{}'.format(model_input_EDH)] = pd.Series(predicted_values_best_model_EDH, index=df_EDH_S.index[
                                                                                                         :len(
                                                                                                             predicted_values_best_model_EDH)])
df_EDH_S['百元平价溢价率-混合模型'] = pd.Series(predicted_values_target_model_EDH,
                                                index=df_EDH_S.index[:len(predicted_values_target_model_EDH)])

df_EDH_S['残差平方和-{}'.format(model_input_EDH)] = pd.Series(residuals_list_best_model_EDH,
                                                              index=df_EDH_S.index[:len(residuals_list_best_model_EDH)])
df_EDH_S['残差平方和-混合模型'] = pd.Series(residuals_list_target_model_EDH,
                                            index=df_EDH_S.index[:len(residuals_list_target_model_EDH)])


df_EDH_S['转债数量1'] = df6.count(axis=1)  # Should be df6_EDH?
df_EDH_S['转债数量2'] = df7.count(axis=1)  # Should be df7_EDH?
df_EDH_S['转债数量1-剔除双高'] = df6_EDH.count(axis=1)
df_EDH_S['转债数量2-剔除双高'] = df7_EDH.count(axis=1)
df_EDH_S['小于30支数'] = df6_EDH.apply(lambda row: sum(row < 30), axis=1)
df_EDH_S['大于30，小于50支数'] = df6_EDH.apply(lambda row: sum((row >= 30) & (row < 50)), axis=1)
df_EDH_S['大于50，小于70支数'] = df6_EDH.apply(lambda row: sum((row >= 50) & (row < 70)), axis=1)
df_EDH_S['大于70，小于90支数'] = df6_EDH.apply(lambda row: sum((row >= 70) & (row < 90)), axis=1)
df_EDH_S['大于90，小于110支数'] = df6_EDH.apply(lambda row: sum((row >= 90) & (row < 110)), axis=1)
df_EDH_S['大于110，小于130支数'] = df6_EDH.apply(lambda row: sum((row >= 110) & (row < 130)), axis=1)
df_EDH_S['大于130支数'] = df6_EDH.apply(lambda row: sum(row >= 130), axis=1)
df_EDH_S['价格-小于30'] = df1_EDH[df6_EDH < 30].apply(np.mean, axis=1)
df_EDH_S['价格-大于30，小于50'] = df1_EDH[(df6_EDH >= 30) & (df6_EDH < 50)].apply(np.mean, axis=1)
df_EDH_S['价格-大于50，小于70'] = df1_EDH[(df6_EDH >= 50) & (df6_EDH < 70)].apply(np.mean, axis=1)
df_EDH_S['价格-大于70，小于90'] = df1_EDH[(df6_EDH >= 70) & (df6_EDH < 90)].apply(np.mean, axis=1)
df_EDH_S['价格-大于90，小于110'] = df1_EDH[(df6_EDH >= 90) & (df6_EDH < 110)].apply(np.mean, axis=1)
df_EDH_S['价格-大于110，小于130'] = df1_EDH[(df6_EDH >= 110) & (df6_EDH < 130)].apply(np.mean, axis=1)
df_EDH_S['价格-大于130'] = df1_EDH[df6_EDH >= 130].apply(np.mean, axis=1)
df_EDH_S['平价-小于30'] = df6_EDH[df6_EDH < 30].apply(np.mean, axis=1)
df_EDH_S['平价-大于30，小于50'] = df6_EDH[(df6_EDH >= 30) & (df6_EDH < 50)].apply(np.mean, axis=1)
df_EDH_S['平价-大于50，小于70'] = df6_EDH[(df6_EDH >= 50) & (df6_EDH < 70)].apply(np.mean, axis=1)
df_EDH_S['平价-大于70，小于90'] = df6_EDH[(df6_EDH >= 70) & (df6_EDH < 90)].apply(np.mean, axis=1)
df_EDH_S['平价-大于90，小于110'] = df6_EDH[(df6_EDH >= 90) & (df6_EDH < 110)].apply(np.mean, axis=1)
df_EDH_S['平价-大于110，小于130'] = df6_EDH[(df6_EDH >= 110) & (df6_EDH < 130)].apply(np.mean, axis=1)
df_EDH_S['平价-大于130'] = df6_EDH[df6_EDH >= 130].apply(np.mean, axis=1)

df_EDH_S['平价溢价率-_40'] = pd.Series(predicted_values_target_model_EDH_40,
                                       index=df_EDH_S.index[:len(predicted_values_target_model_EDH_40)])
df_EDH_S['平价溢价率-_50'] = pd.Series(predicted_values_target_model_EDH_50,
                                       index=df_EDH_S.index[:len(predicted_values_target_model_EDH_50)])
df_EDH_S['平价溢价率-_60'] = pd.Series(predicted_values_target_model_EDH_60,
                                       index=df_EDH_S.index[:len(predicted_values_target_model_EDH_60)])
df_EDH_S['平价溢价率-_70'] = pd.Series(predicted_values_target_model_EDH_70,
                                       index=df_EDH_S.index[:len(predicted_values_target_model_EDH_70)])
df_EDH_S['平价溢价率-_80'] = pd.Series(predicted_values_target_model_EDH_80,
                                       index=df_EDH_S.index[:len(predicted_values_target_model_EDH_80)])
df_EDH_S['平价溢价率-_90'] = pd.Series(predicted_values_target_model_EDH_90,
                                       index=df_EDH_S.index[:len(predicted_values_target_model_EDH_90)])
df_EDH_S['平价溢价率-_100'] = pd.Series(predicted_values_target_model_EDH_100,
                                      index=df_EDH_S.index[:len(predicted_values_target_model_EDH_100)])
df_EDH_S['平价溢价率-_110'] = pd.Series(predicted_values_target_model_EDH_110,
                                        index=df_EDH_S.index[:len(predicted_values_target_model_EDH_110)])
df_EDH_S['平价溢价率-_120'] = pd.Series(predicted_values_target_model_EDH_120,
                                        index=df_EDH_S.index[:len(predicted_values_target_model_EDH_120)])
df_EDH_S['平价溢价率-_130'] = pd.Series(predicted_values_target_model_EDH_130,
                                        index=df_EDH_S.index[:len(predicted_values_target_model_EDH_130)])
df_EDH_S['平价溢价率-_150'] = pd.Series(predicted_values_target_model_EDH_150,
                                        index=df_EDH_S.index[:len(predicted_values_target_model_EDH_150)])

df_EDH_S = df_EDH_S.round(2)
df_EDH_S.index = pd.to_datetime(df_EDH_S.index, format="%Y-%m-%d", errors='ignore')
print(df_EDH_S.head(10))

# 生成 DateFrame '偏股、偏债、平衡型转债估值指标'
# df1.index is DatetimeIndex
df_Essence = pd.DataFrame(index=df1.index)
df_Essence['转债价格-偏债型'] = df1[df9 < -20].apply(np.mean, axis=1).round(2)
df_Essence['转债价格-偏股型'] = df1[df9 > 20].apply(np.mean, axis=1).round(2)
df_Essence['转债价格-平衡型'] = df1[(df9 >= -20) & (df9 <= 20)].apply(np.mean, axis=1).round(2)
df_Essence['转债平价-偏债型'] = df6[df9 < -20].apply(np.mean, axis=1).round(2)
df_Essence['转债平价-偏股型'] = df6[df9 > 20].apply(np.mean, axis=1).round(2)
df_Essence['转债平价-平衡型'] = df6[(df9 >= -20) & (df9 <= 20)].apply(np.mean, axis=1).round(2)
df_Essence['转股溢价率-偏债型'] = df7[df9 < -20].apply(np.mean, axis=1).round(2)
df_Essence['转股溢价率-偏股型'] = df7[df9 > 20].apply(np.mean, axis=1).round(2)
df_Essence['转股溢价率-平衡型'] = df7[(df9 >= -20) & (df9 <= 20)].apply(np.mean, axis=1).round(2)
df_Essence['转债数量-偏债型'] = df9[df9 < -20].count(axis=1)
df_Essence['转债数量-偏股型'] = df9[df9 > 20].count(axis=1)
df_Essence['转债数量-平衡型'] = df9[(df9 >= -20) & (df9 <= 20)].count(axis=1)
df_Essence['转债数量-总计'] = df9.count(axis=1)
df_Essence['YTM'] = df2.apply(np.mean, axis=1).round(2)
df_Essence['YTM-剔除极值'] = df2.apply(trimmed_mean, axis=1).round(2)
df_Essence['YTM-偏债型'] = df2[df9 < -20].apply(np.mean, axis=1).round(2)
df_Essence['YTM-偏债型-剔除极值'] = df2[df9 < -20].apply(trimmed_mean, axis=1).round(2)
df_Essence['YTM-中位数'] = df2.apply(trimmed_median, axis=1).round(2)
df_Essence['YTM-偏债型-中位数'] = df2[df9 < -20].apply(trimmed_median, axis=1).round(2)
df_Essence.index = pd.to_datetime(df_Essence.index, format="%Y-%m-%d", errors='ignore')
print(df_Essence.head(10))

# df10.index and df4.index are DatetimeIndex
# Assuming .map was intended as .applymap for DataFrames
# If df10/df4 are Series, .map is fine. Resulting index will be DatetimeIndex.
df12 = df10.applymap(rate_group) if isinstance(df10, pd.DataFrame) else df10.map(rate_group)
df13 = df4.applymap(balance_group) if isinstance(df4, pd.DataFrame) else df4.map(balance_group)

# 生成 DateFrame '不同评级转债估值指标'
# df1.index is DatetimeIndex
df_Rate = pd.DataFrame(index=df1.index)
df_Rate['价格- 评级低于A'] = df1[df12 == '评级低于A'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['价格- A'] = df1[df12 == 'A'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['价格- A+'] = df1[df12 == 'A+'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['价格- AA-'] = df1[df12 == 'AA-'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['价格- AA'] = df1[df12 == 'AA'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['价格- AA+'] = df1[df12 == 'AA+'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['价格- AAA'] = df1[df12 == 'AAA'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['转股溢价率- 评级低于A'] = df7[df12 == '评级低于A'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['转股溢价率- A'] = df7[df12 == 'A'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['转股溢价率- A+'] = df7[df12 == 'A+'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['转股溢价率- AA-'] = df7[df12 == 'AA-'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['转股溢价率- AA'] = df7[df12 == 'AA'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['转股溢价率- AA+'] = df7[df12 == 'AA+'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['转股溢价率- AAA'] = df7[df12 == 'AAA'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Rate['支数- 评级低于A'] = df12[df12 == '评级低于A'].count(axis=1)
df_Rate['支数- A'] = df12[df12 == 'A'].count(axis=1)
df_Rate['支数- A+'] = df12[df12 == 'A+'].count(axis=1)
df_Rate['支数- AA-'] = df12[df12 == 'AA-'].count(axis=1)
df_Rate['支数- AA'] = df12[df12 == 'AA'].count(axis=1)
df_Rate['支数- AA+'] = df12[df12 == 'AA+'].count(axis=1)
df_Rate['支数- AAA'] = df12[df12 == 'AAA'].count(axis=1)
df_Rate['支数- 总计'] = df12.count(axis=1)
df_Rate.index = pd.to_datetime(df_Rate.index, format="%Y-%m-%d", errors='ignore')
print(df_Rate.head(10))

# df1.index is DatetimeIndex
df_Balance = pd.DataFrame(index=df1.index)
df_Balance['价格-剩余规模<3亿 '] = df1[df13 == '剩余规模<3亿'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Balance['价格-剩余规模3-10亿'] = df1[df13 == '剩余规模3-10亿'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Balance['价格-剩余规模10-50亿'] = df1[df13 == '剩余规模10-50亿'].mean(axis=1, skipna=True, numeric_only=True).round(
    2)
df_Balance['价格-剩余规模>50亿'] = df1[df13 == '剩余规模>50亿'].mean(axis=1, skipna=True, numeric_only=True).round(2)
df_Balance['转股溢价率-剩余规模<3亿'] = df7[df13 == '剩余规模<3亿'].mean(axis=1, skipna=True, numeric_only=True).round(
    2)
df_Balance['转股溢价率-剩余规模3-10亿'] = df7[df13 == '剩余规模3-10亿'].mean(axis=1, skipna=True,
                                                                             numeric_only=True).round(2)
df_Balance['转股溢价率-剩余规模10-50亿'] = df7[df13 == '剩余规模10-50亿'].mean(axis=1, skipna=True,
                                                                               numeric_only=True).round(2)
df_Balance['转股溢价率-剩余规模>50亿'] = df7[df13 == '剩余规模>50亿'].mean(axis=1, skipna=True,
                                                                           numeric_only=True).round(2)
df_Balance['支数- 剩余规模<3亿'] = df13[df13 == '剩余规模<3亿'].count(axis=1)
df_Balance['支数- 剩余规模3-10亿'] = df13[df13 == '剩余规模3-10亿'].count(axis=1)
df_Balance['支数- 剩余规模10-50亿'] = df13[df13 == '剩余规模10-50亿'].count(axis=1)
df_Balance['支数- 剩余规模>50亿'] = df13[df13 == '剩余规模>50亿'].count(axis=1)
df_Balance['支数- 总计'] = df13.count(axis=1)
df_Balance.index = pd.to_datetime(df_Balance.index, format="%Y-%m-%d", errors='ignore')
print(df_Balance.head(10))

with pd.ExcelWriter(input_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    df11.to_excel(writer, sheet_name='11、存量规模')
    df12.to_excel(writer, sheet_name='12、债项评级分类')
    df13.to_excel(writer, sheet_name='13、转债余额分类')
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_LP.to_excel(writer, sheet_name='价格')  # '低价转债指标'
    df_AP.to_excel(writer, sheet_name='绝对')  # '估值指标-绝对价位比较'
    df_AP_BC.to_excel(writer, sheet_name='绝对-大盘')  # '估值指标-绝对价位比较-大盘转债'
    df_AP_S.to_excel(writer, sheet_name='绝对-细分')  # '估值指标-绝对价位比较-细分'
    df_AP_BC_S.to_excel(writer, sheet_name='绝对-细分-大盘')  # '估值指标-绝对价位比较-细分-大盘转债'
    # df.to_excel(writer, sheet_name='转债估值指标-各平价溢价率-{}'.format(model_input)) # '转债估值指标-各平价溢价率-{}'
    # df_EDH.to_excel(writer, sheet_name='估值-平价溢价-剔除双高-不细分') # '转债估值指标-各平价溢价率-剔除双高-拟合'
    df_EDH_S.to_excel(writer, sheet_name='平溢价')  # '转债估值指标-各平价溢价率-剔除双高-细分-拟合'
    df_EDH_BC.to_excel(writer, sheet_name='平溢价-大盘')  # '转债估值指标-各平价溢价率-剔除双高-拟合-大盘转债'
    df_Essence.to_excel(writer, sheet_name='偏好')  # '偏股、偏债、平衡型转债估值指标'
    df_Balance.to_excel(writer, sheet_name='规模')  # '不同规模转债估值指标'
    df_Rate.to_excel(writer, sheet_name='评级')  # '不同评级转债估值指标'
print("Finished")

