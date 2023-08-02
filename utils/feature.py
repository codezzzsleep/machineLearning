import pandas as pd


def data_status(df):
    """
       该函数从多个维度对所给的数据集进行评估

       参数:
       df (DataFrame):一个都是数值数据的DataFrame对象

       返回值:
       num_rows: 数据有多少行
       num_cols: 数据有多少列
       row_means: 每一行的平均值
       column_means: 每一列的平均值
       row_na_counts: 每一行有多少缺失值
       column_na_counts: 每一列有多少缺失值
       duplicate_counts: 统计数据中重复的数量
       """
    # 数据行数
    num_rows = df.shape[0]
    # 数据类数
    num_cols = df.shape[1]
    # 计算每一行的平均值
    row_means = df.mean(axis=1)
    # 计算每一列的平均值
    column_means = df.mean()
    # 统计每一行有多少缺失值(NA)
    row_na_counts = df.isna().sum(axis=1)
    # 统计每一列有多少缺失值(NA)
    column_na_counts = df.isna().sum()
    # 统计重复数据的数量
    duplicate_counts = df.duplicated().sum()




