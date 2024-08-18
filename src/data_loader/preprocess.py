"""
@Description: data prepare utils
@Author: laziyu
@Date: 2023-11-29
"""

import time
import warnings
import os
import numpy as np
import pandas as pd

from data_loader.data_operation import merge, rand_select_col

warnings.filterwarnings("ignore")


def make_overlapping(dataset, overlapping_rate, noise_sizes, seed=0):
    """
    make overlapping dataset by overlapping rate
    """
    print(f"Dataset shape : {dataset.shape}")
    num_column = dataset.shape[1]
    overlapping_num = round(num_column * overlapping_rate)
    overlapping_columns = rand_select_col(dataset, overlapping_num, seed)
    print(f"Overlapping rate in dataset : {overlapping_rate}")
    print(f"Noise sizes: {noise_sizes}")
    # print(f"Overlapping column num in dataset : {overlapping_num}")
    print(f"Additional column sum in dataset: {overlapping_num*len(noise_sizes)}")
    # 删除之前原本的列
    dataset = dataset.drop(overlapping_columns.columns, axis=1)
    for noise in noise_sizes:
        if noise == 0:
            noised_df = overlapping_columns
        else:
            noised_df = add_multiple_noise(noise, overlapping_columns, seed)
        dataset = merge(dataset, noised_df)
        # print(dataset)
    print(f"New dataset shape : {dataset.shape}")
    return dataset


def add_column_name(df):
    """
    Add column name for df
    """
    # 检查df是否已经有列名
    if df.columns.tolist() == range(df.shape[1]):  # 如果列名是默认的整数索引
        # 为列创建新的名称，使用 'feature_i' 格式，其中 i 是列的索引（从1开始）
        new_column_names = ["feature_" + str(i) for i in range(1, df.shape[1] + 1)]
        # 将新的列名赋给df的columns属性
        df.columns = new_column_names
    return df


def add_new_feature(X, y):
    """
    Add new features to X based on the descriptive statistics of y.

    Parameters:
    X (pd.DataFrame): The input data frame to which new features will be added.
    y (pd.DataFrame): The target data frame from which features will be extracted.

    Returns:
    pd.DataFrame: The input data frame X with new features added.
    """
    # 确保X和y都是DataFrame类型
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # 计算y的描述性统计量
    stats = y.describe(include="all")  # 包括所有数值型列的统计描述，默认只包括数值型

    # 从统计描述中提取特征
    for column in y.columns:
        # 已有的特征
        X[f"{column}_min"] = stats.loc["min", column]
        X[f"{column}_max"] = stats.loc["max", column]
        X[f"{column}_mean"] = stats.loc["mean", column]
        X[f"{column}_std"] = stats.loc["std", column]  # 标准差
        X[f"{column}_median"] = stats.loc["50%", column]  # 中位数

        # 可选的特征
        # 偏度
        skewness = y[column].skew()
        X[f"{column}_skew"] = skewness
        # 峰度
        kurtosis = y[column].kurtosis()
        X[f"{column}_kurt"] = kurtosis

    # 返回添加了新特征的X
    return X


def normalize(data):
    """
    Normalize the data to the range [0, 1] using min-max scaling.

    Parameters:
    data (pd.DataFrame): The data to be normalized.

    Returns:
    pd.DataFrame: The normalized data.
    """
    min_vals = data.min()
    max_vals = data.max()
    range_vals = max_vals - min_vals

    # Avoid division by zero in case any feature has all the same values
    range_vals[range_vals == 0] = 1

    normalized_data = (data - min_vals) / range_vals
    return normalized_data


def standardize(data):
    """
    Standardize the data to have mean 0 and standard deviation 1.

    Parameters:
    data (pd.DataFrame): The data to be standardized.

    Returns:
    pd.DataFrame: The standardized data.
    """
    mean_vals = data.mean()
    std_vals = data.std()

    # Avoid division by zero in case any feature has zero standard deviation
    std_vals[std_vals == 0] = 1

    standardized_data = (data - mean_vals) / std_vals
    return standardized_data


def fillna_by_mean(df):
    # 遍历DataFrame的每一列
    for column in df.columns:
        # 计算列的平均值
        mean_value = df[column].mean()
        # 将该列的NaN值替换为列的平均值
        df[column].fillna(mean_value, inplace=True)
    return df


def map_char_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == "object":  # 如果列的数据类型是字符型
            mapping = {val: idx for idx, val in enumerate(df[column].unique())}
            df[column] = df[column].map(mapping)
    return df


def preprocess_dataset(df):
    """
    preprocess dataset
    """
    df = map_char_to_numeric(df)
    # df = standardize(df)
    # df = normalize(df)
    df = df.fillna(0.0)
    return df


def split_dataset(dataset, silo_num=4, row_num=None, seed=0):
    """
    split dataset
    :param dataset: dataset
    :param silo_num: number of silo
    :param row_num: number of rows
    :param seed: random seed
    :return: silos
    """
    dataset_row = dataset.shape[0]
    dataset_column = dataset.shape[1]
    if row_num == None:
        row_num = dataset_row
    else:
        row_num = min(row_num, dataset_row)
    silo_length = dataset_column // silo_num
    # 分成n份,一次取silo_length个,最后一份取剩下的
    silos = []
    np.random.seed(seed)
    for i in range(silo_num - 1):
        silos.append(
            dataset.iloc[:, i * silo_length : (i + 1) * silo_length].sample(row_num)
        )
    silos.append(dataset.iloc[:, (silo_num - 1) * silo_length :].sample(row_num))
    return silos


def add_overlapping_column(silos, overlapping_rate, seed):
    """
    add overlapping columns to each silo from a random selection of other silos.
    :param silos: A list of pandas DataFrames representing different data silos.
    :param overlapping_rate: A float between 0 and 1 representing the proportion of columns to overlap.
    :param seed: An integer used to seed the random number generator for reproducibility.
    :return: A list of pandas DataFrames with added overlapping columns and overlapping num list
    """
    overlapping_num_lst = []
    overlapping_silo_lst = []
    for i, silo in enumerate(silos):
        # 从其他列中抽取num列
        if i == 0:
            other_silos = merge(silos[i + 1 :], ignore_index=True)
        else:
            other_silos = merge(silos[:i], silos[i + 1 :], ignore_index=True)
        # overlapping columns num
        overlapping_col_num = round(other_silos.shape[1] * overlapping_rate)
        select_cols = rand_select_col(
            other_silos, overlapping_col_num, random_state=seed
        )
        silo = merge(silo, select_cols, ignore_index=True)
        overlapping_silo_lst.append(silo)
        overlapping_num_lst.append(overlapping_col_num)
    return overlapping_silo_lst, overlapping_num_lst


def rename_dataset(dataset):
    # 确保dataset是一个DataFrame对象
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("The dataset must be a pandas DataFrame")

    # 获取数据集的列数
    num_features = dataset.shape[1]

    # 创建新的列名列表
    new_column_names = ["feature_" + str(i + 1) for i in range(num_features)]

    # 更换列名为新的列名列表
    dataset.columns = new_column_names

    return dataset


def add_noise(df, noise_col_num=None, seed=0, noise_size=0.5, noise_type="uniform"):
    """
    add noise to data
    :param df: dataframe
    :param noise_col_num: noise column num
    :param noise_size: how big the noise
    :param seed: random seed
    :param noise_type: gauss or uniform
    :return data with noise
    """
    np.random.seed(seed)
    row_num = df.shape[0]
    # noise_row_num = int(row_num * noise_rate)
    noise_row_num = row_num
    if noise_col_num is None:
        noise_col_num = df.shape[1]
    # 生成噪声
    if noise_type == "uniform":
        noise = np.random.uniform(
            -noise_size,
            noise_size,
            size=(noise_row_num, noise_col_num),
        )
    elif noise_type == "gauss":
        noise = np.random.normal(0, noise_size, size=(noise_row_num, noise_col_num))
    else:
        raise ValueError("noise not implemented")
    # 这里默认在最后添加噪声
    noised_df = df.iloc[0:noise_row_num, -noise_col_num:] + noise
    return noised_df


def get_seed_by_time():
    # 获取当前时间的微秒部分
    seed = int((time.time() * 1000000) % 1000000)
    return seed


def add_multiple_noise(noise_size, df, seed):
    """
    add multiple noise
    """
    if noise_size == 0:
        return df
    noise_rate = [0.2, 0.2, 0.4]
    uniform_factor = 5
    gauss_factor = 0.1
    rows = df.shape[0]
    seed = get_seed_by_time()
    indices = np.arange(df.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    row_1 = int(rows * noise_rate[0])
    row_2 = int(rows * noise_rate[1])
    row_3 = int(rows * noise_rate[2])
    indice_1 = indices[:row_1]
    indice_2 = indices[row_1 : row_1 + row_2]
    indice_3 = indices[row_1 + row_2 : row_1 + row_2 + row_3]
    indice_4 = indices[row_1 + row_2 + row_3 :]
    df_1 = df.iloc[indice_1]
    df_2 = df.iloc[indice_2]
    df_3 = df.iloc[indice_3]
    df_4 = df.iloc[indice_4]

    # 40% 添加均值为-noise,noise的均值分布噪声
    df_1 = add_noise(
        df_1, seed=seed, noise_size=noise_size * uniform_factor, noise_type="uniform"
    )
    # 20% 添加方差为noise的高斯分布噪声
    df_2 = add_noise(
        df_2, seed=seed, noise_size=noise_size * gauss_factor, noise_type="gauss"
    )
    # 20% 添加方差为noise*0.1的高斯分布噪声
    df_3 = add_noise(
        df_3, seed=seed, noise_size=noise_size * gauss_factor * 0.1, noise_type="gauss"
    )
    noisy_df = merge(df_1, df_2, df_3, df_4, axis=0)
    # reset index
    noisy_df = noisy_df.reset_index()
    return noisy_df
