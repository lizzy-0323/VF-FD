"""
@Author: laziyu
@Date:2024-6-14
@Description: utils for model train
"""

import pandas as pd
import torch
from data_loader.data_operation import merge, rand_select_col
from data_loader.preprocess import rename_dataset
from utils import distance
from torch.utils.data import TensorDataset
import numpy as np

NUM_CLIENTS = 4
THRESHOLD = 0.2
MODEL_PATH = "./model/trainer/run"
DATASET_PATH = "./data/"
LR = 0.1
BATCH_SIZE = 2048
EPOCHS = 10
SEED = 0
TEST_SIZE = 0.25


def switch_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    print("Running on:", device)
    return device


def preprocess_data(x, y, device):
    # 将 x 和 y 转换为 numpy 数组
    x_array = x.values
    y_array = y.values
    # # 标准化
    # x_array_scaled = mm.fit_transform(x_array)
    # y_array_scaled = mm.fit_transform(y_array)
    # 转换为 PyTorch Tensor
    x_tensor = torch.from_numpy(x_array).float().to(device)
    y_tensor = torch.from_numpy(y_array).float().to(device)
    # 创建 TensorDataset
    tensor_dataset = TensorDataset(x_tensor, y_tensor)
    return tensor_dataset


def partition_data(data):
    """
    partition data into overlapping data and non overlapping data
    """
    # BUG: 有同名同数据列时，貌似有bug
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The input data must be a pandas DataFrame.")
    # Identify overlapping columns (columns with count > 1)
    column_counts = data.columns.value_counts()
    # overlapping_column_name = column_counts[column_counts > 1].index.tolist()
    non_overlapping_column_name = column_counts[column_counts == 1].index.tolist()
    # Each sublist contains the same overlapping column names
    overlapping_df_lst = [
        data[col] * (count - 1) for col, count in column_counts.items() if count > 1
    ]
    # Identify non-overlapping columns (columns with count equal to 1)
    non_overlapping_columns = data[non_overlapping_column_name]
    num_df = overlapping_df_lst[0].shape[1]
    num_columns_in_df = len(overlapping_df_lst)
    # Create a list of lists for overlapping columns
    overlapping_column_lst = [None] * num_df
    for overlapping_df in overlapping_df_lst:
        for column_index in range(num_df):
            # 提取第column_index列
            column = overlapping_df.iloc[:, column_index]
            # 将提取的列作为新的DataFrame添加到new_df_list的对应位置
            overlapping_column_lst[column_index] = merge(
                overlapping_column_lst[column_index], column
            )

    return overlapping_column_lst, non_overlapping_columns


def bin_cut(arr):
    """
    bin cut for array
    """
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # 按照bins对arr进行分箱
    arr_transformed = pd.cut(arr, bins, labels=False, include_lowest=True)
    return arr_transformed


def partition_data_by_default(data):
    return [], data


def partition_data_by_random(data, m, n):
    """
    partition by random
    """
    data = rename_dataset(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The input data must be a pandas DataFrame.")
    num_columns = data.shape[1]
    non_overlapping_columns = data.copy()
    if m * n > num_columns:
        raise ValueError(
            "The product of m and n cannot exceed the number of columns in the data."
        )

    overlapping_groups = []
    # 随机选择m个特征组，每个组n个特征
    for _ in range(m):
        # 随机选择n列
        selected_columns = rand_select_col(non_overlapping_columns, n)
        # 创建新的DataFrame，只包含选中的列
        overlapping_groups.append(selected_columns)
        # 从原始DataFrame中删除选中的列
        non_overlapping_columns.drop(selected_columns.columns, axis=1, inplace=True)
    return overlapping_groups, non_overlapping_columns


def partition_data_by_emd(data, threshold=THRESHOLD, metric="emd"):
    """
    partition data by emd
    """
    data = rename_dataset(data)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The input data must be a pandas DataFrame.")
    # 计算所有列对之间的EMD
    num_columns = data.shape[1]
    overlapping_column_pairs = []
    non_overlapping_columns = data.columns.tolist()

    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            col1 = data.iloc[:, i]
            col2 = data.iloc[:, j]
            emd = distance(bin_cut(col1), bin_cut(col2), metric=metric)
            if emd < threshold:
                # 如果EMD小于阈值，则认为这两列是重叠的
                overlapping_column_pairs.append((i, j))
                # 从非重叠列列表中移除这两列
                if col1.name in non_overlapping_columns:
                    non_overlapping_columns.remove(col1.name)
                if col2.name in non_overlapping_columns:
                    non_overlapping_columns.remove(col2.name)
    """inner function"""

    def find_connected_components(edges):
        """
        Find connected components in a graph represented by edges.
        """
        from collections import defaultdict, deque

        # 初始化图
        graph = defaultdict(set)
        for u, v in edges:
            graph[u].add(v)
            graph[v].add(u)

        # 访问标记数组
        visited = set()
        components = []

        # 遍历所有节点，寻找连通分量
        for node in graph:
            if node not in visited:
                component = []
                queue = deque([node])
                visited.add(node)

                while queue:
                    current = queue.popleft()
                    component.append(current)
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                components.append(component)

        return components

    group = find_connected_components(overlapping_column_pairs)
    overlapping_column_lst = [
        data.iloc[:, overlapping_column] for overlapping_column in group
    ]
    non_overlapping_columns = data[non_overlapping_columns]
    return overlapping_column_lst, non_overlapping_columns


def partition_data_by_distance_and_name(data, threshold=THRESHOLD):
    # Identify overlapping columns (columns with count > 1)
    column_counts = data.columns.value_counts()
    # overlapping_column_name = column_counts[column_counts > 1].index.tolist()
    non_overlapping_column_name = column_counts[column_counts == 1].index.tolist()
    # Each sublist contains the same overlapping column names
    overlapping_df_lst = [
        data[col] for col, count in column_counts.items() if count > 1
    ]
    # Identify non-overlapping columns (columns with count equal to 1)
    non_overlapping_columns = data[non_overlapping_column_name]
    num_df = overlapping_df_lst[0].shape[1]
    num_columns_in_df = len(overlapping_df_lst)
    # Create a list of lists for overlapping columns
    overlapping_column_lst = [None] * num_df
    for overlapping_df in overlapping_df_lst:
        for column_index in range(num_df):
            # 提取第column_index列
            column = overlapping_df.iloc[:, column_index]
            # 将提取的列作为新的DataFrame添加到new_df_list的对应位置
            overlapping_column_lst[column_index] = merge(
                overlapping_column_lst[column_index], column
            )
    for col_1 in non_overlapping_columns.columns:
        for column_index, overlapping_column in enumerate(overlapping_column_lst):
            for col_2 in overlapping_column.columns:
                # 计算earth mover's distance
                # 为了避免col_1已经不存在于non_overlapping_columns中，这里进行判断
                if col_1 not in non_overlapping_columns.columns:
                    break
                emd = distance(
                    bin_cut(overlapping_column[col_2].values),
                    bin_cut(non_overlapping_columns[col_1].values),
                )
                if emd < threshold:
                    # 添加新的列
                    new_column = overlapping_column.copy()
                    new_column[col_2] = non_overlapping_columns[col_1]
                    overlapping_column_lst.append(new_column)
                    non_overlapping_columns.drop(col_1, axis=1, inplace=True)
    return overlapping_column_lst, non_overlapping_columns


def add_noise_to_tensor(tensor, noise_size=0.1, seed=0, noise_type="gauss"):
    # 设置随机数生成器的种子，如果提供了seed参数
    if seed is not None:
        torch.manual_seed(seed)
    if noise_type == "gauss":
        noise = torch.randn_like(tensor) * noise_size
    elif noise_type == "uniform":
        noise = torch.rand_like(tensor) * noise_size
    else:
        raise NotImplementedError("Noise type not implement")
    return tensor + noise


def add_mutiple_noise_to_tensor(tensor, noise_size, seed):
    tensor = add_noise_to_tensor(tensor, noise_size, seed, noise_type="gauss")
    tensor = add_noise_to_tensor(tensor, noise_size, seed, noise_type="uniform")
    return tensor
