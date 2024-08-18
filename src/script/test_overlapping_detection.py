import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(os.getcwd())
from data_loader.dataset import load_dataset_from_csv
from data_loader.preprocess import rename_dataset, split_dataset
from model.trainer.trainer_utils import partition_data_by_emd


def rename(feature_name):
    return feature_name[8:]


def test_emd(data, threshold=1e-2):
    data = rename_dataset(data)
    overlapping_feature_group, non_overlapping_features = partition_data_by_emd(
        data, threshold
    )
    print(len(overlapping_feature_group))
    label_map = {}
    # count = 0
    for i, overlapping_features in enumerate(overlapping_feature_group):
        label_map[i] = [
            rename(feature) for feature in overlapping_features.columns.tolist()
        ]
        # count += 1
    non_overlapping_columns = non_overlapping_features.columns.tolist()
    label_map[i + 1] = [
        rename(non_overlapping_columns[i])
        for i in range(0, len(non_overlapping_columns))
    ]
    print(label_map)
    return label_map


def plot(label_map):
    # 创建两个空列表，一个用于存储特征ID，一个用于存储类别编号
    feature_ids = []
    category_numbers = []

    # 遍历label_map，将特征ID和类别编号添加到相应的列表中
    for feature_id, categories in label_map.items():
        # 将特征ID转换为整数
        feature_ids.append(feature_id)
        # 如果类别是列表，我们取列表的第一个元素作为该特征的类别编号
        if isinstance(categories, list):
            category_number = int(categories[0])
        else:
            # 如果类别不是列表，直接转换为整数
            category_number = int(categories)
        category_numbers.append(category_number)

    # 创建散点图
    plt.figure(figsize=(12, 6))  # 设置图形的大小

    # 设置图形的标题和坐标轴标签
    plt.title("Feature ID vs Category Number")
    plt.xlabel("Feature ID")
    plt.ylabel("Category Number")

    # 优化x轴的显示，使其更加清晰
    plt.xticks(range(min(feature_ids), max(feature_ids) + 1))

    # 显示图形
    plt.show()
    plt.savefig("cluster.png")


if __name__ == "__main__":
    client_num = 5
    threshold = 1e-2
    data = load_dataset_from_csv("./data/sonar/all_features.csv")
    label_map = test_emd(data, threshold)
    plot(label_map)
