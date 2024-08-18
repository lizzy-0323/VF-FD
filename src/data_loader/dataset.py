"""
@Description: load data
@Author: laziyu
@Date: 2024-5-14
"""

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
    make_friedman1,
)

from data_loader.preprocess import (
    make_overlapping,
    map_char_to_numeric,
    preprocess_dataset,
    standardize,
)

SEED = 0


def load_dataset_from_csv(path, ignore_header=False):
    """
    load dataset from csv
    :param path: dataset path
    :return: dataset
    """
    df = pd.read_csv(path, header=None if ignore_header else "infer")
    return df


def load_label(dataset_name):
    """
    load label
    """
    yaml_path = os.path.join("./conf", "dataset", dataset_name + ".yaml")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    path = cfg.get("path")
    target = cfg.get("target", None)
    ignore_header = cfg.get("ignore_header", False)
    df = load_dataset_from_csv(path, ignore_header)
    df = preprocess_dataset(df)
    # 如果没有target， 选择最后一列作为target
    if target is None or target not in df.columns:
        target = df.columns[-1]
    y = df[target]
    return y


def load_dataset(dataset_name):
    """
    load dataset
    :param dataset_name: dataset name
    """
    # print(f"Dataset Name: {dataset_name}")
    yaml_path = os.path.join("./conf", "dataset", dataset_name + ".yaml")
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    path = cfg.get("path")
    target = cfg.get("target", None)
    ignore_header = cfg.get("ignore_header", False)
    df = load_dataset_from_csv(path, ignore_header)
    df = preprocess_dataset(df)
    # 如果没有target， 选择最后一列作为target
    if target is None or target not in df.columns:
        target = df.columns[-1]
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y


def download_dataset(dataset="boston_housing"):
    """
    download dataset
    :param dataset: dataset name
    :return: dataset
    """
    print(f"Download {dataset}")
    # if dataset == "boston_housing":
    #     data = load_boston()
    if dataset == "wine":
        data = load_wine()
    elif dataset == "breast_cancer":
        data = load_breast_cancer()
    elif dataset == "iris":
        data = load_iris()
    else:
        raise ValueError("dataset not found")
    x = data.data
    y = data.target
    # 给数据集添加列名，再加上target
    feature_names = np.append(data.feature_names, "target")
    # 整合数据
    complete_dataset = pd.DataFrame(
        np.concatenate([x, y.reshape(-1, 1)], axis=1),
        columns=feature_names,
    )
    return complete_dataset


def print_dataset_info(dataset):
    print("===============")
    print("Dataset Info:")
    if dataset is not None and isinstance(dataset, pd.DataFrame):
        print("Number of rows:", len(dataset))
        print("Number of columns:", len(dataset.columns))
    else:
        print("Invalid dataset provided. Please provide a valid DataFrame.")


def load_overlapping_dataset(dataset_name, noise_sizes, overlapping_rate=0.2, seed=0):
    X, _ = load_dataset(dataset_name)
    dataset = make_overlapping(
        X, noise_sizes=noise_sizes, overlapping_rate=overlapping_rate, seed=seed
    )
    return dataset
