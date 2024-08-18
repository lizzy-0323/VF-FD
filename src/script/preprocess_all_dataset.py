"""
@Author: laziyu
@Date:2024-6-19
@Description: script for preprocessing all the dataset
"""

import sys
import os
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.append(os.getcwd())
from data_loader.dataset import load_overlapping_dataset
from data_loader.file_reader import save_df_to_csv
from data_loader.preprocess import preprocess_dataset, rename_dataset

NOISE_SIZES = [1, 2, 3, 4]
OVERLAPPING_RATE = 0.2
cfg_file_path = "./conf/downstream_task.yaml"

with open(cfg_file_path, "r") as file:
    cfg = yaml.safe_load(file)
dataset_lst = cfg.get("dataset_lst", [])


def preprocess_clean_dataset():
    sonar_file_name = "./data/sonar/clean_features.csv"
    knn_file_name = "./data/knn/clean_features.csv"
    sonar_dataset = pd.read_csv(sonar_file_name, header=None)
    sonar_dataset = preprocess_dataset(sonar_dataset)
    knn_dataset = pd.read_csv(knn_file_name)
    knn_dataset = preprocess_dataset(knn_dataset)
    save_df_to_csv(sonar_dataset, sonar_file_name)
    save_df_to_csv(knn_dataset, knn_file_name)


def preprocess_physionNet(task=1):
    task = task + 46
    train = pd.read_csv("../dataset/PhysioNet_train.csv")
    valid = pd.read_csv("../dataset/PhysioNet_valid.csv")
    test = pd.read_csv("../dataset/PhysioNet_test.csv")
    train = pd.concat([train, valid])
    train = pd.concat([train, test])
    train = train.reset_index(drop=True)
    data = train.iloc[:, 2:43]
    target = train.iloc[:, task]
    data = pd.concat([data, target], axis=1)
    save_df_to_csv(data, "../dataset/PhysioNet.csv")


def preprocess_activity():
    data = pd.read_csv("../dataset/activity.csv")
    data = rename_dataset(data)
    save_df_to_csv(data, "../dataset/activity-clean.csv")


def preprocess_sylva():
    # 读取数据文件
    data = pd.read_csv(
        "../dataset/sylva_train.data", sep=" ", header=None
    )  # 修改分隔符为实际的分隔符
    labels = pd.read_csv(
        "../dataset/sylva_train.labels", sep=" ", header=None, names=["label"]
    )  # 修改分隔符为实际的分隔符
    # 打印数据集的shape
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    # 合并数据和标签
    data["label"] = labels
    # 存储为CSV文件
    data.to_csv("../dataset/sylva.csv", index=False)


def preprocess_all_dataset():
    for dataset_name in tqdm(dataset_lst, desc="Preprocessing datasets"):
        df = load_overlapping_dataset(
            dataset_name,
            noise_sizes=NOISE_SIZES,
            overlapping_rate=OVERLAPPING_RATE,
            seed=0,
        )
        dataset_dir = os.path.join("./data", dataset_name)
        filename = dataset_dir + "/all_features.csv"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        save_df_to_csv(df, filename)
    return


if __name__ == "__main__":
    # preprocess_sylva()
    preprocess_clean_dataset()
    # preprocess_activity()
    # preprocess_physionNet()
    preprocess_all_dataset()
