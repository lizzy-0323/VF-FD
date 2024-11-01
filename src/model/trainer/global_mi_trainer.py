"""
@Author: laziyu
@Date: 2024-6-24
@Description: feature selector using mutual information
"""

import argparse
import os
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)

from data_loader.data_operation import merge
from data_loader.dataset import load_dataset, load_label
from data_loader.file_reader import read_dataset_from_csv, save_df_to_csv
from data_loader.preprocess import normalize, split_dataset
from model.trainer.lasso_trainer import NUM_CLIENTS

SEED = 0
DATASET_PATH = "./data/"


def select_features(all_features, dataset_name):
    """
    feature selection by mutual information
    """
    y = load_label(dataset_name)
    data_lst = split_dataset(all_features, silo_num=4)
    selected_features = pd.DataFrame()
    for data in data_lst:
        selector = SelectKBest(mutual_info_classif, k=1)
        selector.fit(data, y)
        selected_indices = selector.get_support(indices=True)
        selected_feature = all_features.iloc[:, selected_indices]
        selected_features = merge(selected_features, selected_feature)
    return selected_features


def run(args):
    dataset_path = os.path.join(
        args.dataset_path, args.dataset_name, "all_features.csv"
    )
    result_path = os.path.join(
        args.result_path, args.dataset_name, "global_mi_selected_features.csv"
    )
    all_features = read_dataset_from_csv(dataset_path)
    selected_features = select_features(all_features, args.dataset_name)
    save_df_to_csv(selected_features, result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="knn", help="dataset name")
    parser.add_argument(
        "--dataset_path", type=str, default=DATASET_PATH, help="path to load dataset"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=DATASET_PATH,
        help="path to save result",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
