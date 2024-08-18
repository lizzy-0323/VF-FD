"""
@Author: laziyu
@Date:2024-6-15
@Description: random feature selector
"""

import argparse
import os
import random
import pandas as pd
from torch import seed
from data_loader.data_operation import rand_select_col, merge
from data_loader.file_reader import read_dataset_from_csv, save_df_to_csv
from data_loader.preprocess import split_dataset
from model.trainer.trainer_utils import (
    partition_data,
    partition_data_by_emd,
    NUM_CLIENTS,
)

NUM_FEATURES = 1
DATASET_PATH = "./data/"
SEED = 0


def select_features(all_features, use_emd=False):
    """
    feature selection by random strategy
    """
    selected_features = pd.DataFrame()
    if use_emd:
        overlapping_column_lst, non_overlapping_columns = partition_data_by_emd(
            all_features
        )
        for overlapping_column in overlapping_column_lst:
            selected_feature = rand_select_col(overlapping_column, NUM_FEATURES, SEED)
            selected_features = merge(selected_features, selected_feature)
        selected_features = merge(selected_features, non_overlapping_columns)
        return selected_features
    # 常规的random
    all_feature_lst = split_dataset(all_features, NUM_CLIENTS)
    for client_feature in all_feature_lst:
        select_num = random.randint(0, client_feature.shape[1])
        selected_feature = rand_select_col(client_feature, select_num)
        selected_features = merge(selected_features, selected_feature)
    return selected_features


def run(args):
    dataset_path = os.path.join(
        args.dataset_path, args.dataset_name, "all_features.csv"
    )
    if args.use_emd:
        result_path = os.path.join(
            args.result_path, args.dataset_name, "emd_random_selected_features.csv"
        )
    else:
        result_path = os.path.join(
            args.result_path, args.dataset_name, "random_selected_features.csv"
        )
    all_features = read_dataset_from_csv(dataset_path)
    selected_features = select_features(all_features, args.use_emd)
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
    parser.add_argument("--use_emd", type=bool, default=False, help="use emd")
    args = parser.parse_args()
    # print(args)
    run(args)


if __name__ == "__main__":
    main()
