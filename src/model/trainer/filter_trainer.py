"""
@Author: laziyu
@Date: 2024-6-26
@Description: feature selector using local anova
"""

import argparse
import os
import numpy as np
import pandas as pd
from data_loader.data_operation import rand_select_col, merge
from data_loader.dataset import load_dataset, load_label
from data_loader.preprocess import normalize
from model.overlapping_selection import select
from data_loader.file_reader import read_dataset_from_csv, save_df_to_csv
from model.trainer.trainer_utils import partition_data, partition_data_by_emd

DATASET_PATH = "./data/"
SEED = 0


def select_features(all_features, dataset_name, method, use_emd=False):
    """
    feature selection by local anova
    """
    all_features = normalize(all_features)
    if use_emd:
        overlapping_column_lst, non_overlapping_columns = partition_data_by_emd(
            all_features
        )
    else:
        overlapping_column_lst, non_overlapping_columns = partition_data(all_features)
    label = load_label(dataset_name)
    label = label.to_frame()
    _, overlapping_selection_result = select(
        label, overlapping_column_lst, method=method
    )
    selected_features = merge(overlapping_selection_result, non_overlapping_columns)
    return selected_features


def run(args):
    dataset_path = os.path.join(
        args.dataset_path, args.dataset_name, "all_features.csv"
    )
    if args.use_emd:
        result_path = os.path.join(
            args.result_path, args.dataset_name, "filter_plus_selected_features.csv"
        )
    else:
        result_path = os.path.join(
            args.result_path, args.dataset_name, "filter_selected_features.csv"
        )
    all_features = read_dataset_from_csv(dataset_path)
    selected_features = select_features(
        all_features, args.dataset_name, args.method, args.use_emd
    )
    save_df_to_csv(selected_features, result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="knn", help="dataset name")
    parser.add_argument("--method", type=str, default="anova", help="filter method")
    parser.add_argument(
        "--dataset_path", type=str, default=DATASET_PATH, help="path to load dataset"
    )
    parser.add_argument(
        "--use_emd", type=bool, default=False, help="similarity detection using emd"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=DATASET_PATH,
        help="path to save result",
    )
    args = parser.parse_args()
    # print(args)
    run(args)


if __name__ == "__main__":
    main()
