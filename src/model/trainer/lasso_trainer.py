"""
@Author: laziyu
@Date:2024-6-14
@Description: lasso model for feature selection
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from data_loader.data_operation import merge
from data_loader.dataset import load_dataset, load_label
from data_loader.file_reader import read_dataset_from_csv, save_df_to_csv
from data_loader.preprocess import normalize, split_dataset, standardize
from model.trainer.trainer_utils import (
    partition_data,
    partition_data_by_emd,
    NUM_CLIENTS,
)

DATASET_PATH = "./data/"
THRESHOLD = 1e-8
SEED = 0


def select_features(all_features, dataset_name, method="global"):
    """
    feature selection by lasso strategy
    """
    if method == "global":
        lasso = Lasso(alpha=0.01, random_state=SEED)
        y = load_label(dataset_name)
        # print(y)
        lasso.fit(all_features, y)
        coef = lasso.coef_
        # print(f"Coef: {coef}")
        mask = np.where(np.abs(coef) > THRESHOLD)[0]
        # mask = np.where(np.abs(coef) != 0)[0]
        selected_features = all_features.iloc[:, mask]
        return selected_features
    if method == "local":
        # 常规的local lasso
        selected_features = pd.DataFrame()
        y = load_label(dataset_name)
        feature_lst = split_dataset(all_features, NUM_CLIENTS)
        for client_feature in feature_lst:
            lasso = Lasso(alpha=0.1, random_state=0)
            lasso.fit(client_feature, y)
            coef = lasso.coef_
            mask = np.where(np.abs(coef) > THRESHOLD)[0]
            client_selected_features = client_feature.iloc[:, mask]
            selected_features = merge(selected_features, client_selected_features)
        return selected_features
    if method == "emd":
        selected_features = pd.DataFrame()
        y = load_label(dataset_name)
        overlapping_column_lst, non_overlapping_columns = partition_data_by_emd(
            all_features
        )
        # print(feature_lst)
        # print(len(feature_lst))

        for overlapping_feature in overlapping_column_lst:
            lasso = Lasso(alpha=0.1, random_state=0)
            lasso.fit(overlapping_feature, y)
            coef = lasso.coef_
            mask = np.where(np.abs(coef) > THRESHOLD)[0]
            client_selected_features = overlapping_feature.iloc[:, mask]
            selected_features = merge(selected_features, client_selected_features)
        selected_features = merge(selected_features, non_overlapping_columns)
        return selected_features
    raise NotImplementedError("Method not implement!")


def run(args):
    dataset_path = os.path.join(
        args.dataset_path, args.dataset_name, "all_features.csv"
    )
    result_path = os.path.join(
        args.result_path,
        args.dataset_name,
        args.method + "_lasso_selected_features.csv",
    )
    all_features = read_dataset_from_csv(dataset_path)
    # all_features = normalize(all_features)
    selected_features = select_features(all_features, args.dataset_name, args.method)
    save_df_to_csv(selected_features, result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="knn", help="dataset name")
    parser.add_argument(
        "--dataset_path", type=str, default=DATASET_PATH, help="path to load dataset"
    )
    parser.add_argument("--method", type=str, default="group", help="lasso type")
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
