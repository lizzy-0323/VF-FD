"""
@Author: laziyu
@Date:2024-2-14
@Description: overlapping column selection by filter method
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, pearsonr
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from data_loader.dataset import load_dataset
from data_loader.file_reader import save_df_to_csv
from data_loader.preprocess import add_multiple_noise, make_overlapping


def calculate_score(col, df, method="pearson"):
    """
    calculate pearson score
    :param col: column
    :param df: df
    :return: average pearson correlation score
    """
    scores = []
    for column in df.columns:
        if method == "pearson":
            pearson_corr, _ = np.nan_to_num(
                pearsonr(col, df[column], alternative="greater")
            )
            # print(pearson_corr)
            scores.append(pearson_corr)
        elif method == "mi":
            # 如果是label数据
            if df.shape[1] == 1:
                mi_score = mutual_info_classif(col.values.reshape(-1, 1), df[column])
            else:
                mi_score = mutual_info_regression(col.values.reshape(-1, 1), df[column])
            scores.append(mi_score)
        elif method == "chi2":
            # bins
            col_binned = pd.cut(col, bins=4, include_lowest=True)
            df_column_binned = pd.cut(df[column], bins=4, include_lowest=True)
            # Create a contingency table
            contingency_table = pd.crosstab(col_binned, df_column_binned)
            chi2_score, _, _, _ = chi2_contingency(contingency_table)
            scores.append(chi2_score)
        elif method == "anova":
            col_array = col.values.reshape(-1, 1)
            # Perform ANOVA test and append the F-value to the scores list
            f_val, _ = f_oneway(*[col_array] + [df[column].values.reshape(-1, 1)])
            scores.append(f_val)
        # elif method == "mic":
        #     # 信息值: 不确定逻辑是否正确
        #     mine = MINE()
        #     mine.compute_score(col.values, df[column].values)
        #     mic_score = mine.mic()
        #     scores.append(mic_score)
    avg_score = np.mean(scores)
    return avg_score


def get_score_list(
    candidate_lst, non_overlapping_columns, overlapping_column_names, method="mi"
):
    """
    get score list
    :param: candidate_lst : list of dataframe
    :param: non_overlapping_columns: dataframe
    :param: overlapping_column_names: list
    :param: method: overlapping calculation method
    :return: score_list: score list of each column
    """
    score_list = []
    for df in candidate_lst:
        col_score = {}
        for col in df.columns:
            if col in overlapping_column_names:
                if method == "stats":
                    col_score[col] = 1 - np.var(df[col])
                elif method in ("mi", "pearson", "chi2", "anova", "mic"):
                    col_score[col] = calculate_score(
                        df[col], non_overlapping_columns, method
                    )
                else:
                    raise ValueError("method not implement")
        score_list.append(col_score)
    return score_list


def choose_columns_by_random(candidate_lst, overlapping_columns):
    """
    choose columns by random
    :param overlapping_columns: overlapping columns
    :return: chosen columns
    """
    overlapping_candidates = {}
    for col in overlapping_columns:
        overlapping_candidates[col] = []
        for i, df in enumerate(candidate_lst):
            client_id = i + 1
            if col in df.columns:
                overlapping_candidates[col].append(client_id)
    chosen_columns = {}
    for col, candidates in overlapping_candidates.items():
        if candidates:
            chosen_silo = random.choice(candidates)
            chosen_columns[col] = chosen_silo
    return chosen_columns


def choose_columns_by_score(score_list):
    """
    calculate overlapping choices
    :param score_list: list contains each silo's score in overlapping columns
    :return: overlapping selection dict
    """
    id_dict = {}
    for i, silo in enumerate(score_list):
        client_id = i + 1
        for key, value in silo.items():
            current_info = id_dict.get(key, {"score": 0, "id": client_id})
            if value > current_info["score"]:
                id_dict[key] = {"score": value, "id": client_id}
    # 生成各个silo中的overlapping列的选择结果
    return {key: value["id"] for key, value in id_dict.items()}


def choose_columns(non_overlapping_columns, candidate_lst, overlapping_columns, method):
    """
    choose overlapping columns by filter method
    using non overlapping columns and overlapping columns
    """
    result = {}
    if method == "random":
        result = choose_columns_by_random(candidate_lst, overlapping_columns)
    else:
        score_lst = get_score_list(
            candidate_lst, non_overlapping_columns, overlapping_columns, method
        )
        result = choose_columns_by_score(score_lst)
    return result


def select(columns, candidate_lst, method):
    """
    select overlapping columns by filter method
    using label or non overlapping features to count
    """
    acc = 0
    result = pd.DataFrame()
    overlapping_columns = candidate_lst[0].columns
    total_columns = candidate_lst[0].shape[1]
    correct_column = 0
    choose_dict = choose_columns(columns, candidate_lst, overlapping_columns, method)
    for column_name, client_id in choose_dict.items():
        if client_id == 1:
            correct_column += 1
        client_index = client_id - 1
        selected_client = candidate_lst[client_index]
        selected_column = selected_client.loc[:, column_name]
        result = pd.concat([result, selected_column], axis=1)
    acc = correct_column / total_columns
    # print(f"Filter Selection Method: {method}, Primary selection acc: {acc}")
    return acc, result


# def make_train_data(X, y_lst, method="mi"):
#     """
#     make train data by filter method
#     :param: selection_method
#     """
#     combined_data = pd.DataFrame()
#     overlapping_columns = y_lst[0].columns
#     result = choose_columns(X, y_lst, overlapping_columns, method)
#     result = reverse_dict(result)
#     # print(result)
#     for client_index, column_lst in result.items():
#         # client_index - 1:  means the index
#         selected_client = y_lst[client_index - 1]
#         selected_columns = selected_client.loc[:, column_lst]
#         combined_data = pd.concat([combined_data, selected_columns])
#     return combined_data


def reverse_dict(result):
    """
    reverse the selection dict
    """
    reverse_result = {}
    # 遍历原始字典
    for column_name, client_index in result.items():
        # 如果客户端键不存在，则创建它
        if client_index not in reverse_result:
            reverse_result[client_index] = []

        # 将列名添加到客户端的列表中
        reverse_result[client_index].append(column_name)
    return reverse_result


def run(args):
    X, y = load_dataset(args.dataset)
    noise_sizes = [0, 0.5, 1, 1.5, 2]
    non_overlapping_data, overlapping_data = make_overlapping(
        X, args.overlapping_rate, noise_sizes
    )
    # 用overlapping的数据去 拟合 non-overlapping的数据
    # 添加均匀分布和正态分布的噪声
    overlapping_data_lst = [
        add_multiple_noise(ns, overlapping_data, args.seed) for ns in noise_sizes
    ]
    # 计算所有overlapping的数据和噪声数据，选择的结果
    acc, selection_result = select(
        non_overlapping_data, overlapping_data_lst, args.method
    )
    column_lst = overlapping_data.columns
    selection_result = selection_result[column_lst]
    print(non_overlapping_data.shape)
    print(selection_result.shape)
    print(overlapping_data.shape)
    print(f"acc: {acc}")
    dataset_path = args.save_dataset_path + args.dataset + "/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    X_filename = (
        dataset_path
        + "x"
        + "_"
        + args.dataset
        + "_"
        + str(args.overlapping_rate)
        + ".csv"
    )
    y_filename = (
        dataset_path
        + "y"
        + "_"
        + args.dataset
        + "_"
        + str(args.overlapping_rate)
        + ".csv"
    )
    gc_filename = (
        dataset_path
        + "gc"
        + "_"
        + args.dataset
        + "_"
        + str(args.overlapping_rate)
        + ".csv"
    )
    save_df_to_csv(non_overlapping_data, X_filename)
    save_df_to_csv(selection_result, y_filename)
    save_df_to_csv(overlapping_data, gc_filename)
    # result = (args.overlapping_rate, args.method, acc)
    # result = (args.overlapping_rate, args.dataset, acc)
    # save_tuple_to_csv(result, args.save_result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="knn", help="dataset choice")
    parser.add_argument(
        "--overlapping_rate", type=float, default=0.2, help="overlapping rate"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--method", type=str, default="mi", help="method")
    parser.add_argument("--client_num", type=int, default=4, help="client num")
    parser.add_argument(
        "--save_dataset_path",
        type=str,
        default="../dataset/",
        help="path to save the combined dataset selected by filter method",
    )
    parser.add_argument(
        "--save_result_path",
        type=str,
        default="./result/primary_selection_result.csv",
        help="path to save primary selection result",
    )
    args = parser.parse_args()
    print(args)
    run(args)


if __name__ == "__main__":
    main()
