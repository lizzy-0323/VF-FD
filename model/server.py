"""
@Author: laziyu
@Date:2023-2-14
@Description: server node
"""

from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from data_loader.data_operation import align, merge, overlapping
from model.overlapping_selection import (choose_columns_by_random,
                                         choose_columns_by_score,
                                         get_score_list)
from utils.const import FLOAT_BYTES, HOMOMORPHIC_BYTES, PER_ENCRYPT_TIME
from utils.distance_operation import (get_dist_by_id, get_dist_by_row,
                                      sum_all_distance)


class Server:
    def __init__(self, num_clients, distance_metric) -> None:
        """
        :param cfg: server config
        :param leader_cfg_dict: leader host and port config
        :param client_cfg_dict: clients host and port config
        :param num_clients: clients num
        """
        self.num_clients = num_clients
        self.overlapping_select_dict = {}
        self.id_from_clients = []
        self.ids = []
        self.distance_metric = distance_metric
        self.data_cache = []
        self.num_set = set()
        self.silo_column_dict = {}

    def _cols_alignment(self, data, client_id):
        """
        col alignment
        :param: data
        :param: client_id
        :return: data
        """
        self._update_columns_select_dict(data, client_id)
        selected_cols = self.silo_column_dict[client_id]
        data = data[selected_cols]
        data = data.fillna(0)
        return data

    def _get_overlapping_columns(self):
        """
        get overlapping columns
        :return: overlapping columns
        """
        column_list = []
        for data in self.data_cache:
            columns = data.columns.tolist()
            column_list.append(columns)
        overlapping_columns = overlapping(column_list)
        return overlapping_columns

    def _get_non_overlapping_data(self, overlapping_columns):
        """
        get non overlapping cols
        :param overlapping_columns: overlapping columns
        :return: a merged silo made of non overlapping cols
        """
        non_overlapping_data = pd.DataFrame()
        for data in self.data_cache:
            non_overlapping_data = pd.concat(
                [
                    non_overlapping_data,
                    data.drop(
                        data.columns.intersection(overlapping_columns).tolist(), axis=1
                    ),
                ],
                axis=1,
            )
        return non_overlapping_data

    def _choose_columns(self, non_overlapping_data, overlapping_columns, method):
        """
        calculate overlapping choices
        :param non_overlapping_data: a data silo made of non overlapping columns
        :param overlapping_columns: overlapping columns
        :param method : overlapping calculate method
        """
        if method == "random":
            self.overlapping_select_dict = choose_columns_by_random(
                self.data_cache, overlapping_columns
            )
        else:
            score_list = get_score_list(
                self.data_cache,
                non_overlapping_data,
                overlapping_columns,
                method,
            )
            # print(score_list)
            self.overlapping_select_dict = choose_columns_by_score(score_list)

    def fagin_algorithm(self, data_indices, order_indices, k):
        """
        fagin algorithm
        :param data_indices: data list
        :param order_indices: order list
        :param k: neighbors
        :return: top-k candidates
        """
        id_counter = defaultdict(int)
        id_result = []
        for indices in order_indices:
            for idx in indices:
                id_counter[idx] += 1
                if id_counter[idx] == self.num_clients:
                    id_result.append(idx)
                if len(id_result) == k:
                    result = (
                        merge(
                            sum_all_distance(
                                align(data_indices),
                                type=(
                                    "add"
                                    if self.distance_metric == "l2"
                                    else "multiple"
                                ),
                            )
                        )
                        .reindex(id_result)
                        .sort_values("distance", ascending=self.distance_metric == "l2")
                    )
                    distance_column = result.pop("distance")
                    result["distance"] = distance_column
                    return result
        # if can not found enough result
        print("can not found enough result")
        result = merge(align(data_indices))
        return result

    def threshold_algorithm(self, data_indices, order_indices, k):
        """
        average threshold algorithm
        :param: data_indices: data list
        :param: order_indices: order list
        :param: k: neighbors
        :return: top-k candidates
        """
        candidates = OrderedDict()
        rows = len(data_indices[0])
        for row in range(rows):
            threshold = np.prod(np.array(get_dist_by_row(data_indices, row)))
            # print(f"threshold: {threshold} in row {row}")
            for indices in order_indices:
                # 查看当前的id
                idx = indices[row]
                if candidates.get(idx) is None:
                    candidates[idx] = np.prod(
                        np.array(get_dist_by_id(data_indices, idx))
                    )
            # 如果candidate的数量大于k
            if len(candidates) >= k:
                # 按照candidate的距离降序
                candidates = OrderedDict(
                    sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                )
                # 比较当前阈值和candidate的距离,计算有多少个candidate的距离大于等于阈值
                count = sum(
                    (1 for _, distance in candidates.items() if distance >= threshold)
                )
                if count >= k:
                    index_list = list(candidates.keys())[:k]
                    result = (
                        merge(
                            sum_all_distance(
                                align(data_indices),
                                type=(
                                    "multiple"
                                    if self.distance_metric == "l2"
                                    else "add"
                                ),
                            )
                        )
                        .reindex(index_list)
                        .sort_values("distance", ascending=self.distance_metric == "l2")
                    )
                    distance_column = result.pop("distance")
                    result["distance"] = distance_column
                    return result
        # if can not found enough result
        print("can not found enough result")
        result = merge(align(data_indices))
        return result

    def _update_columns_select_dict(self, data, client_id):
        """
        update data's columns list
        :param data : data
        :param client_id : client_id
        """
        for key, value in self.overlapping_select_dict.items():
            if key in data.columns and value != client_id:
                data.drop(key, axis=1, inplace=True)
        self.silo_column_dict[client_id] = data.columns.tolist()

    # def _psi_with_ss(self):
    #     """
    #     private set intersection
    #     """
    #     n = len(self.id_from_clients)
    #     client_random_list = []
    #     psi_result = []
    #     for client_id_list in self.id_from_clients:
    #         # 添加到集合中
    #         self.num_set = self.num_set.union(set(client_id_list))
    #     # 得到一个总的id集合
    #     num_list = list(self.num_set)
    #     # 对集合中的每个元素，随机生成n个数字,保证n个数的和是0
    #     # 初始化一个映射，用于记录每个client收到的随机数
    #     for i in range(self.num_clients):
    #         client_random_list.append([])
    #     for num in num_list:
    #         random_list = np.random.randint(-100000, 100000, n)
    #         # 最后一个数是前n-1个数的相反数
    #         random_list[-1] = -sum(random_list[:-1])
    #         for i, client_id_list in enumerate(self.id_from_clients):
    #             if num in client_id_list:
    #                 # 发给client生成的随机数
    #                 client_random_list[i].append(random_list[i])
    #             else:
    #                 # 发给这个client一个其他的随机数,保证和之前的不一样
    #                 old_random_num = random_list[i]
    #                 new_random_num = np.random.randint(-100000, 100000)
    #                 while old_random_num == new_random_num:
    #                     new_random_num = np.random.randint(-100000, 100000)
    #                 client_random_list[i].append(new_random_num)

    #     psi_result = self._get_psi_result_from_clients(client_random_list)
    #     self.ids = psi_result
