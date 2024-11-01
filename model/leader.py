"""
@Author: laziyu
@Date:2023-1-24
@Description: leader node
"""

from concurrent import futures

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

import grpc
from data_loader.data_operation import get_row_intersection, overlapping
from grpc_service import (client_rpc_pb2, client_rpc_pb2_grpc, leader_rpc_pb2,
                          leader_rpc_pb2_grpc)
from utils import init_logger, protobuf_to_df
from utils.dataframe_to_protobuf import df_to_protobuf


class Leader(leader_rpc_pb2_grpc.LeaderServiceServicer):
    def __init__(self, cfg, client_cfg_dict, num_clients) -> None:
        """
        :param cfg: leader config file
        :param client_cfg_dict: client host and port dict
        :param num_clients: client num
        """
        self.host = cfg.host
        self.port = cfg.port
        self.num_clients = num_clients
        self.client_cfg_dict = client_cfg_dict
        self.ids = []
        self.id_from_clients = []
        self.data_cache = []
        self.num_set = set()
        self.communication_cost = 0
        self.logger = init_logger("leader", log_path=cfg.log_path)

    def _increase_communication_cost(self):
        self.communication_cost += 1

    def send_id_alignment_order(self, request, context):
        self._get_id_list_from_clients()
        self._psi()
        self._update_clients_ids()
        return leader_rpc_pb2.id_alignment_response(status="success")

    def get_data_cache(self, request, context):
        """get data cache"""
        if self.data_cache == []:
            raise ValueError("leader not have data cache yet")
        return leader_rpc_pb2.data_cache_response(
            data_cache=[df_to_protobuf(data) for data in self.data_cache]
        )

    def _get_client_stub(self, client_id):
        """
        get client stub by id
        :param client_id: client id
        :return: client stub
        """
        host = self.client_cfg_dict[client_id]["host"]
        port = self.client_cfg_dict[client_id]["port"]
        client_address = str(host) + ":" + str(port)
        client_channel = grpc.insecure_channel(client_address)
        client_stub = client_rpc_pb2_grpc.ClientServiceStub(client_channel)
        return client_stub

    def _get_id_list_from_clients(self):
        """
        get id list from clients
        """
        id_from_clients = []
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            response = client_stub.get_id_list(
                client_rpc_pb2.id_request(client_id=client_id)
            )
            id_from_clients.append(response.id_list)
        self._increase_communication_cost()
        self.id_from_clients = id_from_clients

    def _get_overlapping_columns(self):
        """
        get overlapping columns
        :return: overlapping columns
        """
        column_list = []
        if not self.data_cache:
            self._get_data_sample_from_clients()
        for data in self.data_cache:
            columns = data.columns.tolist()
            column_list.append(columns)
        overlapping_columns = overlapping(column_list)
        self._increase_communication_cost()
        return overlapping_columns

    def _choose_columns_by_mi(self, mi_list):
        """
        calculate overlapping choices
        :param silo_mi_list: mi list contains each silo's mi in overlapping columns
        :return: overlapping selection dict
        """
        id_dict = {}
        for i, silo in enumerate(mi_list):
            client_id = i + 1
            for key, value in silo.items():
                current_info = id_dict.get(key, {"mi": 0, "id": client_id})
                if value > current_info["mi"]:
                    id_dict[key] = {"mi": value, "id": client_id}
        # 生成各个silo中的overlapping列的选择结果
        return {key: value["id"] for key, value in id_dict.items()}

    def _get_overlapping_selection(self):
        """
        get overlapping selection dict
        """
        overlapping_columns = self._get_overlapping_columns()
        non_overlapping_data = self._get_non_overlapping_data(
            overlapping_columns=overlapping_columns
        )
        mi_list = self._get_mi_list(non_overlapping_data, overlapping_columns)
        overlapping_selection_dict = self._choose_columns_by_mi(mi_list)
        return overlapping_selection_dict

    def get_overlapping_selection(self, request, context):
        """
        get overlapping selection, used by server
        """
        overlapping_selection_dict = self._get_overlapping_selection()
        return leader_rpc_pb2.overlapping_selection_response(
            overlapping_selection_dict=overlapping_selection_dict
        )

    def _get_mi_list(self, non_overlapping_data, overlapping_columns):
        """
        get mutual information list in silos
        :param non_overlapping_data: a data silo made of non overlapping columns
        :param overlapping_columns: overlapping columns
        :return: mi list in each silo
        """
        mi_list = []
        for silo in self.data_cache:
            silo_mi = {}
            for col in silo.columns:
                # 如果是overlaping列
                if col in overlapping_columns:
                    # 计算互信息
                    silo_mi[col] = self._calculate_mi(silo[col], non_overlapping_data)
                    # print(silo_mi[col])
            mi_list.append(silo_mi)
        return mi_list

    def _get_non_overlapping_data(self, overlapping_columns):
        """
        get non overlapping cols
        :param overlapping_columns: overlapping columns
        :return: a merged silo made of non overlapping cols
        """
        non_overlapping_data = pd.DataFrame()
        if not self.data_cache:
            self._get_data_sample_from_clients()
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

    def _calculate_mi(self, col, silo):
        """
        calculate mutual information between col and silo
        :param col: column
        :param silo: silo
        :return: average mutual information estimated by knn estimator
        """
        return np.mean(
            mutual_info_regression(silo.values, col.values.reshape(-1, 1).ravel())
        )

    def _psi(self):
        """
        private set intersection
        """
        n = len(self.id_from_clients)
        client_random_list = []
        psi_result = []
        for client_id_list in self.id_from_clients:
            # 添加到集合中
            self.num_set = self.num_set.union(set(client_id_list))
        # 得到一个总的id集合
        num_list = list(self.num_set)
        # 对集合中的每个元素，随机生成n个数字,保证n个数的和是0
        # 初始化一个映射，用于记录每个client收到的随机数
        for i in range(self.num_clients):
            client_random_list.append([])
        for num in num_list:
            random_list = np.random.randint(-100000, 100000, n)
            # 最后一个数是前n-1个数的相反数
            random_list[-1] = -sum(random_list[:-1])
            for i, client_id_list in enumerate(self.id_from_clients):
                if num in client_id_list:
                    # 发给client生成的随机数
                    client_random_list[i].append(random_list[i])
                else:
                    # 发给这个client一个其他的随机数,保证和之前的不一样
                    old_random_num = random_list[i]
                    new_random_num = np.random.randint(-100000, 100000)
                    while old_random_num == new_random_num:
                        new_random_num = np.random.randint(-100000, 100000)
                    client_random_list[i].append(new_random_num)

        psi_result = self._get_psi_result_from_clients(client_random_list)
        self.ids = psi_result

    def _get_data_sample_from_clients(self):
        """
        get data sample
        """
        data_cache = []
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            response = client_stub.get_data_sample(
                client_rpc_pb2.data_sample_request(client_id=client_id, sample_num=200)
            )
            data_sample = response.dataframe
            df_format_data_sample = protobuf_to_df(data_sample)
            data_cache.append(df_format_data_sample)
            self.logger.info(f"get data sample from client {client_id}")
        # id alignment
        self.data_cache = get_row_intersection(data_cache)
        self._increase_communication_cost()

    def _get_column_list_from_clients(self):
        """
        get columns list from clients for calculate overlapping columns
        :return column list in each client
        """
        column_list = []
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            response = client_stub.get_column_list(
                client_rpc_pb2.column_request(client_id=client_id)
            )
            column_list.append(response.column_list)
            self.logger.info(f"get client columns from client {client_id}")
        self._increase_communication_cost()
        return column_list

    def _update_clients_ids(self):
        """
        update clients ids with leader alignment ids
        """
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            response = client_stub.update_id_list(
                client_rpc_pb2.update_id_list_request(
                    client_id=client_id, id_list=self.ids
                )
            )
            self.logger.info(f"update id list for client {client_id}")
        self._increase_communication_cost()

    def _get_psi_result_from_clients(self, client_random_list):
        """
        get psi result from clients
        :param client_random_list: random list for each client
        :return psi id alignment result list
        """
        psi_result_list = []
        id_align_list = []
        num_list = list(self.num_set)
        for i in range(self.num_clients):
            psi_result_list.append([])
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            response = client_stub.get_psi_result(
                client_rpc_pb2.psi_request(
                    client_id=client_id, id_list=client_random_list[i]
                )
            )
            psi_result_list[i] = response.id_list
            self.logger.info(f"get psi result from {client_id}")
        for i in range(len((psi_result_list[0]))):
            total = sum(lst[i] for lst in psi_result_list)
            if total == 0:
                id_align_list.append(num_list[i])
        self._increase_communication_cost()
        return id_align_list

    def serve(self):
        """
        start leader serve
        """
        self.logger.info("leader init")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        leader_rpc_pb2_grpc.add_LeaderServiceServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")
        server.start()
        server.wait_for_termination()
