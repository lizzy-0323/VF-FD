"""
@Author: laziyu
@Date:2023-3-11
@Description: rpc server
"""

import random
import time
from concurrent import futures
from math import ceil, sqrt
from test import test_overlapping

import grpc
from data_loader.data_operation import (align, get_row_intersection, merge,
                                        overlapping)
from grpc_service.rpc import (client_rpc_pb2, client_rpc_pb2_grpc,
                              server_rpc_pb2, server_rpc_pb2_grpc)
from model.server import Server
from utils.dataframe_to_protobuf import df_to_protobuf, protobuf_to_df
from utils.log import init_logger


class RpcServer(Server, server_rpc_pb2_grpc.ServerServiceServicer):
    def __init__(self, cfg, client_cfg, num_clients) -> None:
        super().__init__(num_clients, cfg.distance_metric)
        self.host = cfg.host
        self.port = cfg.port
        self.algorithm = cfg.algorithm
        self.k = cfg.k
        self.overlapping_cal_method = cfg.overlapping_cal_method
        self.mode = cfg.mode
        self.gc_path = cfg.gc_path
        self.logger = init_logger("server", cfg.log_path)
        self.communication_cost = 0
        self.client_cfg = client_cfg
        self.candidate_list = self._gen_candidate_list()

    def _increase_communication_cost(self):
        self.communication_cost += 1

    def _get_client_stub(self, client_id):
        """
        get client stub by id
        :param: client_id: client id
        :return: client stub: client stub for grpc communication
        """
        host = self.client_cfg["client_" + str(client_id)]["host"]
        port = self.client_cfg["client_" + str(client_id)]["port"]
        client_address = str(host) + ":" + str(port)
        client_channel = grpc.insecure_channel(client_address)
        client_stub = client_rpc_pb2_grpc.ClientServiceStub(client_channel)
        return client_stub

    def get_client_column_dict(self, request, context):
        client_column_dict = {}
        if not self.silo_column_dict:
            self._get_data_sample_from_clients()
            for i in range(self.num_clients):
                client_id = i + 1
                self._update_columns_select_dict(self.data_cache[i], client_id)
        for key, values in self.silo_column_dict.items():
            # 创建 DataArray 实例并添加 values
            data_array = server_rpc_pb2.DataArray()
            data_array.values.extend(values)
            # 将 DataArray 实例添加到 client_column_dict 中
            client_column_dict[int(key)] = data_array
        return server_rpc_pb2.client_column_dict_response(
            client_column_dict=client_column_dict
        )

    def _get_client_dist_metric(self, client_id):
        """
        get client dist metric
        :param: client_id
        :return: distance metric in client
        """
        client_stub = self._get_client_stub(client_id)
        request = client_rpc_pb2.dist_metric_request(client_id=client_id)
        response = client_stub.get_dist_metric(request)
        return response.dist_metric

    def _get_data_sample_from_clients(self):
        """get data sample"""
        # use dict to avoid concurrent error
        data_dict = {}
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            request = client_rpc_pb2.data_sample_request(
                client_id=client_id, sample_num=200
            )
            response = client_stub.get_data_sample(request)
            data_sample = response.dataframe
            df_format_data_sample = protobuf_to_df(data_sample)
            data_dict[client_id] = df_format_data_sample
            self.logger.info(f"Get data sample from client {client_id}")
        # id alignment
        reordered_data = [data_dict[i] for i in range(1, self.num_clients + 1)]
        self.data_cache = get_row_intersection(reordered_data)
        self._increase_communication_cost()

    def _check_clients_dist_metric(self):
        """
        check if clients distance metric is the same
        """
        metric_list = []
        for i in range(self.num_clients):
            client_id = i + 1
            dist_metric = self._get_client_dist_metric(client_id=client_id)
            metric_list.append(dist_metric)
        if len(set(metric_list)) != 1:
            self.logger.error("Clients metric is not consistent")
        else:
            self.logger.info("Check clients metric consistency ready")

    def _get_id_list_from_clients(self):
        """
        get id list from clients
        """
        id_from_clients = []
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            request = client_rpc_pb2.id_request(client_id=client_id)
            response = client_stub.get_id_list(request)
            id_from_clients.append(response.id_list)
        self._increase_communication_cost()
        self.id_from_clients = id_from_clients

    def _id_alignment(self):
        """
        send id alignment order
        1. get id from clients
        2. psi
        """
        self._get_id_list_from_clients()
        self._psi()

    def _psi(self, status=1):
        """
        psi
        """
        # 1. 调度并规划
        current_round = 0
        total_round = self._gen_psi_total_round()
        original_candidate_list = self.candidate_list
        # 2. 向client发起请求
        # 3. 接受client的该轮次psi status,检查当前轮次是否是最终轮次
        while current_round < total_round:
            current_candidate_list = original_candidate_list.copy()
            new_candidate_list = []
            while len(current_candidate_list) > 1:
                candidate1, candidate2 = self._get_candidate_pair(
                    current_candidate_list
                )
                self.logger.info(
                    f"Psi round {current_round+1}, client {candidate1} psi with client {candidate2}"
                )
                request = client_rpc_pb2.send_psi_order_request(
                    client_id=candidate1, partner_id=candidate2, status=status
                )
                client_stub = self._get_client_stub(candidate1)
                response = client_stub.send_psi_order(request)
                if response.status != 1:
                    self.logger.error(f"Psi round {current_round+1} failed")
                else:
                    new_candidate_list.append(candidate1)
                time.sleep(0.5)
            current_round += 1
            # update candidate list
            original_candidate_list = new_candidate_list
        # 4. 如果是最终轮次，向对应的client发起请求，拿到psi的最终结果
        result = self._get_psi_result_from_client(candidate1)
        # 5. 把最终结果发送给所有client
        for client_id in self.candidate_list:
            # except the result sender client
            if client_id == candidate1:
                continue
            self._send_psi_request_to_client(client_id, result)
        self.logger.info("psi finished")

    def _get_candidate_pair(self, candidate_list):
        """random get two candidate and remove them in candidate list"""
        selected_candidates = random.sample(candidate_list, 2)
        candidate_list.remove(selected_candidates[0])
        candidate_list.remove(selected_candidates[1])
        return selected_candidates

    def _gen_candidate_list(self):
        """gen candidate list"""
        return [i + 1 for i in range(self.num_clients)]

    def _gen_psi_total_round(self):
        """gen psi total round"""
        return ceil(sqrt(self.num_clients))

    def _send_psi_request_to_client(self, client_id, result):
        """
        send psi request to client
        :param: result: psi result
        :return: response
        """
        client_stub = self._get_client_stub(client_id)
        request = client_rpc_pb2.send_psi_result_request(
            client_id=client_id, psi_result=result
        )
        response = client_stub.send_psi_result(request)
        if response.status != 1:
            self.logger.error(f"client {client_id} get psi result faild")

    def _get_psi_result_from_client(self, client_id):
        """
        get psi result from client
        """
        client_stub = self._get_client_stub(client_id=client_id)
        request = client_rpc_pb2.get_psi_result_request(client_id=client_id)
        response = client_stub.get_psi_result(request)
        result = response.id_list
        return result

    def _get_overlapping_selection(self):
        """
        get overlapping selection dict
        """
        if not self.data_cache:
            self._get_data_sample_from_clients()
        overlapping_columns = self._get_overlapping_columns()
        non_overlapping_data = self._get_non_overlapping_data(
            overlapping_columns=overlapping_columns
        )
        self._choose_columns(
            non_overlapping_data, overlapping_columns, self.overlapping_cal_method
        )
        self.logger.info("Overlapping selection finished")
        for i in range(self.num_clients):
            client_id = i + 1
            self._update_columns_select_dict(self.data_cache[i], client_id)
        if self.mode == "test":
            self.logger.info("Overlapping test: ")
            acc = test_overlapping(
                self.num_clients, self.silo_column_dict, self.gc_path
            )
            self.logger.info(f"Accuracy: {acc}")
        self._increase_communication_cost()

    def query(self, query_request):
        """
        federated query base
        :param query request: query request
        :return query result
        """
        data_indices = []
        order_indices = []
        for i in range(self.num_clients):
            client_id = i + 1
            client_stub = self._get_client_stub(client_id)
            request = client_rpc_pb2.query_request(
                client_id=client_id, query=query_request
            )
            response = client_stub.get_single_query_result(request)
            order_list, data = response.order_list, protobuf_to_df(
                response.query_result
            )
            data = self._cols_alignment(data, client_id)
            data_indices.append(data)
            order_indices.append(order_list)
        if self.algorithm == "fagin":
            result = self.fagin_algorithm(data_indices, order_indices, self.k)
            self.logger.info("get query result, algorithm: fagin")
            return result
        if self.algorithm == "threshold":
            result = self.threshold_algorithm(data_indices, order_indices, self.k)
            self.logger.info("get query result, algorithm: threshold")
            return result
        raise ValueError("algorithm not exists")

    def single_query(self, request, context):
        """
        single federated knn query
        :param request: request
        :param context: context
        :return: single query response in protobuf format
        """
        result = self.query(request.query)
        self._increase_communication_cost()
        return server_rpc_pb2.single_query_response(query_result=df_to_protobuf(result))

    def batch_query(self, request, context):
        """
        batch federated knn query
        :param request: request
        :param context: context
        :return: list
        """
        query_lst = request.query_list
        result_lst = []
        for query_df in query_lst:
            result = self.query(query_df)
            result_lst.append(df_to_protobuf(result))
        self._increase_communication_cost()
        return server_rpc_pb2.batch_query_response(query_result=result_lst)

    def get_server_config(self, request, context):
        config = {}
        config["overlapping_cal_method"] = self.overlapping_cal_method
        return server_rpc_pb2.server_config_response(config=config)

    def serve(self):
        """
        start server serve
        """
        self.logger.info(f"Server init at: {self.host}:{self.port}")
        self.logger.info(f"Overlapping calculate method: {self.overlapping_cal_method}")
        self.logger.info(f"Query method: {self.algorithm}")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        server_rpc_pb2_grpc.add_ServerServiceServicer_to_server(self, server)
        server.add_insecure_port(f"{self.host}:{self.port}")
        server.start()
        self._id_alignment()
        self._get_overlapping_selection()
        # self._check_clients_dist_metric()
        server.wait_for_termination()
