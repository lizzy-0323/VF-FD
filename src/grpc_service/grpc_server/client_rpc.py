"""
@Author: laziyu
@Date:2023-3-11
@Description: rpc client
"""

from concurrent import futures

import grpc
from data_loader.file_reader import read_client_cfg, read_dataset_from_csv
from encryption.diffie_hellman import (decrypt, double_encrypt, generate_keys,
                                       get_encrypt_intersection,
                                       hash_and_encrypt)
from grpc_service import client_rpc_pb2, client_rpc_pb2_grpc
from model.client import Client
from utils import df_to_protobuf, init_logger, protobuf_to_df


class RpcClient(Client, client_rpc_pb2_grpc.ClientServiceServicer):
    def __init__(self, client_cfg, client_id) -> None:
        """
        :param cfg: client config file
        :param silo: dataframe
        :param client_id: client id
        """
        self.client_id = client_id
        self.client_cfg = client_cfg
        # 0: not align 1: aligned
        self.psi_status = 0
        self.cfg = client_cfg["client_" + str(client_id)]
        self.silo = read_dataset_from_csv(self.cfg.dataset_path)
        self.host = self.cfg.host
        self.port = self.cfg.port
        self.logger = init_logger(
            "client_" + str(self.client_id), log_path=self.cfg.log_path
        )
        self.communication_cost = 0
        super().__init__(self.silo, self.cfg.distance_metric)

    def _increase_communication_cost(self):
        self.communication_cost += 1

    def send_psi_order(self, request, context):
        """
        server send psi order to client
        """
        assert request.client_id == self.client_id
        # 0 出错 1 对齐 2 强行对齐
        if request.status == 1 and self.psi_status == 1:
            self.logger.info(
                f"Client {request.client_id} already finished id alignment"
            )
        else:
            self.logger.info(
                f"Client {request.client_id} send psi order with client {request.partner_id}"
            )
            self._psi_with_partner(request.partner_id)
        self._increase_communication_cost()
        return client_rpc_pb2.send_psi_order_response(
            client_id=self.client_id, status=1
        )

    # def send_encrypted_id_list(self, request, context):
    #     """
    #     other client send encrypted id list
    #     """
    #     self.other_encrypted_id_list = request.encrypted_id_list
    #     # print(self.other_encrypted_id_list)
    #     return client_rpc_pb2.send_encrypted_id_response(
    #         client_id=self.client_id, status=0
    #     )

    def _send_encrypted_id_list(self, partner_id, encrypted_data):
        client_stub = self._get_client_stub(partner_id)
        request = client_rpc_pb2.send_encrypted_id_request(
            client_id=self.client_id, encrypted_id_list=encrypted_data
        )
        response = client_stub.send_encrypted_id_list(request)
        return response

    def _send_double_encrypted_id_list(self, partner_id, double_encrypted_data):
        client_stub = self._get_client_stub(partner_id)
        request = client_rpc_pb2.send_double_encrypted_id_request(
            client_id=self.client_id, double_encrypted_id_list=double_encrypted_data
        )
        response = client_stub.send_double_encrypted_id_list(request)
        return response

    def _psi_with_partner(self, partner_id):
        """
        psi with partner
        :param: partner id
        """
        # 1. gen keys and encrypt local id
        self.ek, other_key = generate_keys()
        local_encrypted_data = hash_and_encrypt(self.ids, self.ek)
        response = self._get_encrypted_id_list(partner_id, other_key)
        other_encrypted_data = response.encrypted_id_list
        # 2. double encrypted other client 's id
        response = self._get_double_encrypted_id_list(partner_id, local_encrypted_data)
        other_double_encrypted_data = response.double_encrypted_id_list
        local_double_encrypted_data = double_encrypt(other_encrypted_data, self.ek)
        response = self._send_double_encrypted_id_list(
            partner_id, local_double_encrypted_data
        )
        if response.status != 0:
            self.logger.error("double encrypted data send failed...")
        # 3. get psi result
        intersection = get_encrypt_intersection(
            local_double_encrypted_data, other_double_encrypted_data
        )
        try:
            result = decrypt(self.ids, intersection, other_double_encrypted_data)
            # 更新id
            self.ids = result
        except:
            raise ValueError(f"client {self.client_id } wrong")

    def _get_double_encrypted_id_list(self, partner_id, encrypted_data):
        client_stub = self._get_client_stub(partner_id)
        request = client_rpc_pb2.get_double_encrypted_id_request(
            client_id=self.client_id, encrypted_id_list=encrypted_data
        )
        response = client_stub.get_double_encrypted_id_list(request)
        return response

    def get_double_encrypted_id_list(self, request, context):
        encrypted_data = request.encrypted_id_list
        double_encrypted_data = double_encrypt(encrypted_data, self.ek)
        return client_rpc_pb2.get_double_encrypted_id_response(
            client_id=self.client_id, double_encrypted_id_list=double_encrypted_data
        )

    def _get_encrypted_id_list(self, partner_id, other_key):
        """
        get encrypted id from other client
        :param: partner_id
        :param: other_key: key for other's encryption
        """
        request = client_rpc_pb2.get_encrypted_id_request(
            client_id=self.client_id, ek=other_key
        )
        client_stub = self._get_client_stub(partner_id)
        response = client_stub.get_encrypted_id_list(request)
        return response

    def get_encrypted_id_list(self, request, context):
        """
        get encrypted id list
        """
        self.ek = request.ek
        encrypted_data = hash_and_encrypt(self.ids, self.ek)
        response = client_rpc_pb2.get_encrypted_id_response(
            client_id=self.client_id, encrypted_id_list=encrypted_data
        )
        return response

    def send_double_encrypted_id_list(self, request, context):
        """
        other client send double encrypted id list
        """
        self.other_double_encrypted_id_list = request.double_encrypted_id_list
        return client_rpc_pb2.send_double_encrypted_id_response(
            client_id=self.client_id, status=0
        )

    def _get_client_stub(self, client_id, cfg_path="./config.yaml"):
        """
        get client stub by id
        :param: client_id: client id
        :return: client stub: client stub for grpc communication
        """
        other_cfg = self.client_cfg["client_" + str(client_id)]
        host = other_cfg["host"]
        port = other_cfg["port"]
        client_address = str(host) + ":" + str(port)
        client_channel = grpc.insecure_channel(client_address)
        client_stub = client_rpc_pb2_grpc.ClientServiceStub(client_channel)
        return client_stub

    def get_single_query_result(self, request, context):
        """
        get single query result
        :param request: request
        :param context: context
        :return: query response
        """
        client_id = request.client_id
        query = protobuf_to_df(request.query)
        order_list, query_result = self._query(query)
        query_result = df_to_protobuf(query_result)
        self._increase_communication_cost()
        self.logger.info(f"client {self.client_id} get query result")
        return client_rpc_pb2.query_response(
            client_id=client_id,
            order_list=order_list,
            query_result=query_result,
        )

    def get_id_list(self, request, context):
        """
        get id list
        :param request: request
        :param context: context
        :return id response
        """
        # TODO:假设已经加密
        self._increase_communication_cost()
        client_id = request.client_id
        id_list = self._gen_id_list()
        return client_rpc_pb2.id_response(client_id=client_id, id_list=id_list)

    def get_psi_result(self, request, context):
        """
        get psi result from final round leader
        :param request: request
        :param context: context
        :return: psi result response
        """
        self._increase_communication_cost()
        self.psi_status = 1
        client_id = request.client_id
        id_list = self.ids
        return client_rpc_pb2.get_psi_result_response(
            client_id=client_id, id_list=id_list
        )

    def get_column_list(self, request, context):
        """
        get column list
        """
        self._increase_communication_cost()
        client_id = request.client_id
        column_list = self.silo.columns.tolist()
        return client_rpc_pb2.column_response(
            client_id=client_id, column_list=column_list
        )

    def get_data_sample(self, request, context):
        """
        get data sample
        :param request: request
        :param context: context
        :return: data sample
        """
        sample_num = min(len(self.silo), request.sample_num)
        client_id = request.client_id
        data_sample = self.silo.sample(sample_num)
        df_msg = df_to_protobuf(data_sample)
        return client_rpc_pb2.data_sample_response(
            client_id=client_id, dataframe=df_msg
        )

    def send_psi_result(self, request, context):
        """
        send psi result
        """
        psi_result = list(request.psi_result)
        self._increase_communication_cost()
        self._update_index(psi_result)
        self.logger.info(f"client {self.client_id} get psi result")
        self.psi_status = 1
        return client_rpc_pb2.send_psi_result_response(
            client_id=self.client_id, status=self.psi_status
        )

    def update_id_list(self, request, context):
        """
        update id list
        :param request: request
        :param context: context
        :return: update id list response
        """
        new_id_list = list(request.id_list)
        self._increase_communication_cost()
        self._update_index(new_id_list)
        self.logger.info(f"client {self.client_id} update index")
        return client_rpc_pb2.update_id_list_response(client_id=self.client_id)

    def get_dist_metric(self, request, context):
        """
        get distance metric
        :param request: request
        :param context: context
        :return: distance metric
        """
        client_id = self.client_id
        self._increase_communication_cost()
        return client_rpc_pb2.dist_metric_response(
            client_id=client_id, dist_metric=self.distance_metric
        )

    def serve(self):
        """
        start client serve
        """
        self.logger.info(f"client {self.client_id} init at {self.host}:{self.port}")
        client = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        client_rpc_pb2_grpc.add_ClientServiceServicer_to_server(self, client)
        client.add_insecure_port(f"{self.host}:{self.port}")
        client.start()
        client.wait_for_termination()


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="config.yaml", help="config file")
#     parser.add_argument(
#         "--algorithm",
#         type=str,
#         default="threshold",
#         choices=["naive", "threshold", "fagin"],
#         help="algorithm choice",
#     )
#     parser.add_argument(
#         "--metric",
#         type=str,
#         default="l2",
#         choices=["l2", "cosine"],
#         help="distance metric",
#     )
#     parser.add_argument("--seed", type=int, default=0, help="random seed")
#     parser.add_argument("--host", type=str, default=IP, help="client ip")
#     parser.add_argument("--port", type=int, default=50051, help="client port")
#     parser.add_argument("--client_id", type=int, default=1, help="client id")
#     parser.add_argument("--k", type=int, default=K, help="number of neighbors")
#     parser.add_argument(
#         "--dataset_path",
#         type=str,
#         default="./data/client_1.csv",
#         help="dataset choice",
#     )
#     parser.add_argument(
#         "--query_num", type=int, default=QUERY_NUM, help="number of queries"
#     )
#     parser.add_argument("--row_num", type=int, default=ROW_NUM, help="number of rows")
#     args = parser.parse_args()
#     print(args)
#     run(args)


# def run(args):
#     """
#     run client
#     :param args: arguments
#     """
#     silo = read_dataset_from_csv(args.dataset_path)
#     client = Client(
#         silo,
#         args.host,
#         args.port,
#         args.client_id,
#         args.metric,
#         log_path="./log/client_1.txt",
#     )
#     client.serve()


# if __name__ == "__main__":
#     main()
