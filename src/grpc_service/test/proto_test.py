# 调用../protos/leader_and_client_pb2.py
import os
import sys

import grpc
from grpc_service.rpc import leader_and_client_pb2, leader_and_client_pb2_grpc


def rpc_rule_test():
    s = leader_and_client_pb2.IdRequest()
    s.client_id = "233"
    print(s.SerializeToString())


def client_test():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = leader_and_client_pb2_grpc.ClientServiceStub(channel)
        response = stub.GetIdList(leader_and_client_pb2.IdRequest(client_id="233"))
        print(response)
        # response = stub.GetQueryList(
        #     leader_and_client_pb2.QueryRequest(client_id="233", query="233")
        # )
        # print(response)


if __name__ == "__main__":
    # 生成一个请求
    rpc_rule_test()
    client_test()
