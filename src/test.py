"""
@Author: laziyu
@Date:2023-2-14
@Description: query interface 
"""

import argparse

from google.protobuf.empty_pb2 import Empty

import grpc
from data_loader import get_dataset, read_dataset_from_csv
from grpc_service.rpc import server_rpc_pb2, server_rpc_pb2_grpc
from utils import df_to_protobuf
from utils.dataframe_to_protobuf import protobuf_to_df


def get_server_stub(host, port):
    """
    get server stub
    :host: server host
    :port: server port
    """
    server_address = str(host) + ":" + str(port)
    server_channel = grpc.insecure_channel(server_address)
    server_stub = server_rpc_pb2_grpc.ServerServiceStub(server_channel)
    return server_stub


def reorder_columns(query_columns, result):
    """
    reorder columns, used for getting same order with query
    :param: query columns
    :result: query result: dataframe
    """
    return result[query_columns.tolist() + ["distance"]]


def calculate_overlapping_acc(dict1, dict2):
    # 确保两个字典具有相同的键，使用集合操作取交集
    common_keys = set(dict1.keys()) & set(dict2.keys())
    total_values = 0
    correct_values = 0

    # 对于每个键，比较两个字典中该键对应的值
    for key in common_keys:
        values1 = dict1[key]
        values2 = dict2[key]

        # 统计每个键下值的相同数量
        common_elements_count = sum(1 for element in values1 if element in values2)
        correct_values += common_elements_count
        total_values += len(values1)
    # print("incorrect value num:", total_values - correct_values)
    # 计算准确率
    accuracy = correct_values / total_values if total_values > 0 else 0

    return accuracy


def test_overlapping(num_clients, column_dict, gc_path):
    """
    overlapping test
    """
    ground_truth_dict = {}
    # print("reading ground truth file")
    for i in range(num_clients):
        client_id = i + 1
        filename = gc_path + "ground_truth_client_" + str(client_id) + ".csv"
        gc = read_dataset_from_csv(filename)
        ground_truth_dict[client_id] = gc.columns.tolist()
    acc = calculate_overlapping_acc(ground_truth_dict, column_dict)
    return acc


def single_query_test(args):
    """
    single query test
    """
    server_stub = get_server_stub(args.host, args.port)
    dataset = get_dataset(args.dataset)
    data_sample = dataset.sample(1)
    print(data_sample)
    columns = data_sample.columns
    query = df_to_protobuf(data_sample)
    response = server_stub.single_query(
        server_rpc_pb2.single_query_request(query=query)
    )
    query_result = reorder_columns(columns, protobuf_to_df(response.query_result))
    print(query_result)


def batch_query_test(args):
    """
    batch query test
    """
    server_stub = get_server_stub(args.host, args.port)
    dataset = get_dataset(args.dataset)
    data_samples = [dataset.sample(1, replace=True) for _ in range(10)]
    print([index for data in data_samples for index in data.index.tolist()])
    columns = data_samples[0].columns
    protobuf_data_samples = [
        df_to_protobuf(data_sample) for data_sample in data_samples
    ]
    request = server_rpc_pb2.batch_query_request(query_list=protobuf_data_samples)
    response = server_stub.batch_query(request)
    # 处理响应
    query_results = [
        reorder_columns(columns, protobuf_to_df(result))
        for result in response.query_result
    ]
    print(query_results)


def get_server_config(args):
    """
    get server config
    """
    server_stub = get_server_stub(args.host, args.port)
    request = Empty()
    response = server_stub.get_server_config(request)
    print("server config: ", response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_num", type=int, default=1, help="query num")
    parser.add_argument(
        "--gc_path", type=str, default="./data/", help="ground truth file path"
    )
    parser.add_argument("--client_num", type=int, default=4, help="client num")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--k", type=int, default=10, help="number of neighbors")
    parser.add_argument("--dataset", type=str, default="", help="dataset choice")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=str, default=50055, help="server port")
    parser.add_argument("--query_type", type=str, default="single", help="query type")
    args = parser.parse_args()
    print(args)
    run(args)


def run(args):
    get_server_config(args)
    if args.query_type == "single":
        single_query_test(args)
    elif args.query_type == "batch":
        batch_query_test(args)
    else:
        raise ValueError("wrong query type")


if __name__ == "__main__":
    main()
