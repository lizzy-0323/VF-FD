import grpc
from grpc_service.rpc import client_rpc_pb2, client_rpc_pb2_grpc


def test_client_id_request():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = client_rpc_pb2_grpc.ClientServiceStub(channel)
        client_id = 1
        request = client_rpc_pb2.id_request(client_id=client_id)
        response = stub.get_id_list(request)
        print(response.id_list)


def test_client_query_request():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = client_rpc_pb2_grpc.ClientServiceStub(channel)
        client_id = 1
        query = "query"
        response = client_rpc_pb2.query_request(client_id=client_id, query=query)


if __name__ == "__main__":
    test_client_id_request()
    # test_client_query_request()
