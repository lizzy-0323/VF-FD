from concurrent import futures

import grpc
from grpc_service.rpc import leader_and_client_pb2, leader_and_client_pb2_grpc


class ClientServiceServicer(leader_and_client_pb2_grpc.ClientServiceServicer):
    def GetIdList(self, request, context):
        return leader_and_client_pb2.IdResponse(id="233")

    def GetQueryList(self, request, context):
        response = leader_and_client_pb2.QueryResponse()
        response.query_list.extend(["1", "2", "3"])
        return response


def serve():
    print("Server start!")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    leader_and_client_pb2_grpc.add_ClientServiceServicer_to_server(
        ClientServiceServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
