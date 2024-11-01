# !/bin/bash
home_dir=$(cd `dirname $0`; cd ..; pwd)
work_dir=$home_dir/grpc_service/protos
cd $work_dir
# generate rpc file
python -m grpc_tools.protoc -I. ./client_rpc.proto --python_out=../rpc --grpc_python_out=../rpc
sed -i '' 's/import client_rpc_pb2 as client__rpc__pb2/import grpc_service.rpc.client_rpc_pb2 as client__rpc__pb2/g' ../rpc/client_rpc_pb2_grpc.py
python -m grpc_tools.protoc -I. ./server_rpc.proto --python_out=../rpc --grpc_python_out=../rpc
sed -i '' 's/import server_rpc_pb2 as server__rpc__pb2/import grpc_service.rpc.server_rpc_pb2 as server__rpc__pb2/g' ../rpc/server_rpc_pb2_grpc.py
python -m grpc_tools.protoc -I. ./basic.proto --python_out=../rpc --grpc_python_out=../rpc
sed -i '' 's/import basic_pb2 as basic__pb2/import grpc_service.rpc.basic_pb2 as basic__pb2/g' ../rpc/client_rpc_pb2.py
sed -i '' 's/import basic_pb2 as basic__pb2/import grpc_service.rpc.basic_pb2 as basic__pb2/g' ../rpc/server_rpc_pb2.py
# 修改rpc文件夹下的client_rpc_pb2_grpc.py文件和client_rpc_pb2.py文件的第一行import路径
# python -m grpc_tools.protoc -I. ./leader_rpc.proto --python_out=../rpc --grpc_python_out=../rpc
# sed -i '' 's/import leader_rpc_pb2 as leader__rpc__pb2/import grpc_service.rpc.leader_rpc_pb2 as leader__rpc__pb2/g' ../rpc/leader_rpc_pb2_grpc.py
# sed -i '' 's/import basic_pb2 as basic__pb2/import grpc_service.rpc.basic_pb2 as basic__pb2/g' ../rpc/leader_rpc_pb2.py