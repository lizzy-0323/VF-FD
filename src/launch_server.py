import hydra
from omegaconf import DictConfig

from grpc_service.grpc_server.server_rpc import RpcServer


def init_server(server_cfg, client_cfg, num_clients):
    """
    init server
    """
    server = RpcServer(server_cfg, client_cfg, num_clients)
    server.serve()


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def launch_server(cfg: DictConfig):
    server_cfg = cfg.server
    num_clients = cfg.num_clients
    client_cfg = cfg.client
    init_server(server_cfg, client_cfg, num_clients)


if __name__ == "__main__":
    launch_server()
