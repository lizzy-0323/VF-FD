"""
@author: laziyu
@date:2023-2-7
@description:launch leader node
"""

import hydra
from omegaconf import DictConfig

from model.leader import Leader


def init_leader(cfg, client_cfg_dict, num_clients):
    """
    init leader
    """
    leader = Leader(
        cfg,
        client_cfg_dict,
        num_clients,
    )
    leader.serve()


@hydra.main(version_base=None, config_path="./", config_name="config")
def launch_leader(cfg: DictConfig):
    leader_cfg = cfg.leader
    num_clients = cfg.num_clients
    client_cfg_dict = {}
    for i in range(num_clients):
        client_id = str(i + 1)
        client_cfg = cfg.client[f"client_{client_id}"]
        client_cfg_dict[int(client_id)] = {
            "host": client_cfg.host,
            "port": client_cfg.port,
        }
    init_leader(leader_cfg, client_cfg_dict, num_clients)


if __name__ == "__main__":
    launch_leader()
