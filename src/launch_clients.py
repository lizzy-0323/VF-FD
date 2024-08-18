"""
@author: laziyu
@date:2023-2-7
@description:launch client script
"""

import multiprocessing as mp

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from data_loader.preprocess import (add_noise, add_overlapping_column,
                                    get_dataset, print_dataset_info,
                                    split_dataset)
from grpc_service.grpc_server.client_rpc import RpcClient


def init_client(client_cfg, client_id):
    """
    init client
    """
    client = RpcClient(client_cfg, client_id)
    client.serve()


def prepare_data(cfg):
    """partition dataset"""
    dataset = get_dataset(cfg.dataset)
    print_dataset_info(dataset)
    print("Overlapping rate:", cfg.overlapping_rate)
    # split dataset
    splitted_dataset = split_dataset(dataset, cfg.client_num, cfg.sample_num, cfg.seed)
    overlapping_dataset, overlapping_num_lst = add_overlapping_column(
        splitted_dataset, cfg.overlapping_rate, cfg.seed
    )
    client_data = [
        add_noise(
            data,
            noise_col_num,
            cfg.seed,
            cfg.noise_size,
        )
        for data, noise_col_num in zip(overlapping_dataset, overlapping_num_lst)
    ]
    # save dataset
    for i, silo in enumerate(tqdm(client_data, desc="generate dataset")):
        silo.to_csv(f"{cfg.save_path}/client_{i+1}.csv", index=True)
    # save ground truth data
    ground_truth = split_dataset(dataset, cfg.client_num, cfg.sample_num, cfg.seed)
    for i, silo in enumerate(tqdm(ground_truth, desc="generate ground truth")):
        silo.to_csv(f"{cfg.save_path}/ground_truth_client_{i+1}.csv", index=True)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def launch_clients(cfg: DictConfig):
    process = []
    num_clients = cfg.num_clients
    dataset_cfg = cfg.dataset
    prepare_data(dataset_cfg)
    mp.set_start_method("spawn", force=True)
    for i in range(num_clients):
        client_id = i + 1
        client_cfg = cfg.client
        p = mp.Process(
            target=init_client,
            args=(client_cfg, client_id),
        )
        p.start()
        process.append(p)
    for p in process:
        p.join()


if __name__ == "__main__":
    launch_clients()
