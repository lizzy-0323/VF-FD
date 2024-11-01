"""
@Author: laziyu
@Date:2023-2-3
@Description: dataset partition
"""

import argparse

import pandas as pd
from tqdm import tqdm

from data_loader.preprocess import (add_noise, add_overlapping_column,
                                    get_dataset, print_dataset_info,
                                    split_dataset)


def partition(cfg):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="knn", help="dataset choice")
    args = parser.parse_args()
    print(args)
    dataset = get_dataset(args.dataset)
    print_dataset_info(dataset)


if __name__ == "__main__":
    main()
