import os
import sys

sys.path.append(os.getcwd())
from data_loader.dataset import (
    load_dataset,
    load_dataset_from_csv,
    load_overlapping_dataset,
)

overhead_in_sonar = 13.82
overhead_in_nomao = 14.69
vf_fd_sonar = 15
vf_fd_nomao = 116
feast_sonar = 9
feast_nomao = 0


def compute_sample_overhead(data, total_size):
    # 检查数据是否为二维数组
    if data.ndim != 2:
        raise ValueError("数据集必须是二维数组")

    # 计算样本个数（行数）
    num_samples = data.shape[0]

    # 计算维度大小（列数）
    num_features = data.shape[1]

    # 计算每个样本所占的大小
    overhead_per_item = total_size / (num_samples * num_features)

    return f"{overhead_per_item:.2f}"


def compute_all_feature_cost(overhead_per_item, column):
    client_column = column // 4
    send_to_leader_cost = client_column * 3 * 2 * float(overhead_per_item)
    send_to_server_cost = column * float(overhead_per_item)
    cost = 100 * (send_to_leader_cost + send_to_server_cost) / 1024
    print("All_feature cost:", cost)


def compute_feast_cost(overhead_per_item, column):
    client_column = column // 4
    cost = 100 * ((client_column - 9) * float(overhead_per_item) * 4) / 1024
    print("feast cost:", cost)


def compute_vf_fd_cost(overhead_per_item, column, delete_num=vf_fd_nomao):
    client_column = column // 4
    send_to_leader_cost = client_column * 4 * float(overhead_per_item)
    column = column - delete_num
    send_to_server_cost = column * float(overhead_per_item)
    cost = 100 * (send_to_leader_cost + send_to_server_cost) / 1024
    print("vf-fd cost:", cost)


def test_effiency(dataset_name):
    dataset_path = os.path.join("./data/", dataset_name, "all_features.csv")
    data_size = os.path.getsize(dataset_path)
    data = load_dataset_from_csv(dataset_path)
    column = data.shape[1]
    overhead_per_item = compute_sample_overhead(data, data_size)
    print(f"Dataset:{dataset_name}")
    print(f"Columns:{column}")
    # print(overhead_per_item)
    compute_all_feature_cost(overhead_per_item, column)
    compute_feast_cost(overhead_per_item, column)
    compute_vf_fd_cost(overhead_per_item, column)


if __name__ == "__main__":
    # test_effiency("sonar")
    lst = [[117, 155, 175, 192], [204, 235, 248, 263], [289, 312, 319, 331]]
    raw_columns = [192, 264, 336]
    for l, raw_column in zip(lst, raw_columns):
        for el in l:
            compute_vf_fd_cost(14.69, raw_column, el)
