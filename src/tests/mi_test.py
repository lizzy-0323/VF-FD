import numpy as np

from data_loader.preprocess import add_noise, get_dataset, split_dataset
from model.overlapping_selection import get_score_list


def load_data(dataset_name, silo_num=6, seed=0, col_num=4):
    np.random.seed(seed)
    dataset = get_dataset(dataset_name)
    # 抽取指定列
    selected_data = dataset.iloc[:, :col_num]
    data_lst = [selected_data.copy() for _ in range(silo_num)]
    remain_data = dataset.iloc[:, col_num:]
    return data_lst, remain_data


def test():
    overlapping_columns, remain_data = load_data("boston_housing")
    noise_rate_lst = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for data, noise_rate in zip(overlapping_columns, noise_rate_lst):
        data = add_noise(
            silo=data, noise_col_num=4, seed=0, noise_size=0.5, noise_rate=noise_rate
        )
    columns = overlapping_columns[0].columns
    score_lst = get_score_list(overlapping_columns, remain_data, columns, method="mi")

    print(score_lst)
    # print(remain_data)


if __name__ == "__main__":
    test()
