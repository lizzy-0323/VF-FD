import csv

import pandas as pd
import yaml


def read_dataset_from_csv(file_path):
    """
    read data from csv, ignore duplicate column names
    :param file_path: file path
    :return: data
    """
    df = pd.read_csv(file_path, index_col=0)
    df.columns = [col.split(".")[0] for col in df.columns]
    return df


def save_df_to_csv(df, filename, index=True):
    try:
        df.to_csv(filename, index=index)  # index=False表示不将行索引写入到csv文件
        print(f"Save to {filename}")
    except Exception as e:
        print(f"Save error : {e}")


def save_tuple_to_csv(tuple, filename="./result/model_result.csv"):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        # 写入标题
        writer.writerow(tuple)
        # 写入结果
        print(f"Results saved to {filename}")





def read_client_cfg(cfg_path, client_id):
    """
    Read configuration from YAML file
    :param cfg_path: path to the YAML configuration file
    :param client_id: client id
    :return: configuration for the specified client id
    """
    # desperated
    with open(cfg_path, "r") as file:
        cfg = yaml.safe_load(file)
    return cfg["client"]["client_" + str(client_id)]
