"""
@Author: laziyu
@Date:2024-4-13
@Description: regression dnn model
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from data_loader.data_operation import merge
from data_loader.dataset import load_dataset, load_overlapping_dataset
from data_loader.file_reader import read_dataset_from_csv, save_df_to_csv
from data_loader.preprocess import normalize
from model.overlapping_selection import select
from model.trainer.trainer_utils import (
    partition_data_by_emd,
    switch_device,
    preprocess_data,
)
from model.trainer.trainer_utils import *


EPOCHS = 10
criterion = nn.MSELoss()  # 均方误差损失
mm = MinMaxScaler()
device = switch_device()


# 定义模型结构
class ARFfeatureSelector(nn.Module):
    def __init__(self, input_size, output_size):
        super(ARFfeatureSelector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )
        # self.input_layer = nn.Linear(input_size, 128)  # 第一个隐藏层
        # self.hidden_layer = nn.Linear(128, 64)  # 第二个隐藏层
        # self.output_layer = nn.Linear(64, output_size)  # 输出层

    def forward(self, x):
        # x = torch.relu(self.input_layer(x))
        # x = torch.relu(self.hidden_layer(x))
        # output = self.output_layer(x)
        output = self.layers(x)
        return output


def primary_selection(overlapping_column_lst, columns, method):
    """
    return: selection result
    """
    _, selection_result = select(columns, overlapping_column_lst, method)
    return selection_result


def load_train_data(
    overlapping_column_lst,
    non_overlapping_columns,
    dataset_name=None,
    metric="feature",
    method="anova",
):
    if metric == "feature":
        overlapping_selection_result = primary_selection(
            overlapping_column_lst, non_overlapping_columns, method
        )
    elif metric == "label":
        _, label = load_dataset(dataset_name)
        label = label.to_frame()
        overlapping_selection_result = primary_selection(
            overlapping_column_lst, label, method
        )
    else:
        raise ValueError("metric must be 'feature' or 'label'")
    tensor_dataset = preprocess_data(
        non_overlapping_columns, overlapping_selection_result
    )
    # 创建 DataLoader
    train_dataset, test_dataset = train_test_split(
        tensor_dataset, test_size=TEST_SIZE, random_state=SEED
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    input_size = non_overlapping_columns.shape[1]
    output_size = overlapping_selection_result.shape[1]
    return train_loader, test_loader, input_size, output_size


def train(model, criterion, optimizer, epochs, train_loader, filename=None):
    print("=============")
    print("Start Training:")
    for epoch in range(epochs):
        model.train()
        total_loss = 0  # 初始化总损失，用于计算平均损失
        batch_count = 0  # 初始化批次计数
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            total_loss += loss.item() * inputs.size(0)  # 累加损失
            batch_count += inputs.size(0)  # 更新批次计数
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        # 计算当前epoch的平均损失
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    # 保存model
    if filename is not None:
        torch.save(model.state_dict(), filename)


def valid(model, test_loader):
    print("=============")
    print("Start Validing:")
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 测试时不计算梯度
        total_loss = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        average_loss = total_loss / len(test_loader)
        print("Avg MSE Loss on Valid set:", average_loss)


def select_features(model, all_features):
    """
    feature selection by mse loss
    """
    overlapping_column_lst, non_overlapping_columns = partition_data_by_emd(
        all_features
    )
    overlapping_column_name = overlapping_column_lst[0].columns
    # print(len(overlapping_column_lst))
    # print(len(overlapping_column_lst[0]))
    selected_features = np.empty_like(overlapping_column_lst[0].values, dtype=float)
    model.eval()
    loss_func = nn.MSELoss(reduction="none")
    mse_losses = []
    with torch.no_grad():
        # 提取非重叠特征对应的数据
        non_overlapping_columns_transformed = mm.fit_transform(
            non_overlapping_columns.values
        )
        non_overlapping_tensor = (
            torch.from_numpy(non_overlapping_columns_transformed).float().to(device)
        )
        # 通过模型进行预测
        pred = model(non_overlapping_tensor)
        for column_index, columns in enumerate(overlapping_column_lst):
            columns_transformed = mm.fit_transform(columns.values)
            column_tensor = torch.from_numpy(columns_transformed).float().to(device)
            mse_loss = torch.sum(loss_func(pred, column_tensor), dim=1)
            mse_losses.append(mse_loss.cpu())
        min_mse_indices = torch.argmin(torch.stack(mse_losses), dim=0)
    for row, index in enumerate(min_mse_indices):
        # 选择对应的列
        selected_features[row] = overlapping_column_lst[index].iloc[row].values
    overlapping_columns = pd.DataFrame(
        selected_features, columns=overlapping_column_name
    )
    result = merge(non_overlapping_columns, overlapping_columns)
    return result


def run(args):
    if args.metric == "feature":
        result_path = os.path.join(
            args.result_path, args.dataset_name, "arf_feature_selected_features.csv"
        )
    elif args.metric == "label":
        result_path = os.path.join(
            args.result_path, args.dataset_name, "arf_label_selected_features.csv"
        )
    dataset_path = os.path.join(
        args.dataset_path, args.dataset_name, "all_features.csv"
    )
    model_file = "model" + "_" + args.dataset_name + "_" + args.metric + ".pt"
    model_path = os.path.join(args.model_dir, "arf", model_file)
    all_features = read_dataset_from_csv(dataset_path)
    all_features = normalize(all_features)
    overlapping_column_lst, non_overlapping_columns = partition_data_by_emd(
        all_features
    )
    input_size, output_size = (
        non_overlapping_columns.shape[1],
        overlapping_column_lst[0].shape[1],
    )
    model = ARFfeatureSelector(input_size=input_size, output_size=output_size).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if os.path.exists(model_path):
        print("Model loading:")
        model.load_state_dict(torch.load(model_path))
    else:
        train_loader, valid_loader, input_size, output_size = load_train_data(
            overlapping_column_lst,
            non_overlapping_columns,
            args.dataset_name,
            args.metric,
            args.method,
        )
        train(model, criterion, optimizer, EPOCHS, train_loader, model_path)
        valid(model, valid_loader)
    selected_features = select_features(model, all_features)
    save_df_to_csv(selected_features, result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, default=DATASET_PATH, help="path to load dataset"
    )
    parser.add_argument(
        "--model_dir", type=str, default=MODEL_PATH, help="path to load model"
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default=DATASET_PATH,
        help="path to save result",
    )
    parser.add_argument("--dataset_name", type=str, default="knn", help="dataset name")
    parser.add_argument("--method", type=str, default="anova", help="filter method")
    parser.add_argument(
        "--metric",
        type=str,
        default="feature",
        help="metric,feature vs feature or feature vs label",
    )
    args = parser.parse_args()
    # print(args)
    run(args)


if __name__ == "__main__":
    main()
# 测试模型
# model.load_state_dict(torch.load(model_path))
# average_test_loss = test(model, test_loader)
# print(f"Average test loss: {average_test_loss}")
