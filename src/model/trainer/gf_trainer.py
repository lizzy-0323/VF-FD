"""
@Author: laziyu
@Date:2024-7-13
@Description: Gradient feature selector
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_loader.data_operation import merge
from data_loader.dataset import load_label
from model.trainer.trainer_utils import (
    partition_data_by_random,
    partition_data_by_default,
    preprocess_data,
    switch_device,
    partition_data_by_emd,
    DATASET_PATH,
    MODEL_PATH,
    BATCH_SIZE,
    TEST_SIZE,
    SEED,
)
from data_loader.file_reader import read_dataset_from_csv, save_df_to_csv
from data_loader.preprocess import normalize, preprocess_dataset

EPOCHS = 20
device = switch_device()
hidden_size = 32
# 设定损失函数和优化器
criterion = nn.BCELoss()  # 假设是二分类问题，使用二元交叉熵损失


class GradientFeatureSelector(nn.Module):
    """
    gradient feature selector
    """

    def __init__(self, input_size, output_size):
        # 预先定义hidden size
        super(GradientFeatureSelector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output


def apply_l1_regularization_to_gradients(model, l1_lambda=0.1):
    # 遍历模型的所有参数
    grad = model.fc1.weight.grad
    l1_norm = torch.norm(grad, p=1)
    grad.data += l1_lambda * (l1_norm / grad.numel()) * grad.sign()


def train(
    model, criterion, optimizer, epochs, train_loader, filename=None, l1_lambda=0.001
):
    print("=============")
    print("Start Training:")
    for epoch in range(epochs):
        model.train()
        total_loss = 0  # 初始化总损失，用于计算平均损失
        batch_count = 0  # 初始化批次计数
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 清空梯度
            output = model(inputs)  # 前向传播
            output = output.squeeze(1)
            loss = criterion(output, targets)  # 计算损失
            l1_penalty = torch.norm(model.fc1.weight, 1)
            loss += l1_lambda * l1_penalty
            total_loss += loss.item() * inputs.size(0)  # 累加损失，乘以批量大小
            batch_count += inputs.size(0)  # 更新批次计数器
            loss.backward()  # 反向传播
            # 在梯度上应用L1正则化
            # apply_l1_regularization_to_gradients(model, 0.001)
            optimizer.step()  # 更新权重
            # 限定第一层权重参数
            with torch.no_grad():
                model.fc1.weight.data = torch.clamp(model.fc1.weight.data, min=0, max=1)
        avg_loss = total_loss / batch_count
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    # print(model.fc1.weight.data)
    # if filename is not None:
    #     torch.save(model.state_dict(), filename)


def valid(model, criterion, valid_loader, threshold=0.5):
    model.eval()  # 设置模型为评估模式
    correct = 0.0
    total_loss = 0.0
    with torch.no_grad():  # 测试时不计算梯度
        total_loss = 0
        for inputs, targets in valid_loader:
            output = model(inputs)
            output = output.squeeze(1)
            loss = criterion(output, targets)
            total_loss += loss.item() * inputs.size(0)
            # 计算正确预测的数量
            pred = (output > threshold).float()  # 将概率大于阈值的预测为1，否则为0
            correct += (pred.round() == targets).sum().item()  # 计算正确预测的数量

    # 计算平均损失和准确率
    avg_loss = total_loss / len(valid_loader.dataset)
    accuracy = correct / len(valid_loader.dataset)
    print(f"Accuracy: {(100*accuracy):.2f}%, Avg loss: {avg_loss:.4f} \n")


def compute_grad(model, sample, target):
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)
    sample.requires_grad_()
    pred = model(sample)
    pred = pred.squeeze(1)
    loss = criterion(pred, target)  # 计算损失对模型参数的梯度
    # grads = torch.autograd.grad(loss, model.parameters())
    result = torch.autograd.grad(loss, list(model.parameters()))
    # print(sample.grad)
    # print(result[0].shape)

    return result


def compute_sample_grads(model, data, targets):
    """manually process each sample with per sample gradient"""
    # 算出每一个样本对应的梯度值
    sample_grads = [
        compute_grad(model, data[i], targets[i]) for i in range(data.size(0))
    ]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


def compute_sample_feature_score(model, sample_grads):
    with torch.no_grad():
        W = list(model.parameters())[0].data
        W_expanded = W.unsqueeze(0).expand(sample_grads.size(0), -1, -1)
        # feature_grads = torch.bmm(W_expanded, sample_grads)
        # 提取对角线元素
        # feature_scores = torch.diagonal(feature_grads, dim1=1, dim2=2)
        feature_matrix = torch.mul(sample_grads, W_expanded)
        feature_scores = torch.sum(feature_matrix, dim=1)
        return feature_scores


def retrain(model, train_loader, device):
    sample_grads = None
    model.train()  # Set the model to training mode
    print("Starting Retraining")
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        grads = compute_sample_grads(model, data, target)
        if sample_grads is None:
            sample_grads = grads[0]
        else:
            sample_grads = torch.cat([sample_grads, grads[0]], dim=0)
    # sample grads为样本梯度, [sample_num, hidden_size, feature_num]
    feature_score_lst = compute_sample_feature_score(model, sample_grads)
    return feature_score_lst


def get_min_feature_index(feature_score_lst, candidate_index_lst, row_index):
    result = 1000000
    for column_index in candidate_index_lst:
        if row_index >= len(feature_score_lst):
            print(row_index, len(feature_score_lst))
            raise ValueError("row index error")
        if column_index >= len(feature_score_lst[row_index]):
            print(column_index, len(feature_score_lst[row_index]))
            raise ValueError("column index error")
        if feature_score_lst[row_index][column_index] <= result:
            result = column_index
    return result


def get_rand_feature_index(candidate_index_lst):
    return random.choice(candidate_index_lst)


def select_features(feature_score_lst, all_features, threshold=1e-2, method="normal"):
    print("Starting Selection")
    if method == "random-5-5":
        overlapping_column_lst, non_overlapping_columns = partition_data_by_random(
            all_features, 5, 5
        )
    elif method == "random-5-10":
        overlapping_column_lst, non_overlapping_columns = partition_data_by_random(
            all_features, 5, 10
        )
    elif method == "random-10-5":
        overlapping_column_lst, non_overlapping_columns = partition_data_by_random(
            all_features, 10, 5
        )
    elif method == "without-overlapping":
        overlapping_column_lst, non_overlapping_columns = partition_data_by_default(
            all_features
        )
    elif method == "cosine":
        overlapping_column_lst, non_overlapping_columns = partition_data_by_emd(
            all_features, threshold=0.1, metric="cosine"
        )
    else:
        overlapping_column_lst, non_overlapping_columns = partition_data_by_emd(
            all_features, threshold
        )
    if method != "without-overlapping":
        count = 0
        row = all_features.shape[0]
        column = len(overlapping_column_lst)
        candidate_index_lst = []
        print(f"Overlapping Group Num: {column}")
        for overlapping_column in overlapping_column_lst:
            index_lst = []
            for col in overlapping_column.columns:
                count += 1
                if col in all_features.columns:
                    col_index = all_features.columns.get_loc(col)
                    index_lst.append(col_index)
                else:
                    raise ValueError("This column not in Origin Features")
            candidate_index_lst.append(index_lst)
        print(f"Columns: {count-len(overlapping_column_lst)}")
    else:
        row = all_features.shape[0]
        column = all_features.shape[1]
    selected_features = np.empty([row, column])
    # selection using ls-score
    for i in range(row):
        for j in range(column):
            if method == "random":
                column_index = get_rand_feature_index(candidate_index_lst[j])
            elif method == "without-overlapping":
                column_index = get_min_feature_index(
                    feature_score_lst,
                    list(range(column)),
                    i,
                )
            else:
                column_index = get_min_feature_index(
                    feature_score_lst, candidate_index_lst[j], i
                )
            selected_features[i][j] = all_features.iloc[i, column_index]
    selected_features = pd.DataFrame(
        selected_features,
        columns=[f"New_Column_{i}" for i in range(selected_features.shape[1])],
    )
    result = merge(selected_features, non_overlapping_columns)
    # print(result)
    return result


def load_train_data(x, y):
    tensor_dataset = preprocess_data(x, y, device)
    # 创建 DataLoader
    train_dataset, test_dataset = train_test_split(
        tensor_dataset, test_size=TEST_SIZE, random_state=SEED
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    input_size = x.shape[1]
    output_size = 1
    return train_loader, valid_loader, input_size, output_size


def run(args):
    if args.method != "normal":
        result_path = os.path.join(
            args.result_path,
            args.dataset_name,
            "gf_" + args.method + "_selected_features.csv",
        )
    elif args.threshold != 1e-2:
        result_path = os.path.join(
            args.result_path,
            args.dataset_name,
            "gf_" + str(args.threshold) + "_selected_features.csv",
        )
    else:
        result_path = os.path.join(
            args.result_path, args.dataset_name, "gf_selected_features.csv"
        )
    dataset_path = os.path.join(
        args.dataset_path, args.dataset_name, "all_features.csv"
    )
    model_file = "model" + "_" + args.dataset_name + ".pt"
    model_dir = os.path.join(args.model_dir, "gf")
    model_path = os.path.join(model_dir, model_file)
    all_features = read_dataset_from_csv(dataset_path)
    all_features = normalize(all_features)
    label = load_label(args.dataset_name)
    input_size, output_size = all_features.shape[1], 1
    model = GradientFeatureSelector(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if os.path.exists(model_path):
        print("Model loading:")
        model.load_state_dict(torch.load(model_path))
    else:
        train_loader, valid_loader, input_size, output_size = load_train_data(
            all_features, label
        )
        train(model, criterion, optimizer, EPOCHS, train_loader, model_path)
        valid(model, criterion, valid_loader)
    print("Processing retrain data")
    retrain_dataset = preprocess_data(all_features, label, device)
    retrain_loader = DataLoader(retrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    feature_score_lst = retrain(model, retrain_loader, device)
    selected_features = select_features(
        feature_score_lst, all_features, args.threshold, args.method
    )
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
    parser.add_argument("--method", type=str, default="normal", help="method")
    parser.add_argument("--threshold", type=float, default=0.2, help="threshold")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
