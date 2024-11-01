"""
@Author: laziyu
@Date:2024-6-12
@Description: downstream task: nn model
"""

import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_loader.dataset import load_dataset, load_label
from data_loader.file_reader import read_dataset_from_csv
from data_loader.preprocess import normalize
from model.trainer.trainer_utils import switch_device, preprocess_data

SEED = 0
BATCH_SIZE = 2048
device = switch_device()


class NN(nn.Module):
    """
    nn trainer for downstream task: 0/1 classification
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def preprocess_dataset(x, y):
    # 将 x 和 y 转换为 numpy 数组
    x_array = x.values
    y_array = y.values.reshape(-1, 1)
    # 转换为 PyTorch Tensor
    x_tensor = torch.from_numpy(x_array).float().to(device)
    y_tensor = torch.from_numpy(y_array).float().to(device)
    # 创建 TensorDataset
    tensor_dataset = TensorDataset(x_tensor, y_tensor)
    return tensor_dataset


def load_data(file_path, dataset_name):
    y = load_label(dataset_name)
    X = read_dataset_from_csv(file_path)
    X = normalize(X)
    input_size = X.shape[1]
    output_size = 1
    tensor_dataset = preprocess_data(X, y, device)
    # 创建 DataLoader
    train_dataset, test_dataset = train_test_split(
        tensor_dataset, test_size=0.4, random_state=SEED
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, input_size, output_size


def train(model, criterion, optimizer, num_epochs, train_loader, filename=None):
    # print("=============")
    # print("Start Training:")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 清空梯度
            output = model(inputs)  # 前向传播
            output = torch.squeeze(output)
            loss = criterion(output, targets)  # 计算损失
            total_loss += loss.item() * inputs.size(0)  # 累加损失，乘以批量大小
            batch_count += inputs.size(0)  # 更新批次计数器
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
        avg_loss = total_loss / batch_count
        # print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    # if filename is not None:
    #     torch.save(model.state_dict(), filename)


def test(model, criterion, test_loader, threshold=0.5):
    # print("=============")
    # print("Start Validing:")
    model.eval()  # 设置模型为评估模式
    correct = 0.0
    total_loss = 0.0
    with torch.no_grad():  # 测试时不计算梯度
        total_loss = 0
        for inputs, targets in test_loader:
            output = model(inputs)
            targets = targets.view(-1, 1)
            loss = criterion(output, targets)
            total_loss += loss.item() * inputs.size(0)
            # 计算正确预测的数量
            pred = (output > threshold).float()  # 将概率大于阈值的预测为1，否则为0
            correct += (pred.round() == targets).sum().item()  # 计算正确预测的数量

    # 计算平均损失和准确率
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(f"Accuracy: {(100*accuracy):.2f}%, Avg loss: {avg_loss:.4f} \n")


def run(args):
    train_loader, test_loader, input_size, output_size = load_data(
        args.file_path, args.dataset_name
    )
    hidden_size1 = 50  # 第一层隐藏层的神经元数量
    hidden_size2 = 50  # 第二层隐藏层的神经元数量
    num_epochs = 20  # 训练的轮数
    model_file = "model" + "_" + args.dataset_name + ".pt"
    model_path = os.path.join(args.model_path, model_file)
    # 实例化神经网络
    model = NN(input_size, hidden_size1, hidden_size2, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    if os.path.exists(model_path):
        print("Model loading:")
        model.load_state_dict(torch.load(model_path))
    else:
        train(model, criterion, optimizer, num_epochs, train_loader, model_path)
        test(model, criterion, test_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="./model/data/census/all_features.csv",
        help="path to load dataset",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model/downstream_model/run",
        help="path to load model",
    )
    parser.add_argument("--dataset_name", type=str, default="knn", help="dataset name")
    args = parser.parse_args()
    # print(args)
    run(args)


if __name__ == "__main__":
    main()
