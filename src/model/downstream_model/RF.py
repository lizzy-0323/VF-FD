"""
@Author: laziyu
@Date:2024-6-27
@Description: downstream task: random forest model, mainly for multiple labels task
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
from data_loader.dataset import load_dataset
from data_loader.file_reader import read_dataset_from_csv
import numpy as np

SEED = 0
mm = MinMaxScaler()


def preprocess_dataset(X, y):
    x_array = X.values
    y_array = y.values.reshape(-1, 1)
    y_array_scaled = mm.fit_transform(y_array)
    y_array_inv = mm.inverse_transform(y_array_scaled)
    y_array = np.rint(y_array_inv).astype(int).flatten()
    return x_array, y_array


def load_data(file_path, dataset_name, seed):
    _, y = load_dataset(dataset_name)
    X = read_dataset_from_csv(file_path)
    X, y = preprocess_dataset(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, seed):
    model = RandomForestClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")


def run(args):
    X_train, X_test, y_train, y_test = load_data(
        args.file_path, args.dataset_name, args.seed
    )
    # 训练模型
    model = train(X_train, y_train, args.seed)
    # 预测测试集结果
    y_pred = predict(model, X_test)

    # 评估模型
    evaluate(y_test, y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="./data/activity/all_features.csv",
        help="path to load dataset",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="activity", help="dataset name"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    # print(args)
    run(args)


if __name__ == "__main__":
    main()
