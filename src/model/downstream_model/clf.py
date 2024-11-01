"""
@Author: laziyu
@Date:2024-6-12
@Description: downstream task: multiple classifiers
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse
from data_loader.dataset import load_dataset, load_dataset_from_csv
from data_loader.file_reader import read_dataset_from_csv
import xgboost as xgb
import numpy as np

SEEDS = [3, 5, 9, 24, 42]
mm = MinMaxScaler()


def preprocess_dataset(X, y):
    x_array = X.values
    y_array = y.values.reshape(-1, 1)
    # x_array_scaled = mm.fit_transform(x_array)
    # y_array_scaled = mm.fit_transform(y_array)
    # return x_array_scaled, y_array_scaled
    return x_array, y_array


def load_data(file_path, dataset_name, seed):
    _, y = load_dataset(dataset_name)
    X = read_dataset_from_csv(file_path)
    X = X.fillna(0)
    X, y = preprocess_dataset(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def train(X_train, y_train, model_name, seed):
    if model_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(50, 50), random_state=seed)
    elif model_name == "logistic":
        model = LogisticRegression(random_state=seed)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=seed)
    elif model_name == "xgboost":
        model = xgb.XGBClassifier(random_state=seed)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def evaluate(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def run(args):
    accuracies = []

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = load_data(
            args.file_path, args.dataset_name, seed
        )
        # 训练模型
        model = train(X_train, y_train, args.model_name, seed)
        # 预测测试集结果
        y_pred = predict(model, X_test)
        # 评估模型并记录Accuracy
        accuracy = evaluate(y_test, y_pred)
        accuracies.append(accuracy)

    # 计算均值和方差
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.std(accuracies)

    # 打印结果
    print(f"{mean_accuracy*100:.2f}±{variance_accuracy:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="./data/knn/local_lasso_selected_features.csv",
        help="path to load dataset",
    )
    parser.add_argument("--dataset_name", type=str, default="knn", help="dataset name")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mlp",
        choices=["mlp", "logistic", "random_forest", "xgboost"],
        help="model name to use",
    )
    args = parser.parse_args()
    # print(args)
    run(args)


def test_knn_and_sonar():
    accuracies = []
    for seed in SEEDS:
        for dataset_name in ["knn", "sonar"]:
            X, y = load_dataset(dataset_name)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )  # 训练模型
            for model_name in ["logistic"]:
                model = train(X_train, y_train, model_name, seed)
                # 预测测试集结果
                y_pred = predict(model, X_test)
                # 评估模型
                accuracy = evaluate(y_test, y_pred)
                accuracies.append(accuracy)

    # 计算均值和方差
    mean_accuracy = np.mean(accuracies)
    variance_accuracy = np.var(accuracies)
    # 打印结果
    print(f"{mean_accuracy*100:.2f}±{variance_accuracy:.2f}")


if __name__ == "__main__":
    # test_knn_and_sonar()
    main()
