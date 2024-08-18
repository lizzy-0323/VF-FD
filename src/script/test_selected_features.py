import sys
import os
import yaml
import glob

# 获取脚本所在目录的上级目录
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(home_dir)

# 配置文件路径
cfg_file_path = "./conf/downstream_task.yaml"

with open(cfg_file_path, "r") as file:
    cfg = yaml.safe_load(file)
dataset_lst = cfg.get("dataset_lst", [])
method_file_dict = cfg.get("method_file_dict", {})
model_lst = cfg.get("model_lst", [])
feature_dir = os.path.join(home_dir, cfg.get("feature_dir", "./data"))


def delete_log(dir):
    # 构建文件路径模式
    pattern = os.path.join(dir, "*.log")

    # 使用glob.glob获取所有匹配的文件路径
    files = glob.glob(pattern)

    # 遍历文件列表并删除每个文件
    for file in files:
        os.remove(file)


def test(dataset_name):
    print(f"Starting processing with dataset: {dataset_name}")
    result_file = "./log/" + f"{dataset_name}_results.log"
    for model_name in model_lst:
        print(f"Downsteam task: {model_name}")
        with open(result_file, "a") as f:
            f.write(f"Downsteam task: {model_name}\n")
        for method, file_name in method_file_dict.items():
            file_path = os.path.join(feature_dir, dataset_name, file_name)
            print(f"Method: {method}")
            with open(result_file, "a") as f:
                f.write(f"Method: {method}\n")
            # command = f"python3 -m model.downstream_model.NN --dataset_name={dataset_name} --file_path={file_path}"
            command = f"python3 -m model.downstream_model.clf --model_name={model_name} --dataset_name={dataset_name} --file_path={file_path} >> {result_file} 2>&1"
            os.system(command)
        with open(result_file, "a") as f:
            f.write("\n")


# 检查是否提供了dataset_name参数
if len(sys.argv) == 2:
    # 读取dataset_name参数
    dataset_name = sys.argv[1]
    test(dataset_name)
else:
    for dataset_name in dataset_lst:
        test(dataset_name)


# 开始训练前的信息输出
# print(f"Starting processing with dataset: {dataset_name}")
# if dataset_name in RF_task_dataset:
#     print("Dataset is for Multiple label task.")
#     # 调用Python模块处理LR任务
#     for method, file_name in method_file_dict.items():
#         file_path = os.path.join(feature_dir, dataset_name, file_name)
#         command = f"python3 -m model.downstream_model.RF --dataset_name={dataset_name} --file_path={file_path}"
#         print(f"Method: {method}")
#         os.system(command)

# 判断dataset_name属于哪个列表，并调用相应的Python模块
# if dataset_name in NN_task_dataset:
#     print("Dataset is for Neural Network tasks.")
#     # 调用Python模块处理NN任务
#     for method, file_name in method_file_dict.items():
#         file_path = os.path.join(feature_dir, dataset_name, file_name)
#         command = f"python3 -m model.downstream_model.clf --model_name=mlp --dataset_name={dataset_name} --file_path={file_path}"
#         print(f"Method: {method}")
#         os.system(command)

# if dataset_name in LR_task_dataset:
#     print("Dataset is for Logistic Regression tasks.")
#     # 调用Python模块处理LR任务
#     for method, file_name in method_file_dict.items():
#         file_path = os.path.join(feature_dir, dataset_name, file_name)
#         command = f"python3 -m model.downstream_model.LR --dataset_name={dataset_name} --file_path={file_path}"
#         print(f"Method: {method}")
#         os.system(command)
