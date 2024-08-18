#!/usr/bin/env python3
import os
import sys
import yaml

cfg_file_path = "./conf/downstream_task.yaml"

with open(cfg_file_path, "r") as file:
    cfg = yaml.safe_load(file)
dataset_lst = cfg.get("dataset_lst", [])


def generate(dataset_name):

    # 开始训练前的信息输出
    print(f"Starting training with dataset: {dataset_name}")

    # 运行 Python 模型训练脚本
    def run_trainer(
        trainer_name, method=None, metric=None, threshold=None, use_emd=False
    ):
        cmd = f'python3 -m model.trainer.{trainer_name} --dataset_name="{dataset_name}"'
        if method:
            cmd += f' --method="{method}"'
        elif use_emd:
            cmd += f' --use_emd="{use_emd}"'
        elif metric:
            cmd += f' --metric="{metric}"'
        elif threshold:
            cmd += f' --threshold="{threshold}"'
        print(f"Running {cmd}...")
        os.system(cmd)

    # # 运行 arf 训练器
    # run_trainer("arf_trainer", metric="feature")

    # # 运行 arf 训练器，采用label与feature之间的互信息来计算
    # run_trainer("arf_trainer", metric="label")

    # 运行 gf训练器
    # run_trainer("gf_trainer")
    run_trainer("gf_trainer", threshold=0.01)
    # run_trainer("gf_trainer", threshold=0.2)
    # run_trainer("gf_trainer", threshold=0.1)
    # run_trainer("gf_trainer", threshold=0.5, method="cosine")
    # 测试random 分组
    # run_trainer("gf_trainer", method="random-5-5")
    # run_trainer("gf_trainer", method="random-10-5")
    # run_trainer("gf_trainer", method="random-5-10")
    # # 运行 local Filter训练器，使用mi方法
    # run_trainer("filter_trainer", method="mi")

    # 运行 local Filter训练器，使用mi方法, 并且使用emd距离
    # run_trainer("filter_trainer", method="mi", use_emd=True)

    # 运行 Lasso 训练器，使用 local 方法
    run_trainer("lasso_trainer", "local")

    # 运行 Lasso 训练器，使用 global 方法
    run_trainer("lasso_trainer", "global")

    # # 运行 Lasso 训练器，使用 emd 方法
    # run_trainer("lasso_trainer", "emd")

    # 运行 gloabl Filter训练器，使用Mi方法
    run_trainer("global_mi_trainer")

    # 运行 Random 训练器
    run_trainer("random_trainer")
    # 运行 EMD + Random 训练器
    # run_trainer("random_trainer", use_emd=True)
    # 训练完成后的信息输出
    print("All training sessions completed.")


def main():
    # 获取脚本所在目录
    home_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current dir: {home_dir}")
    # 检查命令行参数数量
    if len(sys.argv) == 2:
        dataset_name = sys.argv[1]
        generate(dataset_name)
    else:
        for dataset_name in dataset_lst:
            generate(dataset_name)


if __name__ == "__main__":
    main()
