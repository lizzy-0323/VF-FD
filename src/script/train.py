#!/usr/bin/env python3
import os
import subprocess

# 获取脚本所在目录的上级目录
home_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(home_dir)

# 预处理所有数据集
preprocess_script = os.path.join(home_dir, "script", "preprocess_all_dataset.py")
generate_feature_script = os.path.join(home_dir, "script", "generate_features.py")
test_script = os.path.join(home_dir, "script", "test_selected_features.py")
# 数据预处理
subprocess.run(["python", preprocess_script], check=True)
# 生成特征
subprocess.run(["python", generate_feature_script], check=True)
# 下游任务测试
subprocess.run(["python", test_script], check=True)
