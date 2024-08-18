#!/bin/bash

# 设置工作目录为脚本所在目录的上级目录
home_dir=$(cd "$(dirname "$0")"; cd ..; pwd)

# 设置工作目录
work_dir="$home_dir"

# 确保工作目录存在
if [ ! -d "$work_dir" ]; then
    echo "工作目录不存在：$work_dir"
    exit 1
fi

# 删除工作目录下的 outputs 文件夹
rm -rf "$work_dir/outputs"
