from datetime import datetime

import pandas as pd

def csv_to_pivot_excel(input_file, output_file):
    # 从CSV文件中读取数据
    with open(input_file, "r") as file:
        data = file.readlines()

    # 初始化一个空的DataFrame
    df_list = []
    name_1 = "Overlapping rate"
    name_2 = "Method"
    grid_name = "Accuracy"
    # 遍历每一行数据，分割并添加到列表中
    for row in data:
        row_data = row.strip().split(",")
        if len(row_data) == 3:  # 确保行数据格式正确
            df_list.append(
                {
                    name_1: row_data[0],
                    name_2: row_data[1],
                    grid_name: float(row_data[2]),  # 将Accuracy转换为浮点数
                }
            )

    # 将列表转换为DataFrame
    df = pd.DataFrame(df_list)

    # 使用pivot方法转换DataFrame
    pivot_df = df.pivot(index=name_2, columns=name_1, values=grid_name)

    # 将转换后的DataFrame导出为Excel文件
    pivot_df.to_excel(output_file)


if __name__ == "__main__":
    csv_to_pivot_excel("./result/knn.csv", "knn.xlsx")
