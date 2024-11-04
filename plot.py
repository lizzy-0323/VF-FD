import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["Times New Roman"]
font_title = FontProperties(size=16, weight="bold")
colors = ["#82B0D2", "#FFBE7A", "#FA7F6F", "#8ECFC9"]


def set_border(plt):
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(bwith)  # 图框下边
    ax.spines["left"].set_linewidth(bwith)  # 图框左边
    ax.spines["top"].set_linewidth(bwith)  # 图框上边
    ax.spines["right"].set_linewidth(bwith)  # 图框右边


def plot_accuracy_effect_of_tau():
    # 假设的数据，表示不同tau值下的准确率
    data = {
        "0.01": [95.02, 96.14, 94.48],
        "0.1": [96.07, 95.03, 94.08],
        "0.2": [94.88, 94.78, 92.40],
        "0.5": [90.48, 90.03, 89.98],
    }
    df = pd.DataFrame(data).T
    # 因为每个tau有两个准确率，我们创建两个点的索引
    x_values = [0.2, 0.4, 0.6]  # 例如，如果我们有两个数据点
    # 准备画图
    plt.figure(figsize=(10, 6))  # 可以调整图形大小

    # 绘制折线图
    for tau, accuracy in df.iterrows():
        plt.plot(x_values, accuracy, marker="o", label=f"τ={tau}")

    plt.grid(alpha=0.4, linestyle="--")
    # 添加一些图形元素
    plt.ylabel("Accuracy")
    plt.xlabel("Dataset Split")
    plt.legend()

    # 展示图形
    plt.tight_layout()
    plt.savefig("./result/fig/accuracy_result.png", bbox_inches="tight", dpi=1200)
    plt.show()


def plot_efficiency_effect_of_tau():
    # 从文本数据创建DataFrame
    data = {
        "0.01": [614.41, 281.88, 444],
        "0.1": [585, 213, 444],
        "0.2": [472.73, 208.88, 222],
        "0.5": [472.73, 208.88, 222],
    }
    df = pd.DataFrame(data)
    # 准备画图
    plt.figure(figsize=(10, 6))  # 可以调整图形大小

    # 绘制柱状图
    bar_width = 0.2  # 柱子的宽度
    index = range(len(df["0.1"]))  # 索引位置
    # 绘制三个特征的柱状图
    plt.bar(
        [i - bar_width / 2 for i in index],
        df["0.01"],
        bar_width,
        label="tau=0.01",
        color=colors[0],
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width / 2 for i in index],
        df["0.1"],
        bar_width,
        color=colors[1],
        label="tau=0.1",
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width * 3 / 2 for i in index],
        df["0.2"],
        bar_width,
        color=colors[2],
        label="tau=0.01",
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width * 5 / 2 for i in index],
        df["0.5"],
        bar_width,
        color=colors[3],
        label="tau=0.01",
        edgecolor="black",
        linewidth=2,
    )
    set_border(plt)
    plt.grid(alpha=0.4, linestyle="--")
    # 添加一些图形元素
    plt.ylabel("Communication Cost (KB)")
    plt.xlabel("Overlapping feature rate setting in dataset")
    plt.xticks(
        [i + bar_width for i in index], ("20%", "40%", "60%")
    )  # 根据你的数据调整特征名称
    plt.legend()

    # 展示图形
    plt.tight_layout()
    plt.savefig("./result/fig/efficiency_result.png", dpi=1200)
    plt.show()


def plot_dataset_and_effiency():
    # 从文本数据创建DataFrame
    data = {
        "All-Features": [614.41, 281.88],
        "FEAST": [585, 213],
        "VF-FD": [472.73, 208.88],
    }
    df = pd.DataFrame(data)

    # 准备画图
    plt.figure(figsize=(10, 6))  # 可以调整图形大小

    # 绘制柱状图
    bar_width = 0.25  # 柱子的宽度
    index = range(len(df["All-Features"]))  # 索引位置

    # 绘制三个特征的柱状图
    plt.bar(
        [i - bar_width / 2 for i in index],
        df["All-Features"],
        bar_width,
        label="All Features",
        color=colors[0],
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width / 2 for i in index],
        df["FEAST"],
        bar_width,
        color=colors[1],
        label="FEAST",
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width * 3 / 2 for i in index],
        df["VF-FD"],
        bar_width,
        color=colors[2],
        label="VF-FD",
        edgecolor="black",
        linewidth=2,
    )
    set_border(plt)
    plt.grid(alpha=0.4, linestyle="--")
    # 添加一些图形元素
    plt.ylabel("Communication Cost (KB)")
    plt.xticks(
        [i + bar_width / 2 for i in index], ("Nomao", "Sonar")
    )  # 根据你的数据调整特征名称
    plt.legend()

    # 展示图形
    plt.tight_layout()
    plt.savefig("./result/fig/result.png", dpi=1200)
    plt.show()


if __name__ == "__main__":
    # plot_overlapping_rate_and_method("./primary_overlapping_dataset_result.xlsx")
    plot_dataset_and_effiency()
    plot_efficiency_effect_of_tau()
    # 调用函数
    plot_accuracy_effect_of_tau()
