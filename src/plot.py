import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from matplotlib import font_manager as fm

# font_path = "./Times New Roman.ttf"
# fm.fontManager.addfont(font_path)
mpl.rcParams["font.serif"] = ["Times New Roman"]
font_title = FontProperties(size=16, weight="bold")
colors = ["#82B0D2", "#FFBE7A", "#FA7F6F", "#8ECFC9"]
font_label = FontProperties(size=20, weight="bold")
hatches = ["|", "-", "/"]


def set_border(plt):
    bwith = 4  # 边框宽度设置为2
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(bwith)  # 图框下边
    ax.spines["left"].set_linewidth(bwith)  # 图框左边
    ax.spines["top"].set_linewidth(bwith)  # 图框上边
    ax.spines["right"].set_linewidth(bwith)  # 图框右边


def plot_accuracy_effect_of_tau():
    # 假设的数据，表示不同tau值下的准确率
    data = {
        "0.01": [95.02, 96.14, 94.48],
        "0.10": [96.07, 95.03, 94.08],
        "0.20": [94.88, 94.78, 92.40],
        "0.50": [90.48, 90.03, 89.98],
    }
    df = pd.DataFrame(data).T
    # 因为每个tau有两个准确率，我们创建两个点的索引
    x_values = [0.2, 0.4, 0.6]  # 例如，如果我们有两个数据点
    # 准备画图
    plt.figure(figsize=(10, 6))  # 可以调整图形大小
    # 定义一个包含不同标记形状的列表
    markers = ["o", "s", "^", "D", "*"]  # 圆圈，正方形，向上的三角形，菱形，星号
    # 绘制折线图
    for tau, accuracy in enumerate(df.iterrows()):  # 使用enumerate获取索引和值
        plt.plot(
            x_values,
            accuracy[1],
            marker=markers[tau],
            label=f"τ={accuracy[0]}",
            linewidth=5,
            markersize=15,
        )

    plt.grid(alpha=0.8, linestyle="--")
    # 添加一些图形元素
    plt.ylabel("Accuracy", fontproperties=font_label)
    plt.xlabel("Overlapping rate setting in dataset", fontproperties=font_label)
    # plt.title("Accuracy Effect of τ", fontproperties=font_title)
    plt.xticks(size=20, weight="bold")
    plt.yticks(size=20, weight="bold")
    plt.legend(prop={"size": 25})
    set_border(plt)
    # 展示图形
    plt.tight_layout()
    plt.savefig("accuracy_tau.png", bbox_inches="tight")
    # plt.show()


def plot_efficiency_effect_of_tau():
    # 从文本数据创建DataFrame
    data = {
        "0.01": [383.03, 464, 549.44],
        "0.1": [328.51, 420, 516.44],
        "0.2": [299.82, 401, 504.4],
        "0.5": [275.73, 380.16, 489.18],
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
        label=" τ=0.01",
        color=colors[0],
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width / 2 for i in index],
        df["0.1"],
        bar_width,
        color=colors[1],
        label=" τ=0.10",
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width * 3 / 2 for i in index],
        df["0.2"],
        bar_width,
        color=colors[2],
        label=" τ=0.20",
        edgecolor="black",
        linewidth=2,
    )
    plt.bar(
        [i + bar_width * 5 / 2 for i in index],
        df["0.5"],
        bar_width,
        color=colors[3],
        label=" τ=0.50",
        edgecolor="black",
        linewidth=2,
    )
    set_border(plt)
    plt.grid(alpha=0.8, linestyle="--")
    # 添加一些图形元素
    plt.ylabel("Communication Cost (KB)", fontproperties=font_label)
    plt.xlabel("Overlapping rate setting in dataset", fontproperties=font_label)
    plt.xticks(
        [i + bar_width for i in index],
        ("20%", "40%", "60%"),
        size=20,
        weight="bold",
    )  # 根据你的数据调整特征名称

    plt.yticks(size=20, weight="bold")
    plt.legend(prop={"size": 18})

    # 展示图形
    plt.tight_layout()
    plt.savefig("efficiency_tau.png", bbox_inches="tight")
    # plt.show()


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
    bar_width = 0.20  # 柱子的宽度
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
        # hatch=hatches[0],
    )
    plt.bar(
        [i + bar_width / 2 for i in index],
        df["FEAST"],
        bar_width,
        color=colors[1],
        label="FEAST",
        edgecolor="black",
        linewidth=2,
        # hatch=hatches[1],
    )
    plt.bar(
        [i + bar_width * 3 / 2 for i in index],
        df["VF-FD"],
        bar_width,
        color=colors[2],
        label="VF-FD",
        edgecolor="black",
        linewidth=2,
        # hatch=hatches[2],
    )
    set_border(plt)
    plt.grid(alpha=0.8, linestyle="--")
    # 添加一些图形元素
    plt.ylabel("Communication Cost (KB)", fontproperties=font_label)
    plt.xticks(
        [i + bar_width / 2 for i in index], ("Nomao", "Sonar"), size=20, weight="bold"
    )  # 根据你的数据调整特征名称
    plt.yticks(size=20, weight="bold")
    plt.legend(prop={"size": 25})

    # 展示图形
    plt.tight_layout()
    plt.savefig("efficiency_result.png", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # plot_overlapping_rate_and_method("./primary_overlapping_dataset_result.xlsx")
    plot_dataset_and_effiency()
    plot_efficiency_effect_of_tau()
    # 调用函数
    plot_accuracy_effect_of_tau()
