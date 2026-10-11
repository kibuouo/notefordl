
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"


def export_accuracy_chart(csv_path, model_name="CNN"):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    epochs = df["epoch"]
    accuracy = df["test_accuracy"] * 100
    loss = df["train_loss"]

    best_idx = accuracy.idxmax()
    best_epoch = int(epochs.loc[best_idx])
    best_acc = accuracy.loc[best_idx]

    # 画布
    fig = plt.figure(figsize=(12, 7), facecolor="#F7F9FC")

    # 标题
    fig.text(
        0.07, 0.92,
        f"{model_name} Training Performance",
        fontsize=23, weight="bold", color="#172033"
    )
    fig.text(
        0.07, 0.865,
        f"MNIST  /  {len(df)} Epochs  /  Experiment Results",
        fontsize=11, color="#64748B"
    )

    # 统计卡片数据
    cards = [
        ("BEST ACCURACY", f"{best_acc:.2f}%",
         f"Epoch {best_epoch}"),
        ("FINAL ACCURACY", f"{accuracy.iloc[-1]:.2f}%",
         "Final epoch"),
        ("INITIAL TRAIN LOSS", f"{loss.iloc[0]:.4f}",
         "Epoch 1"),
        ("FINAL TRAIN LOSS", f"{loss.iloc[-1]:.4f}",
         "Final epoch"),
    ]

    for i, (label, value, note) in enumerate(cards):
        x = 0.07 + i * 0.22
        y = 0.67

        card = FancyBboxPatch(
            (x, y), 0.20, 0.155,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="#E2E8F0",
            linewidth=1
        )
        fig.add_artist(card)

        fig.text(
            x + 0.015, y + 0.112, label,
            fontsize=9, color="#64748B",
            weight="medium"
        )
        fig.text(
            x + 0.015, y + 0.050, value,
            fontsize=23, weight="bold",
            color="#16A085" if i == 0 else "#172033"
        )
        fig.text(
            x + 0.015, y + 0.017, note,
            fontsize=9, color="#94A3B8"
        )

    # 折线图
    fig.text(
        0.07, 0.615, "Test Accuracy",
        fontsize=16, weight="bold", color="#172033"
    )

    ax = fig.add_axes([0.08, 0.14, 0.84, 0.43])
    ax.set_facecolor("#FFFFFF")

    bottom = max(0, math.floor(accuracy.min() - 2))
    top = min(100, math.ceil(accuracy.max() + 2))
    ax.set_ylim(bottom, top)

    # 主曲线
    ax.plot(
        epochs, accuracy,
        color="#2563EB",
        linewidth=2.7,
        solid_capstyle="round",
        zorder=3
    )

    # 浅色填充
    ax.fill_between(
        epochs, accuracy, bottom,
        color="#2563EB", alpha=0.07
    )

    # 最高点
    ax.scatter(
        [best_epoch], [best_acc],
        s=80, color="#16A085",
        edgecolors="white", linewidths=2,
        zorder=5
    )

    ax.annotate(
        f"Best  {best_acc:.2f}%",
        xy=(best_epoch, best_acc),
        xytext=(-12, -28),
        textcoords="offset points",
        ha="right",
        fontsize=10, color="#0F766E",
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc="#E7F7F2", ec="none"
        )
    )

    # 坐标轴和网格
    ax.set_xlabel(
        "Epoch", fontsize=11,
        color="#64748B", labelpad=10
    )
    ax.set_ylabel(
        "Test Accuracy", fontsize=11,
        color="#64748B", labelpad=10
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.0f}%")
    )

    ax.grid(axis="y", color="#E2E8F0", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both", colors="#64748B",
        labelsize=10, length=0, pad=9
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.margins(x=0.015)

    # 导出矢量图
    output = csv_path.with_name(
        csv_path.stem + "_report.svg"
    )

    fig.savefig(
        output,
        format="svg",
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )
    plt.close(fig)

    return output
