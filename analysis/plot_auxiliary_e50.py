import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
import pandas as pd


MODEL_ORDER = ["CNN-SW", "CNN-M", "CNN-DN"]
SHIFT_ORDER = ["gaussian_noise", "blur", "brightness_shift", "contrast_shift"]
SHIFT_LABELS = {
    "gaussian_noise": "Gaussian\nnoise",
    "blur": "Blur",
    "brightness_shift": "Brightness",
    "contrast_shift": "Contrast",
}
MODEL_COLORS = {
    "CNN-SW": "#0072B2",  # blue
    "CNN-M": "#009E73",  # green
    "CNN-DN": "#D55E00",  # vermillion
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create auxiliary OOD and representation-analysis plots."
    )
    parser.add_argument("--ood-results", default="results_e50/ood_results.csv")
    parser.add_argument(
        "--complexity-summary",
        default="analysis/outputs_e50/complexity_joined_summary.csv",
    )
    parser.add_argument("--plot-dir", default="analysis/plots_e50")
    return parser.parse_args()


def prepare_data(ood_results_path, complexity_summary_path):
    ood = pd.read_csv(ood_results_path)
    summary = pd.read_csv(complexity_summary_path)

    ood["model"] = pd.Categorical(ood["model"], MODEL_ORDER, ordered=True)
    ood["ood_shift"] = pd.Categorical(ood["ood_shift"], SHIFT_ORDER, ordered=True)
    ood = ood.sort_values(["ood_shift", "model"])

    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model")
    return ood, summary


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)


def save_ood_drop_heatmap(ood, plot_dir):
    matrix = (
        ood.pivot(index="model", columns="ood_shift", values="ood_drop")
        .loc[MODEL_ORDER, SHIFT_ORDER]
    )

    base_cmap = plt.get_cmap("BuPu")
    truncated_cmap = LinearSegmentedColormap.from_list(
        "BuPu_truncated", base_cmap(np.linspace(0.22, 1.0, 256))
    )

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    image = ax.imshow(
        matrix.values,
        cmap=truncated_cmap,
        aspect="auto",
        norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=0.72),
    )

    ax.set_xticks(np.arange(len(SHIFT_ORDER)))
    ax.set_xticklabels([SHIFT_LABELS[s] for s in SHIFT_ORDER])
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title("OOD Accuracy Drop by Shift", pad=12)

    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, shift in enumerate(SHIFT_ORDER):
            value = matrix.loc[model, shift]
            color = "white" if value > 0.42 else "#1f2933"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("IID-to-OOD accuracy drop")
    fig.tight_layout()
    fig.savefig(plot_dir / "ood_drop_heatmap.png", dpi=300)
    plt.close(fig)


def save_ood_accuracy_grouped_bar(ood, plot_dir):
    pivot = (
        ood.pivot(index="ood_shift", columns="model", values="ood_acc")
        .loc[SHIFT_ORDER, MODEL_ORDER]
    )

    x = np.arange(len(SHIFT_ORDER))
    width = 0.23

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for idx, model in enumerate(MODEL_ORDER):
        offset = (idx - 1) * width
        ax.bar(
            x + offset,
            pivot[model].values,
            width=width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SHIFT_LABELS[s] for s in SHIFT_ORDER])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("OOD accuracy")
    ax.set_title("OOD Accuracy under Individual Distribution Shifts", pad=12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "ood_accuracy_grouped_bar.png", dpi=300)
    plt.close(fig)


def save_ood_drop_profile(ood, plot_dir):
    pivot = (
        ood.pivot(index="ood_shift", columns="model", values="ood_drop")
        .loc[SHIFT_ORDER, MODEL_ORDER]
    )

    x = np.arange(len(SHIFT_ORDER))
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for model in MODEL_ORDER:
        ax.plot(
            x,
            pivot[model].values,
            marker="o",
            markersize=6,
            linewidth=2.4,
            label=model,
            color=MODEL_COLORS[model],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SHIFT_LABELS[s] for s in SHIFT_ORDER])
    ax.set_ylabel("IID-to-OOD accuracy drop")
    ax.set_title("Robustness Drop Profile across OOD Shifts", pad=12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "ood_drop_profile.png", dpi=300)
    plt.close(fig)


def save_complexity_per_shift_scatter(ood, summary, plot_dir):
    merged = ood.merge(
        summary[["model", "effective_dim_pr", "conv_layers"]],
        on="model",
        how="left",
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True)
    axes = axes.flatten()

    for ax, shift in zip(axes, SHIFT_ORDER):
        subset = merged[merged["ood_shift"] == shift].copy()
        for _, row in subset.iterrows():
            model = str(row["model"])
            ax.scatter(
                row["effective_dim_pr"],
                row["ood_drop"],
                s=72,
                color=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            ax.annotate(
                model,
                (row["effective_dim_pr"], row["ood_drop"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8.5,
            )

        ax.set_title(SHIFT_LABELS[shift].replace("\n", " "))
        ax.set_ylabel("OOD drop")
        ax.grid(alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[2:]:
        ax.set_xlabel("Effective dimensionality (PR)")

    x_min = summary["effective_dim_pr"].min() - 0.08
    x_max = summary["effective_dim_pr"].max() + 0.08
    for ax in axes:
        ax.set_xlim(x_min, x_max)

    fig.suptitle("Effective Dimensionality vs Per-shift OOD Drop", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "complexity_vs_per_shift_ood_drop.png", dpi=300)
    plt.close(fig)


def main():
    args = parse_args()
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    ood, summary = prepare_data(args.ood_results, args.complexity_summary)

    save_ood_drop_heatmap(ood, plot_dir)
    save_ood_accuracy_grouped_bar(ood, plot_dir)
    save_ood_drop_profile(ood, plot_dir)
    save_complexity_per_shift_scatter(ood, summary, plot_dir)

    print(f"Saved auxiliary plots to: {plot_dir}")


if __name__ == "__main__":
    main()
