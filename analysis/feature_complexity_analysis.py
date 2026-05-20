import argparse
import ast
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PlainCNN(nn.Module):
    def __init__(self, blocks_per_stage, channels_per_stage, num_classes=10):
        super().__init__()
        layers = []
        in_channels = 3

        for stage_idx, (num_blocks, out_channels) in enumerate(
            zip(blocks_per_stage, channels_per_stage)
        ):
            for block_idx in range(num_blocks):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                layers.append(ConvBlock(in_channels, out_channels, stride=stride))
                in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels_per_stage[-1], num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.pool(x)
        features = torch.flatten(x, 1)
        logits = self.classifier(features)

        if return_features:
            return logits, features

        return logits


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract penultimate-layer CIFAR-10 features and compute "
            "effective dimensionality proxies for trained CNN checkpoints."
        )
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="analysis/outputs")
    parser.add_argument("--plot-dir", default="analysis/plots")
    parser.add_argument("--model-summary", default="results/model_summary.csv")
    parser.add_argument("--iid-results", default="results/final_iid_results.csv")
    parser.add_argument("--ood-summary", default="results/ood_summary.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Use auto to select CUDA when available.",
    )
    return parser.parse_args()


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(name)


def load_model_configs(model_summary_path):
    model_summary = pd.read_csv(model_summary_path)
    configs = {}

    for _, row in model_summary.iterrows():
        configs[row["model"]] = {
            "blocks_per_stage": ast.literal_eval(row["blocks_per_stage"]),
            "channels_per_stage": ast.literal_eval(row["channels_per_stage"]),
            "type": row.get("type", ""),
            "parameters": int(row["parameters"]),
            "parameters_M": float(row["parameters_M"]),
            "conv_layers": int(row["conv_layers"]),
        }

    return configs, model_summary


def make_test_loader(data_root, batch_size, num_workers, max_samples):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=test_transform,
    )

    if max_samples and max_samples > 0:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_features(model, loader, device):
    model.eval()
    feature_batches = []
    labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            logits, features = model(images, return_features=True)
            predictions = logits.argmax(dim=1)
            correct += predictions.eq(batch_labels).sum().item()
            total += batch_labels.numel()

            feature_batches.append(features.detach().cpu().numpy())
            labels.append(batch_labels.detach().cpu().numpy())

    feature_matrix = np.concatenate(feature_batches, axis=0)
    label_vector = np.concatenate(labels, axis=0)
    accuracy = correct / total
    return feature_matrix, label_vector, accuracy


def compute_spectrum_metrics(features):
    x = features.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    n = x.shape[0]
    cov = (x.T @ x) / max(n - 1, 1)

    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)

    total_energy = float(eigenvalues.sum())
    squared_energy = float(np.square(eigenvalues).sum())
    participation_ratio = (
        total_energy * total_energy / squared_energy if squared_energy > 0 else 0.0
    )

    normalized = eigenvalues / total_energy if total_energy > 0 else eigenvalues
    positive = normalized[normalized > 0]
    spectral_entropy = float(-(positive * np.log(positive)).sum())
    effective_rank = float(math.exp(spectral_entropy))
    cumulative = np.cumsum(normalized) if total_energy > 0 else normalized

    def components_for_energy(threshold):
        if cumulative.size == 0:
            return 0
        return int(np.searchsorted(cumulative, threshold, side="left") + 1)

    metrics = {
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "effective_dim_pr": float(participation_ratio),
        "effective_rank_entropy": effective_rank,
        "top1_energy": float(normalized[0]) if normalized.size else 0.0,
        "top5_energy": float(normalized[:5].sum()) if normalized.size else 0.0,
        "top10_energy": float(normalized[:10].sum()) if normalized.size else 0.0,
        "components_90_energy": components_for_energy(0.90),
        "components_95_energy": components_for_energy(0.95),
    }

    spectrum = pd.DataFrame(
        {
            "component": np.arange(1, len(eigenvalues) + 1),
            "eigenvalue": eigenvalues,
            "explained_energy": normalized,
            "cumulative_energy": cumulative,
        }
    )
    return metrics, spectrum


def merge_results(metrics_df, model_summary, iid_path, ood_path):
    merged = metrics_df.merge(
        model_summary[
            [
                "model",
                "blocks_per_stage",
                "channels_per_stage",
                "conv_layers",
                "type",
                "parameters",
                "parameters_M",
            ]
        ],
        on="model",
        how="left",
    )

    if Path(iid_path).exists():
        iid = pd.read_csv(iid_path)
        merged = merged.merge(iid, on="model", how="left")

    if Path(ood_path).exists():
        ood = pd.read_csv(ood_path)
        duplicate_cols = [
            col for col in ["best_test_acc", "generalization_gap"] if col in ood.columns
        ]
        if duplicate_cols:
            ood = ood.drop(columns=duplicate_cols)
        merged = merged.merge(ood, on="model", how="left")

    return merged


def annotate_points(ax, df, x_col, y_col):
    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=9,
        )


def save_plots(summary_df, spectrum_df, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary_df.sort_values("conv_layers")

    plt.figure(figsize=(7, 4.5))
    plt.bar(ordered["model"], ordered["effective_dim_pr"])
    plt.xlabel("Model")
    plt.ylabel("Participation ratio")
    plt.title("Effective Dimensionality of Penultimate Features")
    plt.tight_layout()
    plt.savefig(plot_dir / "effective_dimensionality_pr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    for model_name, group in spectrum_df.groupby("model", sort=False):
        plt.semilogy(group["component"], group["explained_energy"], label=model_name)
    plt.xlabel("Spectrum component")
    plt.ylabel("Explained energy")
    plt.title("Feature Covariance Spectrum")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "feature_spectrum_decay.png", dpi=300)
    plt.close()

    scatter_specs = [
        ("best_test_acc", "complexity_vs_iid_accuracy.png", "Best IID accuracy"),
        ("generalization_gap", "complexity_vs_generalization_gap.png", "Generalization gap"),
        ("mean_ood_drop", "complexity_vs_mean_ood_drop.png", "Mean OOD drop"),
        ("mean_ood_acc", "complexity_vs_mean_ood_accuracy.png", "Mean OOD accuracy"),
    ]

    for y_col, filename, ylabel in scatter_specs:
        if y_col not in summary_df.columns or summary_df[y_col].isna().all():
            continue

        plt.figure(figsize=(6, 4.5))
        plt.scatter(summary_df["effective_dim_pr"], summary_df[y_col], s=60)
        annotate_points(plt.gca(), summary_df, "effective_dim_pr", y_col)
        plt.xlabel("Effective dimensionality (participation ratio)")
        plt.ylabel(ylabel)
        plt.title(f"Effective Dimensionality vs {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=300)
        plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    plot_dir = Path(args.plot_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    configs, model_summary = load_model_configs(args.model_summary)
    loader = make_test_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    metric_rows = []
    spectrum_tables = []

    for model_name, cfg in configs.items():
        print(f"Processing {model_name}")
        model = PlainCNN(
            blocks_per_stage=cfg["blocks_per_stage"],
            channels_per_stage=cfg["channels_per_stage"],
        ).to(device)

        checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        model.load_state_dict(load_checkpoint(checkpoint_path, device))
        features, _, feature_test_acc = extract_features(model, loader, device)
        metrics, spectrum = compute_spectrum_metrics(features)

        metrics["model"] = model_name
        metrics["feature_test_acc"] = feature_test_acc
        metric_rows.append(metrics)

        spectrum.insert(0, "model", model_name)
        spectrum_tables.append(spectrum)

        print(
            f"  feature_dim={metrics['feature_dim']} "
            f"PR={metrics['effective_dim_pr']:.3f} "
            f"test_acc={feature_test_acc:.4f}"
        )

    metrics_df = pd.DataFrame(metric_rows)
    spectrum_df = pd.concat(spectrum_tables, ignore_index=True)
    summary_df = merge_results(
        metrics_df=metrics_df,
        model_summary=model_summary,
        iid_path=args.iid_results,
        ood_path=args.ood_summary,
    )

    metrics_path = results_dir / "complexity_metrics.csv"
    spectrum_path = results_dir / "feature_spectra.csv"
    summary_path = results_dir / "complexity_joined_summary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    spectrum_df.to_csv(spectrum_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    save_plots(summary_df, spectrum_df, plot_dir)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved spectra: {spectrum_path}")
    print(f"Saved joined summary: {summary_path}")
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
