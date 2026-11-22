"""Build figures, numeric summaries and the five-minute presentation from saved runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .presentation import build_presentation
from .visual_style import COLORS, INK, PLOT_STYLE, STYLES

LABELS = {"burgers": "FNO on viscous Burgers", "oscillator": "Laplace operator on forced oscillator"}


def _build(root):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    scores = pd.read_csv(root / "scores.csv")
    metrics = pd.read_csv(root / "effect_metrics.csv")
    summary = scores.groupby(["system", "regime", "split"])[["rel_l2", "qoi_rmse"]].agg(["mean", "std"])
    summary.to_csv(root / "score_summary.csv")
    metrics.groupby(["system", "regime", "method"]).mean(numeric_only=True).to_csv(root / "effect_summary.csv")
    checks = {}
    for system in LABELS:
        data = scores[scores.system == system]
        means = data.groupby(["regime", "split"]).rel_l2.mean()
        checks[system] = {
            "underfit_train_above_well_fit": bool(means["underfit", "train"] > 3 * means["well_fit", "train"]),
            "underfit_validation_above_well_fit": bool(means["underfit", "val"] > 3 * means["well_fit", "val"]),
            "overfit_train_below_well_fit": bool(means["overfit", "train"] < means["well_fit", "train"]),
            "overfit_validation_above_well_fit": bool(means["overfit", "val"] > means["well_fit", "val"]),
            "overfit_validation_training_ratio": float(means["overfit", "val"] / means["overfit", "train"]),
        }
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for j, metric in enumerate(["rel_l2", "qoi_rmse"]):
            for i, (regime, color) in enumerate(COLORS.items()):
                for k, split in enumerate(["train", "val", "test"]):
                    v = data[(data.regime == regime) & (data.split == split)][metric]
                    axes[j].bar(
                        i + (k - 1) * 0.24,
                        v.mean(),
                        width=0.22,
                        color=color,
                        alpha=[0.4, 0.7, 1][k],
                        yerr=v.std() if len(v) > 1 else None,
                        capsize=3,
                    )
            axes[j].set_xticks(range(3), ["Underfit", "Well-fit", "Overfit"])
            axes[j].set_ylabel("Mean relative field L2" if j == 0 else "QoI RMSE")
            axes[j].set_title("Train / validation / test (light to dark)")
        fig.suptitle(LABELS[system])
        fig.savefig(figures / f"{system}_fit.png", dpi=160)
        plt.close(fig)
        ref = np.load(root / f"{system}_oracle.npz")
        feature = "Viscosity" if system == "burgers" else "Forcing frequency"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for j, method in enumerate(["pdp", "ale"]):
            x = ref["grid"] if method == "pdp" else ref["edges"]
            axes[j].plot(x, ref[method], color=INK, lw=3, label="Simulator")
            for regime, color in COLORS.items():
                e = np.load(root / f"{system}_{regime}_effects.npz")[method]
                axes[j].plot(x, e.mean(0), color=color, ls=STYLES[regime], lw=2, label=regime.replace("_", " "))
                if len(e) > 1:
                    axes[j].fill_between(
                        x, e.mean(0) - e.std(0, ddof=1), e.mean(0) + e.std(0, ddof=1), color=color, alpha=0.15
                    )
            axes[j].set(title=method.upper(), xlabel=feature, ylabel="Centered RMS effect")
        axes[0].legend(fontsize=8)
        fig.suptitle(LABELS[system] + "; bands = one refit SD")
        fig.savefig(figures / f"{system}_effects.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
        for ax, (regime, color) in zip(axes, COLORS.items(), strict=True):
            e = np.load(root / f"{system}_{regime}_effects.npz")["ice"]
            for i in range(0, 128, 8):
                ax.plot(ref["grid"], ref["ice"][i], color="black", alpha=0.22, lw=0.8)
                ax.plot(ref["grid"], e[0, i], color=color, alpha=0.6, lw=0.8)
            ax.set(title=regime.replace("_", " "), xlabel=feature)
        axes[0].set_ylabel("Centered individual RMS effect")
        fig.suptitle("Same 16 ICE instances; black = simulator, color = first refit")
        fig.savefig(figures / f"{system}_ice.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, (regime, color) in zip(axes, COLORS.items(), strict=True):
            for f in sorted(root.glob(f"{system}_{regime}_*_history.csv")):
                h = pd.read_csv(f)
                ax.semilogy(h.step, h.train_mse, color=color, alpha=0.6)
                ax.semilogy(h.step, h.val_mse, color=color, ls="--", alpha=0.6)
            ax.set(title=regime.replace("_", " "), xlabel="Optimization step", ylabel="Scaled MSE")
        fig.suptitle("Learning histories: solid train, dashed validation")
        fig.savefig(figures / f"{system}_learning.png", dpi=160)
        plt.close(fig)
    (root / "regime_checks.json").write_text(json.dumps(checks, indent=2))
    build_presentation(root)
    return checks


def build(root=Path("results/study")):
    with plt.rc_context(PLOT_STYLE):
        return _build(root)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/study"))
    build(p.parse_args().output)
