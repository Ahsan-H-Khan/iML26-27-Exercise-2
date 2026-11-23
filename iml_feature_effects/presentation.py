"""Build native LaTeX Beamer slides with measured, vector-plot assets.

The editable slide source is presentation/presentation.tex. Requires pdflatex,
Beamer, TikZ, amsmath, booktabs and Latin Modern fonts. No training is performed.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from .co12 import ensure, latex_rows
from .operators import Operator
from .physics import coordinates, encode, sample, solve
from .visual_style import BLUE, COLORS, DISPLAY, INK, MUTED, PLOT_STYLE, STYLES, TEAL


def _curve(ax, root, system, method="pdp"):
    ref = np.load(root / f"{system}_oracle.npz")
    x = ref["edges"] if method == "ale" else ref["grid"]
    ax.plot(x, ref[method], color=INK, lw=3, zorder=2)
    for regime, color in COLORS.items():
        values = np.load(root / f"{system}_{regime}_effects.npz")[method]
        mean, sd = values.mean(0), values.std(0, ddof=1)
        ax.plot(x, mean, color=color, ls=STYLES[regime], lw=2, zorder=3)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)
    ax.set_xlabel("Viscosity" if system == "burgers" else "Forcing frequency")
    ax.set_ylabel("Centered RMS effect")
    ax.grid(axis="y", linewidth=0.6)
    ax.set_axisbelow(True)


def _legend(fig, y=1.01):
    handles = [Line2D([0], [0], color=INK, lw=3, label="Simulator")]
    handles += [Line2D([0], [0], color=c, ls=STYLES[r], lw=2, label=DISPLAY[r]) for r, c in COLORS.items()]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=4,
        fontsize=10,
        handlelength=2.4,
        columnspacing=1.7,
    )


def _assets(root, destination):
    """Only visual assets; all equations and diagram labels are native LaTeX."""
    destination.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(destination / f"{name}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    with plt.rc_context(PLOT_STYLE):
        for system in ["burgers", "oscillator"]:
            p = sample(system, 1, 101)
            xx = coordinates(system)
            signal = encode(system, p)[0, 0]
            target = solve(system, p)[0]
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.65), layout="constrained")
            for ax, yy, color in zip(
                axes, [signal, target], [MUTED, BLUE if system == "burgers" else TEAL], strict=True
            ):
                ax.plot(xx, yy, color=color, lw=2)
                ax.axhline(0, color=MUTED, lw=0.5, alpha=0.25)
                ax.set_yticks([])
                ax.set_xlabel("Space" if system == "burgers" else "Time", fontsize=10)
                ax.tick_params(labelsize=9)
            axes[0].set_title(
                "Input: initial profile" if system == "burgers" else "Input: forcing history", fontsize=11
            )
            axes[1].set_title("Target: later profile" if system == "burgers" else "Target: displacement", fontsize=11)
            save(fig, f"{system}_io")
        fig, axes = plt.subplots(1, 2, figsize=(9.7, 3), layout="constrained")
        for ax, system in zip(axes, ["burgers", "oscillator"], strict=True):
            _curve(ax, root, system)
            ax.set_title("Burgers" if system == "burgers" else "Forced oscillator", fontsize=13)
        _legend(fig, 1.13)
        save(fig, "pdp_results")
        metrics = pd.read_csv(root / "effect_metrics.csv")
        bm = (
            metrics[(metrics.system == "burgers") & (metrics.method == "pdp")].groupby("regime").mean(numeric_only=True)
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.65), layout="constrained")
        for ax, key, title in zip(
            axes, ["effect_nrmse", "refit_dispersion"], ["Effect error", "Variation across refits"], strict=True
        ):
            values = [bm.loc[r, key] for r in ["well_fit", "overfit"]]
            ax.bar([0, 1], values, color=[BLUE, COLORS["overfit"]], width=0.5)
            ax.set_xticks([0, 1], ["Well-fit", "Overfit"])
            ax.set_title(title + " (lower is better)", fontsize=11)
            ax.set_ylim(0, max(values) * 1.35)
            ax.grid(axis="y", lw=0.5)
            ax.set_axisbelow(True)
            for i, value in enumerate(values):
                ax.text(i, value + max(values) * 0.035, f"{value:.3f}", ha="center", fontsize=11)
        save(fig, "stability")
        ref = np.load(root / "oscillator_oracle.npz")
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.1), layout="constrained")
        for i in range(0, 128, 16):
            axes[0].plot(ref["grid"], ref["ice"][i], color=BLUE, alpha=0.5, lw=1)
            axes[1].plot(ref["grid"], ref["ice"][i], color=MUTED, alpha=0.15, lw=0.8)
        axes[1].plot(ref["grid"], ref["pdp"], color=BLUE, lw=2)
        axes[2].plot(ref["edges"], ref["ale"], color=TEAL, marker="o", ms=3, lw=2)
        for ax, title in zip(axes, ["Individual curves", "Average curve", "Local changes accumulated"], strict=True):
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Forcing frequency", fontsize=10)
            ax.set_yticks([])
            ax.tick_params(labelsize=9)
        save(fig, "methods")
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 4.35), layout="constrained")
        for i, system in enumerate(["burgers", "oscillator"]):
            for j, method in enumerate(["pdp", "ale"]):
                _curve(axes[i, j], root, system, method)
                axes[i, j].set_title(f"{'Burgers' if i == 0 else 'Oscillator'} · {method.upper()}", fontsize=12)
        _legend(fig, 1.08)
        save(fig, "global_effects")
        fig, axes = plt.subplots(2, 3, figsize=(10.8, 4.2), layout="constrained")
        for i, system in enumerate(["burgers", "oscillator"]):
            ref = np.load(root / f"{system}_oracle.npz")
            for j, regime in enumerate(COLORS):
                ax = axes[i, j]
                curves = np.load(root / f"{system}_{regime}_effects.npz")["ice"][0]
                for k in range(0, 128, 8):
                    ax.plot(ref["grid"], ref["ice"][k], color=INK, alpha=0.25, lw=0.8)
                    ax.plot(ref["grid"], curves[k], color=COLORS[regime], ls=STYLES[regime], alpha=0.65, lw=1)
                if i == 0:
                    ax.set_title(DISPLAY[regime], fontsize=12, color=COLORS[regime])
                if j == 0:
                    ax.set_ylabel(("Burgers" if i == 0 else "Oscillator") + "\nCentered RMS effect")
                ax.set_xlabel("Viscosity" if i == 0 else "Forcing frequency")
                ax.tick_params(labelsize=9)
        save(fig, "ice")


def _values(root, destination):
    """Numeric TeX rows come from the actual checkpoints and recorded metrics."""
    ensure(root)
    scores = pd.read_csv(root / "scores.csv")
    means = scores.groupby(["system", "regime", "split"]).rel_l2.mean() * 100
    lines = []

    def macro(name, value):
        lines.append("\\newcommand{\\" + name + "}{" + value + "}")

    rows = []
    for regime in COLORS:
        values = [
            f"{means[system, regime, split]:.2f}\\%"
            for system in ["burgers", "oscillator"]
            for split in ["train", "test"]
        ]
        rows.append(
            r"\textcolor{" + regime.replace("_", "") + "}{" + DISPLAY[regime] + "} & " + " & ".join(values) + r" \\"
        )
    macro("ScoreRows", "\n".join(rows))
    rows = []
    counts = []
    for system, model in [("burgers", "FNO"), ("oscillator", "LNO")]:
        for regime in COLORS:
            payload = torch.load(root / f"{system}_{regime}_0.pt", weights_only=True)
            cfg = payload["config"]
            label = r"\textcolor{" + regime.replace("_", "") + "}{" + DISPLAY[regime] + "}"
            variant = "linear" if cfg.get("linear", False) else "nonlinear"
            rows.append(
                f"{model} & {label} & {cfg['width']} & {cfg['modes']} & {cfg['depth']} "
                f"& {cfg['wd']:g} & {cfg['steps']:,}" + r" \\"
            )
            net = Operator(system, cfg["width"], cfg["modes"], cfg["depth"], linear=cfg.get("linear", False))
            dof = sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())
            counts.append(f"{model} & {label} & {variant} & {dof:,}" + r" \\")
    macro("ConfigRows", "\n".join(rows))
    macro("CountRows", "\n".join(counts))
    macro("CoBurgersRows", latex_rows(root, "burgers"))
    macro("CoOscillatorRows", latex_rows(root, "oscillator"))
    destination.write_text("\n".join(lines) + "\n")


def build_presentation(root=Path("results/study"), target=Path("presentation/presentation.pdf")):
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    template = target.parent / "presentation.tex"
    compiler = shutil.which("pdflatex")
    if not compiler:
        raise RuntimeError("Presentation requires pdflatex with Beamer, TikZ, amsmath, booktabs and lmodern.")
    if not template.exists():
        raise FileNotFoundError(f"Editable slide source not found: {template}")
    generated = target.parent / "generated"
    _assets(root, generated)
    _values(root, generated / "values.tex")
    with tempfile.TemporaryDirectory(prefix="iml-latex-") as temp:
        command = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={temp}",
            template.name,
        ]
        for _ in range(2):
            process = subprocess.run(command, cwd=target.parent, text=True, capture_output=True)
            log = Path("/tmp/iml-presentation-latex.log")
            log.write_text(process.stdout + process.stderr)
            if process.returncode:
                raise RuntimeError(f"LaTeX failed; see {log}\n" + process.stdout[-3500:])
        output = Path(temp) / "presentation.pdf"
        shutil.copyfile(output, target)
    renderer = shutil.which("pdftoppm")
    if renderer:
        for page, name in [(2, "preview"), (11, "co12-preview")]:
            subprocess.run(
                [
                    renderer,
                    "-f",
                    str(page),
                    "-singlefile",
                    "-scale-to",
                    "1600",
                    "-png",
                    str(target),
                    str(target.with_name(name)),
                ],
                check=True,
                capture_output=True,
            )
