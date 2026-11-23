"""Vector presentation: beginner explanations, measured protocol and study results.

Nine narrated slides total five minutes; three backup slides support questions.
Every result is read from saved experimental outputs. Introductory wave diagrams
are explicitly conceptual illustrations rather than model measurements.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from .physics import coordinates, encode, sample, solve
from .visual_style import BG, BLUE, BORDER, COLORS, CORAL, DISPLAY, INK, MUTED, PLOT_STYLE, SLATE, STYLES, TEAL


def text(fig, x, y, content, size=18, color=INK, weight="normal", **kwargs):
    return fig.text(x, y, content, fontsize=size, color=color, weight=weight, va="top", **kwargs)


def card(fig, x, y, w, h, color="white", edge=BORDER):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        transform=fig.transFigure,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=color,
        zorder=-1,
    )
    fig.add_artist(patch)


def arrow(fig, start, end, color=MUTED, style="-|>", connection="arc3"):
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle=style,
            mutation_scale=15,
            linewidth=1.7,
            color=color,
            connectionstyle=connection,
        )
    )


def frame(number, tag, title, subtitle, takeaway=None, appendix=False):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.add_artist(Rectangle((0, 0.976), 1, 0.024, transform=fig.transFigure, facecolor=BLUE, linewidth=0))
    text(fig, 0.045, 0.942, tag.upper(), 11, BLUE, "bold")
    text(fig, 0.045, 0.892, title, 29, weight="bold")
    text(fig, 0.045, 0.823, subtitle, 15, MUTED)
    if takeaway:
        card(fig, 0.045, 0.08, 0.91, 0.065, "#E6EEF9", edge="#E6EEF9")
        text(fig, 0.061, 0.126, takeaway, 15, BLUE, "bold")
    text(fig, 0.045, 0.038, "iML 26/27  ·  Feature effects in neural operators", 10, MUTED)
    text(fig, 0.955, 0.038, f"{'BACKUP ' if appendix else ''}{number:02d}", 10, MUTED, ha="right")
    return fig


def chart(fig, rect, *, xlabel=None, ylabel=None):
    ax = fig.add_axes(rect)
    ax.tick_params(labelsize=10, length=3)
    ax.grid(axis="y", linewidth=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, labelpad=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, labelpad=7)
    return ax


def spark(fig, rect, x, y, color=BLUE, label=None):
    ax = fig.add_axes(rect)
    ax.plot(x, y, color=color, lw=2)
    ax.axhline(0, color=BORDER, lw=0.7, zorder=-1)
    ax.set_xlim(x.min(), x.max())
    ax.axis("off")
    if label:
        ax.text(0.5, -0.06, label, transform=ax.transAxes, fontsize=11, color=MUTED, ha="center")
    return ax


def curve_plot(ax, root, system, method="pdp", bands=True):
    ref = np.load(root / f"{system}_oracle.npz")
    x = ref["edges"] if method == "ale" else ref["grid"]
    # Broad oracle underneath, distinct dashes/colors on top retain all identities.
    ax.plot(x, ref[method], color=INK, lw=3.2, label="Simulator reference", zorder=2)
    for regime in COLORS:
        values = np.load(root / f"{system}_{regime}_effects.npz")[method]
        mean = values.mean(0)
        ax.plot(x, mean, color=COLORS[regime], ls=STYLES[regime], lw=2.1, label=DISPLAY[regime], zorder=3)
        if bands:
            sd = values.std(0, ddof=1)
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[regime], alpha=0.11, linewidth=0)
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.35)


def legend(fig, y=0.72):
    handles = [Line2D([0], [0], color=INK, lw=3, label="Simulator reference")]
    handles += [Line2D([0], [0], color=c, ls=STYLES[r], lw=2.2, label=DISPLAY[r]) for r, c in COLORS.items()]
    fig.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.5, y), ncol=4, fontsize=12, columnspacing=2.7, handlelength=3
    )


def build_presentation(root=Path("results/study"), target=Path("presentation/presentation.pdf")):
    target.parent.mkdir(exist_ok=True)
    scores = pd.read_csv(root / "scores.csv")
    metrics = pd.read_csv(root / "effect_metrics.csv")
    p = sample("burgers", 1, 101)
    x = coordinates("burgers")
    initial = encode("burgers", p)[0, 0]
    solution = solve("burgers", p)[0]
    with (
        plt.rc_context(PLOT_STYLE),
        PdfPages(
            target,
            metadata={
                "Title": "Can we trust a model's explanations?",
                "Subject": "Neural operators, controlled fit regimes and simulator-referenced feature effects",
                "Author": "Ahsan H. Khan · prepared with OpenAI Codex",
            },
        ) as pdf,
    ):

        def save(fig):
            if pdf.get_pagecount() == 2:
                fig.savefig(target.with_name("preview.png"), dpi=110, facecolor=BG)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)

        # 1. The function-to-function task, followed by the explanation question.
        fig = frame(
            1,
            "The question",
            "Can we trust a model's explanations?",
            "We study how underfitting and overfitting change the physical story a model tells.",
            "A good-looking explanation still needs a reference check.",
        )
        card(fig, 0.045, 0.29, 0.91, 0.45)
        text(fig, 0.075, 0.69, "INPUT FUNCTION", 12, MUTED, "bold")
        text(fig, 0.42, 0.69, "NEURAL OPERATOR", 12, BLUE, "bold")
        text(fig, 0.733, 0.69, "OUTPUT FUNCTION", 12, MUTED, "bold")
        spark(fig, [0.075, 0.42, 0.22, 0.20], x, initial, SLATE, "initial flow profile")
        card(fig, 0.39, 0.43, 0.22, 0.17, "#E6EEF9", "#E6EEF9")
        text(fig, 0.5, 0.565, r"$\widehat{u}=G_\theta(u_0,\nu)$", 24, BLUE, ha="center")
        text(fig, 0.5, 0.492, "learned from examples", 13, MUTED, ha="center")
        spark(fig, [0.71, 0.42, 0.22, 0.20], x, solution, BLUE, "later flow (prediction target)")
        arrow(fig, (0.31, 0.515), (0.375, 0.515))
        arrow(fig, (0.625, 0.515), (0.697, 0.515))
        text(fig, 0.075, 0.35, "Here: one complete curve goes in, another comes out.", 17)
        text(fig, 0.065, 0.238, "Explanation question", 14, BLUE, "bold")
        text(fig, 0.27, 0.238, "If viscosity changes, how does the predicted flow strength change?", 18)
        save(fig)

        # 2. Actual physics, actual analytic input/output curves, concise notation.
        fig = frame(
            2,
            "Two datasets",
            "Two systems with known physical solutions",
            "Known simulators let us compare each model's explanation with a reference explanation.",
            "Both models predict a full function; the explanation targets its RMS strength.",
        )
        for left in [0.045, 0.515]:
            card(fig, left, 0.26, 0.44, 0.48)
        text(fig, 0.07, 0.707, "1D viscous Burgers flow", 21, weight="bold")
        text(fig, 0.07, 0.651, r"$u_t+u\,u_x=\nu\,u_{xx}$", 23, BLUE)
        text(fig, 0.07, 0.585, "viscosity " + r"$\nu$" + " controls smoothing", 15, MUTED)
        ax = chart(fig, [0.078, 0.345, 0.365, 0.18], xlabel="Space")
        ax.plot(x, initial, color=SLATE, lw=1.6, label="initial profile")
        ax.plot(x, solution, color=BLUE, lw=2, label="at time T = 1")
        ax.set_yticks([])
        ax.set_xticks([0, np.pi, 2 * np.pi], ["0", r"$\pi$", r"$2\pi$"])
        ax.legend(fontsize=10, loc="upper right")
        text(fig, 0.54, 0.707, "Forced, damped oscillator", 21, weight="bold")
        text(fig, 0.54, 0.651, r"$\ddot{x}+0.4\dot{x}+4x=A\cos(\omega t+\phi)$", 22, TEAL)
        text(fig, 0.54, 0.585, "forcing frequency " + r"$\omega$" + " controls response", 15, MUTED)
        op = np.array([[2.0, 1.0, 0.0, 0.15]])
        t = coordinates("oscillator")
        ax = chart(fig, [0.548, 0.345, 0.365, 0.18], xlabel="Time")
        ax.plot(t, encode("oscillator", op)[0, 0], color=SLATE, lw=1.3, label="forcing")
        ax.plot(t, solve("oscillator", op)[0], color=TEAL, lw=2, label="displacement")
        ax.set_yticks([])
        ax.legend(fontsize=10, loc="upper right")
        text(fig, 0.07, 0.223, r"$Q(y)=\sqrt{\mathrm{mean}(y^2)}$", 22, BLUE)
        text(fig, 0.39, 0.218, "RMS = typical magnitude across space or time.", 17)
        save(fig)

        # 3. FNO: one spectral layer, with the local path represented explicitly.
        fig = frame(
            3,
            "FNO · Fourier neural operator",
            "FNO learns how to combine spatial waves",
            "A curve can be represented as a sum of waves with different spatial frequencies.",
            "Learned Fourier weights mix information across the spatial domain.",
        )
        nodes = [
            (0.055, "FEATURE CURVE"),
            (0.30, "FOURIER MODES"),
            (0.555, "LEARNED WEIGHTS"),
            (0.805, "UPDATED CURVE"),
        ]
        for left, label in nodes:
            card(fig, left, 0.43, 0.14 if left > 0.8 else 0.17, 0.28)
            text(fig, left + 0.012, 0.68, label, 10, MUTED, "bold")
        z = np.linspace(0, 2 * np.pi, 160)
        spark(fig, [0.068, 0.475, 0.14, 0.12], z, np.sin(z) + 0.35 * np.sin(3 * z), BLUE)
        ax = fig.add_axes([0.32, 0.475, 0.13, 0.12])
        ax.bar([1, 2, 3, 4], [1, 0, 0.35, 0], color=[BLUE, SLATE, CORAL, SLATE], width=0.45)
        ax.set(xticks=[1, 2, 3, 4], yticks=[])
        ax.tick_params(labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        text(fig, 0.64, 0.59, r"$R_\ell(k)$", 29, BLUE, ha="center")
        text(fig, 0.64, 0.507, "scale + mix modes", 12, MUTED, ha="center")
        spark(fig, [0.817, 0.475, 0.118, 0.12], z, 0.65 * np.sin(z) + 0.12 * np.sin(3 * z), BLUE)
        for a, b, label in [(0.23, 0.29, "transform"), (0.48, 0.545, "learn"), (0.735, 0.795, "inverse")]:
            arrow(fig, (a, 0.55), (b, 0.55))
            text(fig, (a + b) / 2, 0.425, label, 11, MUTED, ha="center")
        # Local mixing bypass below the spectral branch, before activation.
        arrow(fig, (0.135, 0.43), (0.135, 0.365), SLATE)
        arrow(fig, (0.135, 0.365), (0.875, 0.365), SLATE)
        arrow(fig, (0.875, 0.365), (0.875, 0.43), SLATE)
        text(
            fig,
            0.50,
            0.385,
            "+ local channel mixing",
            12,
            MUTED,
            ha="center",
            bbox={"facecolor": BG, "edgecolor": "none", "pad": 4},
        )
        text(
            fig,
            0.5,
            0.296,
            r"$v_{\ell+1}=\sigma\!\left(W_\ell v_\ell+\mathcal{F}^{-1}[R_\ell\mathcal{F}(v_\ell)]\right)$",
            26,
            BLUE,
            ha="center",
        )
        text(
            fig,
            0.5,
            0.208,
            r"$v$: hidden feature curves    $\mathcal{F}$: Fourier transform    $\sigma$: activation",
            15,
            MUTED,
            ha="center",
        )
        save(fig)

        # 4. LNO: inverse Laplace kernels are damped oscillations; causal convolution.
        fig = frame(
            4,
            "LNO · Laplace neural operator",
            "LNO learns a system's decaying memory",
            "Past forcing contributes to the current response through a learned causal kernel.",
            "In this custom LNO, stable poles set decay and frequency; residues set the mixture.",
        )
        for left, w in [(0.045, 0.27), (0.36, 0.28), (0.685, 0.27)]:
            card(fig, left, 0.405, w, 0.32)
        text(fig, 0.068, 0.688, "PAST FORCING", 12, MUTED, "bold")
        text(fig, 0.385, 0.688, "LEARNED MEMORY", 12, MUTED, "bold")
        text(fig, 0.708, 0.688, "RESPONSE AT TIME t", 12, MUTED, "bold")
        tt = np.linspace(0, 8, 200)
        spark(fig, [0.07, 0.47, 0.22, 0.15], tt, np.cos(2 * tt), SLATE)
        ax = fig.add_axes([0.385, 0.465, 0.23, 0.16])
        for decay, freq, c in [(0.25, 1.3, BLUE), (0.65, 2.2, TEAL), (1.0, 3.2, CORAL)]:
            ax.plot(tt, np.exp(-decay * tt) * np.sin(freq * tt), color=c, lw=1.7)
        ax.axhline(0, color=BORDER, lw=0.7)
        ax.axis("off")
        spark(fig, [0.71, 0.47, 0.22, 0.15], tt, np.sin(2 * tt) * (1 - np.exp(-0.25 * tt)), TEAL)
        arrow(fig, (0.325, 0.56), (0.35, 0.56))
        arrow(fig, (0.65, 0.56), (0.675, 0.56))
        text(
            fig,
            0.5,
            0.375,
            r"$K(t)=\sum_j e^{-\alpha_j t}\,[a_j\cos(\beta_j t)+b_j\sin(\beta_j t)]$",
            25,
            TEAL,
            ha="center",
        )
        text(fig, 0.5, 0.294, r"$h(t)=\int_0^t K(t-\tau)\,v(\tau)\,d\tau$", 25, TEAL, ha="center")
        text(
            fig,
            0.5,
            0.212,
            r"$\alpha_j>0$: decay    $\beta_j$: frequency    $a_j,b_j$: learned amplitudes",
            15,
            MUTED,
            ha="center",
        )
        text(
            fig,
            0.5,
            0.173,
            "One temporal layer, plus local/channel mixing; diagrams are conceptual.",
            12,
            MUTED,
            ha="center",
        )
        save(fig)

        # 5. All split counts and repetition pairing shown in a single workflow.
        fig = frame(
            5,
            "Experimental protocol",
            "Change the model fit; keep the comparison fixed",
            "Each complete function is one sample. No time or spatial points are split across sets.",
            "2 systems × 3 fit regimes × 3 matched refits = 18 trained models.",
        )
        card(fig, 0.045, 0.34, 0.20, 0.38)
        text(fig, 0.066, 0.683, "GENERATE DATA", 12, BLUE, "bold")
        text(fig, 0.066, 0.619, "Known simulator", 19, weight="bold")
        text(
            fig,
            0.066,
            0.558,
            "Burgers or oscillator\nIndependent parameters\nNo added target noise",
            14,
            MUTED,
            linespacing=1.6,
        )
        text(fig, 0.066, 0.385, "Reference solutions", 13, BLUE)
        arrow(fig, (0.255, 0.53), (0.29, 0.53))
        card(fig, 0.30, 0.34, 0.25, 0.38)
        text(fig, 0.32, 0.683, "FIX THE SPLITS", 12, BLUE, "bold")
        rows = [
            ("Train", "48 from a pool of 512"),
            ("Validation", "128 · select checkpoint"),
            ("Explain", "128 · fixed background"),
            ("Final test", "128 · report errors"),
        ]
        for i, (a, b) in enumerate(rows):
            yy = 0.615 - i * 0.067
            text(fig, 0.32, yy, a, 14, weight="bold")
            text(fig, 0.397, yy, b, 11.5, MUTED)
        arrow(fig, (0.56, 0.53), (0.597, 0.53))
        text(fig, 0.62, 0.73, "FIT THREE CONFIGURATIONS", 12, BLUE, "bold")
        for i, (regime, color) in enumerate(COLORS.items()):
            yy = 0.588 - i * 0.105
            card(fig, 0.61, yy, 0.325, 0.079, "white", color)
            text(fig, 0.626, yy + 0.055, DISPLAY[regime], 16, color, "bold")
            label = {
                "underfit": "too restricted",
                "well_fit": "validation-selected",
                "overfit": "larger; fit training closely",
            }[regime]
            text(fig, 0.76, yy + 0.051, label, 11, MUTED)
        text(fig, 0.065, 0.265, "REPEAT ×3", 13, BLUE, "bold")
        text(fig, 0.205, 0.265, "Resample training functions + change initialization; pair them across regimes.", 16)
        text(
            fig,
            0.205,
            0.205,
            "Validation, explanation background, feature grid and final test remain fixed.",
            14,
            MUTED,
        )
        save(fig)

        # 6. Explain methods with actual saved simulator curves and local differences.
        fig = frame(
            6,
            "Feature effects",
            "One question, three views of the model",
            "Change one physical parameter, rebuild the input signal, then measure predicted RMS.",
            "Compute each explanation again with the simulator: that is the method-matched reference.",
        )
        ref = np.load(root / "oscillator_oracle.npz")
        grid = ref["grid"]
        labels = [
            ("ICE", "Individual conditional expectation", "One curve per instance"),
            ("PDP", "Partial dependence plot", "Average the ICE curves"),
            ("ALE", "Accumulated local effects", "Accumulate local changes"),
        ]
        for left, (title, full_name, desc) in zip([0.045, 0.36, 0.675], labels, strict=True):
            card(fig, left, 0.35, 0.28, 0.39)
            text(fig, left + 0.018, 0.706, title, 23, BLUE, "bold")
            text(fig, left + 0.018, 0.652, full_name, 11, MUTED)
            text(fig, left + 0.018, 0.614, desc, 13, INK)
        ax = chart(fig, [0.079, 0.407, 0.21, 0.18], xlabel="Forcing frequency")
        for i in range(0, 128, 16):
            ax.plot(grid, ref["ice"][i], color=BLUE, alpha=0.45, lw=1)
        ax.set_yticks([])
        ax = chart(fig, [0.394, 0.407, 0.21, 0.18], xlabel="Forcing frequency")
        for i in range(0, 128, 16):
            ax.plot(grid, ref["ice"][i], color=SLATE, alpha=0.14, lw=1)
        ax.plot(grid, ref["pdp"], color=BLUE, lw=2.4)
        ax.set_yticks([])
        ax = chart(fig, [0.709, 0.407, 0.21, 0.18], xlabel="Forcing frequency")
        ax.plot(ref["edges"], ref["ale"], color=TEAL, marker="o", markersize=3, lw=2)
        for xx in ref["edges"]:
            ax.axvline(xx, color=BORDER, ls=":", lw=0.6)
        ax.set_yticks([])
        text(fig, 0.5, 0.306, r"$h_\theta(z,w)=Q\!\left(G_\theta(\mathrm{input}(z,w))\right)$", 25, BLUE, ha="center")
        text(
            fig,
            0.5,
            0.225,
            r"$z$: viscosity or frequency    $w$: other parameters    $Q$: RMS of the output",
            15,
            MUTED,
            ha="center",
        )
        text(
            fig,
            0.5,
            0.177,
            "128 instances · 41 ICE/PDP values · 8 ALE bins · simulator curves shown",
            12,
            MUTED,
            ha="center",
        )
        save(fig)

        # 7. A readable train -> test table makes the generalization gap explicit.
        fig = frame(
            7,
            "Fit diagnostics",
            "The measured errors support all three regimes",
            "Mean relative function error across 3 matched refits; lower is better.",
            "Overfit = very small training error, with a larger error on unseen functions.",
        )
        card(fig, 0.045, 0.27, 0.91, 0.47)
        text(fig, 0.07, 0.696, "CONFIGURATION", 12, MUTED, "bold")
        text(fig, 0.405, 0.696, "BURGERS · FNO", 14, BLUE, "bold", ha="center")
        text(fig, 0.77, 0.696, "OSCILLATOR · LNO", 14, TEAL, "bold", ha="center")
        for xx in [0.405, 0.77]:
            text(fig, xx, 0.647, "training → final test", 13, MUTED, ha="center")
        means = scores.groupby(["system", "regime", "split"]).rel_l2.mean()
        for i, (regime, color) in enumerate(COLORS.items()):
            yy = 0.562 - i * 0.105
            fig.add_artist(
                Rectangle((0.066, yy - 0.033), 0.007, 0.046, transform=fig.transFigure, facecolor=color, linewidth=0)
            )
            text(fig, 0.088, yy, DISPLAY[regime], 20, color, "bold")
            for xx, system in [(0.405, "burgers"), (0.77, "oscillator")]:
                train = 100 * means[system, regime, "train"]
                test = 100 * means[system, regime, "test"]
                text(fig, xx, yy, f"{train:.2f}%  →  {test:.2f}%", 24, ha="center", weight="bold")
        text(fig, 0.065, 0.224, "Small gaps alone are insufficient: underfit models have high errors on both sets.", 16)
        text(
            fig,
            0.065,
            0.177,
            "Oscillator well-fit model: linear pole/residue stack, reflecting the linear physical system.",
            12,
            MUTED,
        )
        save(fig)

        # 8. Main results focus on PDP; full ALE and ICE are in the backup slides.
        fig = frame(
            8,
            "Explanation results",
            "Underfitting weakens the inferred physical effects",
            "PDP = average effect on output RMS. Compare every curve with the simulator.",
            "Underfit models understate viscosity sensitivity and the contrast near resonance.",
        )
        legend(fig, 0.737)
        for left, system, title in [
            (0.075, "burgers", "Burgers: viscosity effect"),
            (0.56, "oscillator", "Oscillator: frequency effect"),
        ]:
            ax = chart(
                fig,
                [left, 0.35, 0.38, 0.32],
                xlabel="Viscosity" if system == "burgers" else "Forcing frequency",
                ylabel="Centered RMS effect" if system == "burgers" else None,
            )
            curve_plot(ax, root, system)
            ax.set_title(title, fontsize=16, pad=12)
            m = metrics[(metrics.system == system) & (metrics.method == "pdp")].groupby("regime").effect_nrmse.mean()
            text(fig, left, 0.242, "NORMALIZED PDP ERROR  ↓", 11, MUTED, "bold")
            for j, regime in enumerate(COLORS):
                text(
                    fig,
                    left + j * 0.133,
                    0.203,
                    f"{DISPLAY[regime]}  {m[regime]:.3f}",
                    12,
                    COLORS[regime],
                    weight="bold",
                )
        text(
            fig,
            0.5,
            0.163,
            "Shading: ±1 standard deviation across 3 refits; descriptive spread, not a confidence interval.",
            11,
            MUTED,
            ha="center",
        )
        save(fig)

        # 9. One measured counterexample to the claim that stable means correct.
        bm = (
            metrics[(metrics.system == "burgers") & (metrics.method == "pdp")].groupby("regime").mean(numeric_only=True)
        )
        fig = frame(
            9,
            "Takeaway",
            "Repeatable does not always mean reliable",
            "Burgers PDP: the overfit model is more stable across refits, yet farther from the simulator.",
            "Trust explanations by checking prediction error, effect accuracy and repeatability together.",
        )
        for left, key, title in [
            (0.075, "effect_nrmse", "Explanation error"),
            (0.38, "refit_dispersion", "Variation across refits"),
        ]:
            card(fig, left - 0.025, 0.30, 0.27, 0.43)
            ax = chart(fig, [left, 0.375, 0.21, 0.265], ylabel="Normalized score")
            values = [bm.loc[r, key] for r in ["well_fit", "overfit"]]
            ax.bar([0, 1], values, color=[BLUE, CORAL], width=0.5)
            ax.set_xticks([0, 1], ["Well-fit", "Overfit"])
            ax.set_ylim(0, max(values) * 1.35)
            ax.set_title(title + " ↓", fontsize=14, pad=15)
            for j, v in enumerate(values):
                ax.text(j, v + max(values) * 0.04, f"{v:.3f}", ha="center", fontsize=13, color=INK, weight="bold")
        text(fig, 0.70, 0.69, "WHAT THIS MEANS", 12, BLUE, "bold")
        text(fig, 0.70, 0.627, "Stable can still be biased.", 19, weight="bold")
        text(fig, 0.70, 0.556, "A smooth curve can still\nunderstate a physical effect.", 16, MUTED, linespacing=1.5)
        text(fig, 0.70, 0.434, "Scope", 15, weight="bold")
        text(
            fig,
            0.70,
            0.382,
            "Three refits, restricted inputs;\ncustom LNO with a linear baseline.\nNo FNO-vs-LNO superiority claim.",
            13,
            MUTED,
            linespacing=1.6,
        )
        text(
            fig,
            0.065,
            0.224,
            "Grid/bin sensitivity preserved the first-refit error ordering; ALE values still depend on binning.",
            14,
            MUTED,
        )
        save(fig)

        # 10. Backup: exact architecture and split controls; no narrative time assigned.
        fig = frame(
            10,
            "Backup · implementation",
            "What is held fixed, and what changes?",
            "Configuration choices are validated by measured errors rather than their labels.",
            appendix=True,
        )
        card(fig, 0.045, 0.22, 0.91, 0.52)
        configs = [
            ["Underfit · either", 2, 1, 1, 650, "0.1"],
            ["Well-fit · FNO", 16, 8, 2, 1400, "0.002"],
            ["Well-fit · linear LNO", 8, 8, 1, 3500, "0"],
            ["Overfit · FNO", 48, 16, 3, 2200, "0"],
            ["Overfit · nonlinear LNO", 48, 16, 3, 3500, "0"],
        ]
        ax = fig.add_axes([0.065, 0.385, 0.87, 0.315])
        ax.axis("off")
        table = ax.table(
            cellText=configs,
            colLabels=["Configuration", "Width", "Modes / poles", "Layers", "Steps", "Decay"],
            loc="center",
            cellLoc="center",
            colWidths=[0.33, 0.10, 0.18, 0.10, 0.14, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.05)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor(BORDER)
            cell.set_facecolor("#E6EEF9" if row == 0 else "white")
            cell.set_text_props(color=INK, weight="bold" if row == 0 else "normal")
        text(fig, 0.07, 0.325, "Same training subset and target scale within each refit; full-batch AdamW.", 15)
        text(
            fig,
            0.07,
            0.27,
            "Well-fit checkpoint minimizes validation MSE. Other models use their final checkpoint.",
            14,
            MUTED,
        )
        text(
            fig,
            0.065,
            0.175,
            "Nonlinear LNO uses cosine learning-rate decay; all inputs and signals are rebuilt after intervention.",
            12,
            MUTED,
        )
        save(fig)

        # 11. Backup: all global effect views at a common, legible scale.
        fig = frame(
            11,
            "Backup · matched effects",
            "PDP and ALE tell a consistent broad story here",
            "Each method is compared with its own simulator counterpart; their finite-grid definitions differ.",
            appendix=True,
        )
        legend(fig, 0.744)
        for i, system in enumerate(["burgers", "oscillator"]):
            for j, method in enumerate(["pdp", "ale"]):
                ax = chart(
                    fig,
                    [0.09 + j * 0.48, 0.435 - i * 0.29, 0.36, 0.19],
                    xlabel=("Viscosity" if system == "burgers" else "Forcing frequency") if i else None,
                    ylabel="Centered RMS effect" if j == 0 else None,
                )
                curve_plot(ax, root, system, method)
                ax.set_title(f"{'Burgers' if system == 'burgers' else 'Oscillator'} · {method.upper()}", fontsize=14)
        save(fig)

        # 12. Backup: individual differences that averaging may conceal.
        fig = frame(
            12,
            "Backup · individual effects",
            "ICE reveals differences that averaging can hide",
            "Same 16 background instances; black is simulator, color is model from the first refit.",
            appendix=True,
        )
        for i, system in enumerate(["burgers", "oscillator"]):
            ref = np.load(root / f"{system}_oracle.npz")
            for j, regime in enumerate(COLORS):
                ax = chart(
                    fig,
                    [0.08 + j * 0.315, 0.455 - i * 0.31, 0.245, 0.215],
                    xlabel=("Viscosity" if system == "burgers" else "Forcing frequency") if i else None,
                    ylabel=("Burgers" if system == "burgers" else "Oscillator") + "\nCentered RMS effect"
                    if j == 0
                    else None,
                )
                e = np.load(root / f"{system}_{regime}_effects.npz")["ice"][0]
                for k in range(0, 128, 8):
                    ax.plot(ref["grid"], ref["ice"][k], color=INK, alpha=0.3, lw=0.8)
                    ax.plot(ref["grid"], e[k], color=COLORS[regime], alpha=0.65, lw=1, ls=STYLES[regime])
                if i == 0:
                    ax.set_title(DISPLAY[regime], color=COLORS[regime], fontsize=16)
        save(fig)
