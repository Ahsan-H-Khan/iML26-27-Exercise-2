"""Shared, accessible styling for the study's figures and presentation."""

INK = "#14263D"
MUTED = "#52657A"
BLUE = "#2463B4"
CORAL = "#C6513E"
SLATE = "#687D94"
TEAL = "#167D8D"
BG = "#F3F6FA"
BORDER = "#DCE4EE"
COLORS = {"underfit": SLATE, "well_fit": BLUE, "overfit": CORAL}
STYLES = {"underfit": ":", "well_fit": "-", "overfit": "--"}
DISPLAY = {"underfit": "Underfit", "well_fit": "Well-fit", "overfit": "Overfit"}
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": BORDER,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": BORDER,
    "grid.alpha": 0.6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
}
