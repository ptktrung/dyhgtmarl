"""Shared plotting helpers. Every figure is saved as PDF and as 300 dpi PNG."""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})
PALETTE = ["#1f5f8b", "#c1553b", "#4b8f6f", "#8a6bbf", "#c9a227", "#6b6b6b", "#2b7a9b"]


def need(*paths):
    """Missing input check. Figures never invent data."""
    missing = [str(p) for p in paths if not pathlib.Path(p).exists()]
    if missing:
        print("SKIPPED: required input not found:\n  " + "\n  ".join(missing))
        sys.exit(0)
    return True


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"saved results/figures/{name}.pdf and .png")
