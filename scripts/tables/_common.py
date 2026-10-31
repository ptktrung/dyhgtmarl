"""Shared table helpers. Tables are written as CSV and as Markdown."""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TAB = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "docs"


def need(*paths):
    missing = [str(p) for p in paths if not pathlib.Path(p).exists()]
    if missing:
        print("SKIPPED: required input not found:\n  " + "\n  ".join(missing))
        sys.exit(0)


def pm(mean, sd, nd=3):
    if sd is None or (isinstance(sd, float) and not np.isfinite(sd)):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} +/- {sd:.{nd}f}"


def agg(df, group, cols, nd=3):
    g = df.groupby(group)[cols].agg(["mean", "std"])
    out = pd.DataFrame(index=g.index)
    for c in cols:
        out[c] = [pm(m, s, nd) for m, s in zip(g[(c, "mean")], g[(c, "std")])]
    return out.reset_index()


def emit(df, name, caption):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    with open(TAB / f"{name}.md", "w", encoding="utf-8") as fh:
        fh.write(f"**{caption}**\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")
    print(f"wrote results/tables/{name}.csv and .md ({len(df)} rows)")
