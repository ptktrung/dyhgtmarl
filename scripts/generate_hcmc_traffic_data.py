"""Seeded reproduction of the supplied 720 by 28 traffic generator.

The supplied hcmc_traffic_data.csv is retained as the primary raw input of the
study. This script exists so the pipeline can be run end to end without it.
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from dyhgt.datagen import generate_traffic_matrix
from dyhgt.utils import PROC, ensure_dirs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=720)
    ap.add_argument("--nodes", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ensure_dirs()
    df, adj, ev = generate_traffic_matrix(a.hours, a.nodes, a.seed)
    df.to_csv(PROC / "regenerated_traffic.csv", index=False)
    pd.DataFrame(adj, index=df.columns, columns=df.columns).to_csv(PROC / "travel_adjacency.csv")
    ev.to_csv(PROC / "regenerated_traffic_events.csv", index=False)
    print("traffic matrix", df.shape, "events", ev.shape)

if __name__ == "__main__":
    main()
