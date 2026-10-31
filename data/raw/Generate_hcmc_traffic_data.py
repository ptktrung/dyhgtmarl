"""Generate hcmc_traffic_data.csv for GTST travel-time forecasting.

This script reconstructs a reproducible, HCMC-inspired synthetic traffic dataset
with 720 hourly observations across 28 graph nodes, matching the schema used by
the travel-time paper: one column per node (Node_0 ... Node_27).

The output is synthetic and must be described as HCMC-inspired simulation data,
not as measured municipal traffic data.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def build_topology(n_nodes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)

    for i in range(n_nodes):
        adjacency[i, (i + 1) % n_nodes] = 1.0
        adjacency[(i + 1) % n_nodes, i] = 1.0

    for i in range(n_nodes):
        for offset in (2, 5, 9):
            j = (i + offset) % n_nodes
            if rng.random() < 0.58:
                weight = rng.uniform(0.35, 0.90)
                adjacency[i, j] = max(adjacency[i, j], weight)
                adjacency[j, i] = max(adjacency[j, i], weight)

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    a_hat = adjacency + np.eye(adjacency.shape[0])
    degree = a_hat.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(degree, 1e-12, None)))
    return d_inv_sqrt @ a_hat @ d_inv_sqrt


def generate_traffic(hours: int, n_nodes: int, seed: int) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    adjacency = build_topology(n_nodes, seed)
    propagation = normalized_adjacency(adjacency)

    index = pd.date_range("2025-01-01 00:00:00", periods=hours, freq="h")
    t = np.arange(hours)
    hour = t % 24
    day = t // 24

    morning_peak = np.exp(-0.5 * ((hour - 8.0) / 1.55) ** 2)
    evening_peak = np.exp(-0.5 * ((hour - 17.3) / 1.85) ** 2)
    daily_cycle = 5.8 * morning_peak + 7.4 * evening_peak
    weekly_cycle = 1.2 * np.sin(2 * np.pi * day / 7.0)
    weekend_relief = np.where((day % 7 >= 5), -2.2, 0.0)

    node_base = rng.uniform(17.0, 27.0, n_nodes)
    node_volatility = rng.uniform(0.7, 2.2, n_nodes)
    node_susceptibility = rng.uniform(0.45, 1.25, n_nodes)
    node_trend = rng.normal(0.0, 0.008, n_nodes)

    traffic = np.zeros((hours, n_nodes), dtype=float)
    latent_state = rng.normal(0.0, 0.4, n_nodes)
    event_log = []

    for step in range(hours):
        peak_intensity = morning_peak[step] + evening_peak[step]
        local_noise = rng.normal(0.0, node_volatility * (1.0 + 0.75 * peak_intensity), n_nodes)
        propagated = propagation @ latent_state
        shared_shock = rng.normal(0.0, 0.65 + 1.35 * peak_intensity)

        spike = np.zeros(n_nodes)
        if peak_intensity > 0.35 and rng.random() < 0.22:
            root = int(rng.integers(0, n_nodes))
            amplitude = rng.exponential(5.0) + 4.0
            spatial_effect = np.linalg.matrix_power(propagation, int(rng.integers(1, 4)))[root]
            spike = amplitude * spatial_effect * node_susceptibility
            event_log.append({
                "timestamp": index[step],
                "event_type": "peak_hour_bottleneck",
                "root_node": f"Node_{root}",
                "amplitude_minutes": round(float(amplitude), 3),
            })

        latent_state = 0.62 * latent_state + 0.28 * propagated + local_noise + spike
        values = (
            node_base
            + daily_cycle[step]
            + weekly_cycle[step]
            + weekend_relief[step]
            + shared_shock
            + latent_state
            + node_trend * step
        )
        traffic[step] = np.clip(values, 4.0, None)

    traffic_df = pd.DataFrame(traffic, index=index, columns=[f"Node_{i}" for i in range(n_nodes)])
    traffic_df.index.name = "Timestamp"
    nodes_df = pd.DataFrame({
        "NodeID": [f"Node_{i}" for i in range(n_nodes)],
        "Zone": ["Core" if i < 14 else "Periphery" for i in range(n_nodes)],
        "BaseTravelTime_min": np.round(node_base, 3),
        "VolatilityScale": np.round(node_volatility, 3),
        "Susceptibility": np.round(node_susceptibility, 3),
    })
    events_df = pd.DataFrame(event_log)
    return traffic_df, adjacency, nodes_df, events_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=720)
    parser.add_argument("--nodes", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--include-timestamp", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traffic_df, adjacency, nodes_df, events_df = generate_traffic(args.hours, args.nodes, args.seed)

    traffic_path = output_dir / "hcmc_traffic_data.csv"
    if args.include_timestamp:
        traffic_df.to_csv(traffic_path, index=True)
    else:
        traffic_df.to_csv(traffic_path, index=False)

    pd.DataFrame(adjacency, index=traffic_df.columns, columns=traffic_df.columns).to_csv(
        output_dir / "hcmc_traffic_adjacency.csv"
    )
    nodes_df.to_csv(output_dir / "hcmc_traffic_nodes.csv", index=False)
    events_df.to_csv(output_dir / "hcmc_traffic_events.csv", index=False)

    metadata = {
        "dataset_name": "HCMC-inspired synthetic travel-time dataset",
        "seed": args.seed,
        "hours": args.hours,
        "nodes": args.nodes,
        "frequency": "hourly",
        "target": "synthetic travel time in minutes",
        "schema": "wide: Node_0 through Node_(N-1)",
        "important_limitations": [
            "Synthetic simulation; not measured HCMC traffic.",
            "Graph topology and event processes are illustrative research assumptions.",
            "Use real or public benchmark data for external validation."
        ]
    }
    with open(output_dir / "hcmc_traffic_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {traffic_path} with shape {traffic_df.shape}")
    print(f"Saved adjacency, node metadata, event log, and metadata JSON in {output_dir}")


if __name__ == "__main__":
    main()
