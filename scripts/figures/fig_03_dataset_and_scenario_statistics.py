"""Figure 3: benchmark composition and scenario statistics."""
from _common import PALETTE, PROC, need, save
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    need(PROC / "scenario_summary.json", PROC / "fault_true.npy", PROC / "event_log.csv",
         PROC / "node_mapping.csv")
    s = json.load(open(PROC / "scenario_summary.json"))
    fault = np.load(PROC / "fault_true.npy")
    ev = pd.read_csv(PROC / "event_log.csv")
    mapping = pd.read_csv(PROC / "node_mapping.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    ax = axes[0, 0]
    ft = {k: v for k, v in s["fault_type_counts"].items() if k != "None"}
    ax.bar(list(ft), list(ft.values()), color=PALETTE[:4])
    ax.set_ylabel("node steps"); ax.set_title("Fault type composition")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[0, 1]
    mm = {k: v for k, v in s["missing_mechanism_counts"].items() if k != "observed" and v > 0}
    ax.bar(list(mm), list(mm.values()), color=PALETTE[2:6])
    ax.set_ylabel("node steps"); ax.set_title("Structural missingness by mechanism")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[1, 0]
    per_step = (fault > 0).sum(axis=1)
    ax.plot(np.arange(len(per_step)) / 96.0, per_step, linewidth=0.55, color=PALETTE[0])
    for lo, hi in s["closure_windows"]:
        ax.axvspan(lo / 96, hi / 96, color=PALETTE[1], alpha=0.16)
    ax.axvline(s["distribution_shift_start_step"] / 96, color=PALETTE[3], linestyle="--", linewidth=1)
    ax.set_xlabel("day"); ax.set_ylabel("faulty nodes")
    ax.set_title("Faulty node count over time, closures shaded, shift dashed")

    ax = axes[1, 1]
    zone = mapping.set_index("TrafficNodeID").Zone.values
    rate_core = (fault[:, zone == "Core"] > 0).mean()
    rate_peri = (fault[:, zone == "Periphery"] > 0).mean()
    strat = ev[ev.CentralityStratum != "na"].CentralityStratum.value_counts()
    ax.bar(["Core", "Periphery"], [rate_core, rate_peri], color=[PALETTE[0], PALETTE[1]])
    ax.set_ylabel("fault rate")
    ax.set_title("Zone fault rate. Cascade roots per stratum: "
                 + ", ".join(f"{k}={v}" for k, v in strat.items()), fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_03_dataset_and_scenario_statistics")


if __name__ == "__main__":
    main()
