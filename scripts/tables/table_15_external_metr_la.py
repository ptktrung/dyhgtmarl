"""Table XV: forecasting on METR-LA, a public corpus the authors did not make."""
from _common import TAB, agg, emit, need
import pandas as pd

NICE = {"dyhgt": "DyHGT-MARL mixture density", "ha": "Historical average",
        "ridge": "Ridge autoregression", "gru": "GRU",
        "dcrnn": "DCRNN, reduced", "stgcn": "STGCN, reduced",
        "transformer": "Temporal Transformer"}


def main():
    src = TAB / "exp_i_external_metr_la.csv"
    need(src)
    d = pd.read_csv(src)
    lo, hi = d.missing_rate.min(), d.missing_rate.max()
    clean = d[d.missing_rate == lo]
    sweep = d.pivot_table(index="model", columns="missing_rate", values="crps")

    base = agg(clean, "model", ["mae", "rmse", "mape", "crps", "picp95", "ece"], 3)
    base = base.set_index("model")
    rows = []
    for m in sweep.sort_values(lo).index:
        if m not in base.index:
            continue
        rows.append({
            "Model": NICE.get(m, m),
            "MAE": base.loc[m, "mae"],
            "MAPE (%)": base.loc[m, "mape"],
            "CRPS": base.loc[m, "crps"],
            "PICP95": base.loc[m, "picp95"],
            "ECE": base.loc[m, "ece"],
            f"CRPS at {hi:.0%} loss": f"{sweep.loc[m, hi]:.3f}",
            "Score lost (%)": f"{(sweep.loc[m, hi] / sweep.loc[m, lo] - 1) * 100:+.1f}",
        })

    emit(pd.DataFrame(rows), "table_15_external_metr_la",
         "Table XV. Probabilistic forecasting on METR-LA, 207 loop detectors on the "
         "Los Angeles freeway network, with the protocol of Section V. The last column "
         "is the share of continuous ranked probability score lost between zero and "
         f"{hi * 100:g} percent added observation loss, which is the robustness claim "
         f"of this paper "
         "tested on a corpus the authors did not generate.")


if __name__ == "__main__":
    main()
