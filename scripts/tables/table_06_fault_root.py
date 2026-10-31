"""Table VI: fault detection and root cause attribution results."""
from _common import TAB, agg, emit, need
import pandas as pd


def main():
    need(TAB / "exp_a_fault_detection.csv")
    d = pd.read_csv(TAB / "exp_a_fault_detection.csv")
    cols = ["precision", "recall", "f1", "pr_auc", "roc_auc", "false_alarm_rate"]
    t = agg(d, "model", cols)
    n = d.groupby("model")[["n_false_alarms", "n_missed"]].mean().round(0).astype(int).reset_index()
    t = t.merge(n, on="model")
    if (TAB / "exp_a_detection_delay.csv").exists():
        dd = pd.read_csv(TAB / "exp_a_detection_delay.csv")[
            ["model", "mean_delay_steps", "detected", "undetected"]]
        t = t.merge(dd, on="model", how="left")
    if (TAB / "exp_a_root_cause.csv").exists():
        rc = agg(pd.read_csv(TAB / "exp_a_root_cause.csv"), "model", ["top1", "top3"])
        t = pd.concat([t, rc], axis=0, ignore_index=True)
    emit(t, "table_06_fault_root",
         "Table VI. Fault detection, operational false alerts, detection delay and root cause "
         "attribution on the chronological test block. Values are mean and standard deviation "
         "over model seeds.")


if __name__ == "__main__":
    main()
