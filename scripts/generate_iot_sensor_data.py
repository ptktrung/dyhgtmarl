"""Seeded reproduction of the supplied IoT sensor generator.

Kept faithful to the original, including the row-order cascade mechanism whose
limitation is analysed in the manuscript.
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dyhgt.datagen import generate_iot_source_style
from dyhgt.utils import PROC, ensure_dirs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ensure_dirs()
    df = generate_iot_source_style(a.rows, a.seed)
    df.to_csv(PROC / "regenerated_iot_sensor_data.csv", index=False)
    print("iot rows", df.shape, "anomaly rate", round(float(df.Is_Anomaly.mean()), 5))

if __name__ == "__main__":
    main()
