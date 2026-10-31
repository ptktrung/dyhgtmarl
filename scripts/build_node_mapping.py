"""Write node_mapping.csv, node metadata and relation metadata."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from dyhgt.graph import build_node_mapping, node_metadata, relation_metadata
from dyhgt.utils import PROC, ensure_dirs

def main():
    ensure_dirs()
    build_node_mapping().to_csv(PROC / "node_mapping.csv", index=False)
    node_metadata().to_csv(PROC / "node_metadata.csv", index=False)
    relation_metadata().to_csv(PROC / "relation_metadata.csv", index=False)
    print("wrote node_mapping.csv, node_metadata.csv, relation_metadata.csv")

if __name__ == "__main__":
    main()
