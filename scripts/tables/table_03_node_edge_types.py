"""Table III: node types, relation types and their dynamic status."""
from _common import PROC, emit, need
import pandas as pd


def main():
    need(PROC / "node_metadata.csv", PROC / "relation_metadata.csv", PROC / "graph_snapshots.csv")
    nodes = pd.read_csv(PROC / "node_metadata.csv")
    rels = pd.read_csv(PROC / "relation_metadata.csv")
    counts = nodes.NodeType.value_counts()
    edges = pd.read_csv(PROC / "edge_list_static.csv")
    per_rel = edges.RelationID.value_counts().to_dict()
    node_rows = [{"Element": "node type", "Name": k, "Count": str(int(v)),
                  "Temporal status": "static membership", "Provenance": "simulated",
                  "Definition": ""} for k, v in counts.items()]
    rel_rows = [{"Element": "relation type", "Name": r.RelationID,
                 "Count": (str(int(per_rel[r.RelationID]))
                           if r.RelationID in per_rel else "computed per split"),
                 "Temporal status": r.Temporal, "Provenance": r.Provenance,
                 "Definition": r.Definition} for r in rels.itertuples()]
    emit(pd.DataFrame(node_rows + rel_rows), "table_03_node_edge_types",
         "Table III. Node types, relation types and dynamic features of the benchmark graph. Counts are node instances for node types and directed edge instances for relation types. Relations r1, r3, r4 and r7 carry encoder message passing; r2, r5 and r6 are the operational provenance and authority metadata consumed by the decision layer.")


if __name__ == "__main__":
    main()
