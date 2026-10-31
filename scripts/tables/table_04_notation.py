"""Table IV: notation. Curated list, written by script for consistency."""
from _common import emit
import pandas as pd

NOTATION = [
    ("G_t = (V_t, E_t, R, X_t, A_t)", "dynamic heterogeneous graph at step t"),
    ("V_t", "node set at step t, five node types"),
    ("E_t", "edge set at step t, typed by relation"),
    ("R", "set of relation types, seven in the schema, four used for message passing"),
    ("X_t", "node feature matrix at step t"),
    ("A_t", "stack of relation adjacency matrices active at step t"),
    ("N", "number of transit nodes, 28"),
    ("T", "number of time steps, 2880 at 15 minute resolution"),
    ("r", "relation index"),
    ("W_Q^r, W_K^r, W_V^r", "relation specific query, key and value projections"),
    ("alpha_ij^(r,t)", "attention weight from node j to node i under relation r at step t"),
    ("b_ij^(r,t)", "learned edge bias for the edge active in the snapshot at step t"),
    ("q_i", "reliability score of node i, in the unit interval"),
    ("psi(q_i, q_j)", "reliability conditioning term of the attention logit"),
    ("d", "hidden dimension of the encoder"),
    ("phi_k", "k-th eigenvector of the normalized graph Laplacian"),
    ("PE(v_i)", "spectral positional encoding of node i"),
    ("y_it", "true travel time at node i and step t, minutes"),
    ("pi_k, mu_k, sigma_k", "mixture weight, mean and scale of component k"),
    ("K", "number of mixture components, three"),
    ("m_it", "missingness indicator, one if the observation is absent"),
    ("L_total", "multi-task objective"),
    ("lambda_f, lambda_r, lambda_m, lambda_y, lambda_c, lambda_s",
     "weights of the fault, root cause, mask, forecast, calibration and stability terms"),
    ("U_t", "policy utility at step t, in currency units"),
    ("gamma", "discount factor of the decision process"),
    ("C_fair, C_latency, C_risk", "expected fairness, latency and risk costs"),
    ("epsilon_f, epsilon_l, epsilon_r", "budgets for the three constraints"),
    ("lambda_f^dual, lambda_l^dual, lambda_r^dual", "dual variables of the constrained problem"),
    ("a in A", "action from accept, reject, impute, corroborate, defer, isolate, escalate"),
]


def main():
    emit(pd.DataFrame(NOTATION, columns=["Symbol", "Meaning"]),
         "table_04_notation", "Table IV. Notation used throughout the paper.")


if __name__ == "__main__":
    main()
