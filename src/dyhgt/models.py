"""DyHGT-MARL model family and the neural baselines it is compared against.

The encoder operates on the transit projection of the heterogeneous graph plus
the gateway node set, so message passing uses four relation types:
r1 travel connectivity, r3 communication dependency in both directions,
r4 shared-gateway association, and r7 learned functional similarity. The
remaining relations of the schema (r2 sensor to location, r5 data provenance,
r6 operational authority) are one to one or purely administrative and are
consumed by the policy layer rather than by message passing. This is stated
explicitly because it limits how much of the schema the trained model exercises.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Relation aware, reliability conditioned attention
# ---------------------------------------------------------------------------
class RelationAttention(nn.Module):
    """One spatial layer. For each relation r,

        alpha_ij^(r,t) = softmax_j( (Q_i Wq_r)(K_j Wk_r)^T / sqrt(d)
                                    + b_ij^{r,t} + psi(q_i, q_j) )

    b_ij^{r,t} is a learned scaling of the edge weight in the snapshot that is
    active at step t, so a suppressed edge contributes nothing. psi is the
    reliability term, a linear function of the sender and receiver reliability
    scores and their absolute difference. Setting use_reliability to False
    removes psi and gives the ablation variant.
    """

    def __init__(self, d: int, n_rel: int, use_reliability: bool = True):
        super().__init__()
        self.d, self.n_rel, self.use_reliability = d, n_rel, use_reliability
        self.wq = nn.Parameter(torch.randn(n_rel, d, d) * (1.0 / math.sqrt(d)))
        self.wk = nn.Parameter(torch.randn(n_rel, d, d) * (1.0 / math.sqrt(d)))
        self.wv = nn.Parameter(torch.randn(n_rel, d, d) * (1.0 / math.sqrt(d)))
        self.edge_bias = nn.Parameter(torch.zeros(n_rel))
        self.psi = nn.Parameter(torch.zeros(n_rel, 3))
        self.out = nn.Linear(d * n_rel, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # h (B, N, d); adj (n_rel, N, N) weights, zero means no edge; q (B, N)
        outs = []
        qi = q.unsqueeze(2)
        qj = q.unsqueeze(1)
        rel_term = torch.stack([qi.expand(-1, -1, q.shape[1]),
                                qj.expand(-1, q.shape[1], -1),
                                (qi - qj).abs()], dim=-1)
        for r in range(self.n_rel):
            qz = h @ self.wq[r]
            kz = h @ self.wk[r]
            vz = h @ self.wv[r]
            score = qz @ kz.transpose(1, 2) / math.sqrt(self.d)
            score = score + self.edge_bias[r] * adj[r].unsqueeze(0)
            if self.use_reliability:
                score = score + (rel_term * self.psi[r]).sum(-1)
            mask = (adj[r] > 0).unsqueeze(0)
            score = score.masked_fill(~mask, float("-inf"))
            alpha = torch.softmax(score, dim=-1)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            outs.append(alpha @ vz)
        return self.norm(h + self.out(torch.cat(outs, dim=-1)))


class TemporalBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.shape[1]
        causal = torch.triu(torch.ones(w, w, dtype=torch.bool, device=x.device), 1)
        a, _ = self.att(x, x, x, attn_mask=causal, need_weights=False)
        x = self.norm1(x + a)
        return self.norm2(x + self.ff(x))


class DyHGT(nn.Module):
    """Dynamic heterogeneous graph Transformer with multi-task heads."""

    def __init__(self, n_feat: int, n_nodes: int, n_rel: int, pe: np.ndarray,
                 d: int = 48, heads: int = 4, layers: int = 2, k_mix: int = 3,
                 n_types: int = 2, use_graph: bool = True, use_reliability: bool = True,
                 heterogeneous: bool = True):
        super().__init__()
        self.use_graph = use_graph
        self.heterogeneous = heterogeneous
        self.n_rel = n_rel if heterogeneous else 1
        self.k_mix = k_mix
        self.proj = nn.Linear(n_feat, d)
        self.pe = nn.Parameter(torch.tensor(pe, dtype=torch.float32), requires_grad=False)
        self.pe_proj = nn.Linear(pe.shape[1], d)
        self.type_emb = nn.Embedding(n_types, d)
        self.spatial = nn.ModuleList([RelationAttention(d, self.n_rel, use_reliability)
                                      for _ in range(layers)])
        self.temporal = nn.ModuleList([TemporalBlock(d, heads) for _ in range(layers)])
        self.head_fault = nn.Linear(d, 1)
        self.head_root = nn.Linear(d, 1)
        self.head_recon = nn.Linear(d, 1)
        self.head_mdn = nn.Linear(d, 3 * k_mix)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def encode(self, x, adj, q, ntype):
        # x (B, W, N, Fh); adj (n_rel, N, N); q (B, W, N); ntype (N,)
        b, w, n, _ = x.shape
        h = self.proj(x) + self.pe_proj(self.pe).unsqueeze(0).unsqueeze(0) \
            + self.type_emb(ntype).unsqueeze(0).unsqueeze(0)
        for sp, tp in zip(self.spatial, self.temporal):
            if self.use_graph:
                hs = h.reshape(b * w, n, -1)
                hs = sp(hs, adj, q.reshape(b * w, n))
                h = hs.reshape(b, w, n, -1)
            ht = h.permute(0, 2, 1, 3).reshape(b * n, w, -1)
            ht = tp(ht)
            h = ht.reshape(b, n, w, -1).permute(0, 2, 1, 3)
        return h

    def forward(self, x, adj, q, ntype):
        h = self.encode(x, adj, q, ntype)
        last = h[:, -1]
        mdn = self.head_mdn(last)
        k = self.k_mix
        pi = torch.softmax(mdn[..., :k], dim=-1)
        mu = mdn[..., k:2 * k]
        sigma = F.softplus(mdn[..., 2 * k:]) + 1e-3
        return {
            "fault_logit": self.head_fault(h).squeeze(-1),
            "root_score": self.head_root(h).squeeze(-1),
            "recon": self.head_recon(h).squeeze(-1),
            "pi": pi, "mu": mu, "sigma": sigma,
            "temperature": torch.exp(self.log_temp),
        }


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def mdn_nll(pi, mu, sigma, y):
    comp = -0.5 * ((y.unsqueeze(-1) - mu) / sigma) ** 2 \
        - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(torch.log(pi + 1e-9) + comp, dim=-1).mean()


def mdn_moments(pi, mu, sigma):
    mean = (pi * mu).sum(-1)
    var = (pi * (sigma ** 2 + mu ** 2)).sum(-1) - mean ** 2
    return mean, torch.clamp(var, min=1e-6).sqrt()


def brier(logits, target):
    return ((torch.sigmoid(logits) - target) ** 2).mean()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class GRUBaseline(nn.Module):
    """Shared GRU over the multivariate series with a node embedding."""

    def __init__(self, n_nodes, d=48, layers=1):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, d)
        self.gru = nn.GRU(1 + d, d, num_layers=layers, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, x, node_idx):
        # x (B, W, N) -> per node sequence
        b, w, n = x.shape
        seq = x.permute(0, 2, 1).reshape(b * n, w, 1)
        e = self.emb(node_idx).unsqueeze(0).expand(b, -1, -1).reshape(b * n, 1, -1)
        seq = torch.cat([seq, e.expand(-1, w, -1)], dim=-1)
        h, _ = self.gru(seq)
        return self.out(h[:, -1]).reshape(b, n)


class TransformerBaseline(nn.Module):
    """Temporal Transformer with no graph term, the ST-Trans style control."""

    def __init__(self, n_nodes, d=48, heads=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, d)
        self.proj = nn.Linear(1, d)
        self.blocks = nn.ModuleList([TemporalBlock(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, 1)

    def forward(self, x, node_idx):
        b, w, n = x.shape
        seq = self.proj(x.permute(0, 2, 1).reshape(b * n, w, 1))
        seq = seq + self.emb(node_idx).unsqueeze(0).expand(b, -1, -1).reshape(b * n, 1, -1)
        for blk in self.blocks:
            seq = blk(seq)
        return self.out(seq[:, -1]).reshape(b, n)


class STGCNLite(nn.Module):
    """Graph convolution over a fixed normalized adjacency plus a GRU."""

    def __init__(self, n_nodes, adj, d=48):
        super().__init__()
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.gc1 = nn.Linear(1, d)
        self.gc2 = nn.Linear(d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, x, node_idx=None):
        b, w, n = x.shape
        h = torch.relu(self.gc1(x.unsqueeze(-1)))
        h = torch.relu(self.gc2(torch.einsum("ij,bwjd->bwid", self.adj, h)))
        h = h.permute(0, 2, 1, 3).reshape(b * n, w, -1)
        o, _ = self.gru(h)
        return self.out(o[:, -1]).reshape(b, n)


class DCRNNLite(nn.Module):
    """Two step diffusion convolution with a GRU, a reduced DCRNN."""

    def __init__(self, n_nodes, adj, d=48, steps=2):
        super().__init__()
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.steps = steps
        self.proj = nn.Linear(steps + 1, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, 1)

    def forward(self, x, node_idx=None):
        b, w, n = x.shape
        diff = [x]
        cur = x
        for _ in range(self.steps):
            cur = torch.einsum("ij,bwj->bwi", self.adj, cur)
            diff.append(cur)
        h = self.proj(torch.stack(diff, dim=-1))
        h = h.permute(0, 2, 1, 3).reshape(b * n, w, -1)
        o, _ = self.gru(h)
        return self.out(o[:, -1]).reshape(b, n)
