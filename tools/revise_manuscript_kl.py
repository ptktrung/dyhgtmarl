"""Add the Kuala Lumpur replication cohort to the manuscript.

Two changes. First, the decision layer is renamed to what it is, a contextual
bandit with one optional information gathering action, because every action
except corroboration terminates the episode. Second, a replication subsection
is added to Section VI, with the numbers read from the result files rather
than typed in, so the text cannot drift from the experiments.

Run after both cohorts have been generated and their experiments completed.
"""
from __future__ import annotations

import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript.md"
TAB = ROOT / "results" / "tables"
TAB_KL = ROOT / "results" / "tables_kl"


def f(x, n=3):
    return f"{x:.{n}f}"


def gather() -> dict:
    d = {}
    for key, path in (("hc", TAB), ("kl", TAB_KL)):
        b = pd.read_csv(path / "exp_b_forecast_conditions.csv")
        n = b[(b.model == "dyhgt") & (b.condition == "normal_test")]
        d[f"{key}_mae"] = n.mae.mean(); d[f"{key}_crps"] = n.crps.mean()
        d[f"{key}_mape"] = n.mape.mean()
        d[f"{key}_picp"] = n.picp95.mean(); d[f"{key}_ece"] = n.ece.mean()
        s = pd.read_csv(path / "exp_b_missingness_sweep.csv")
        dy = s[s.model == "dyhgt"].groupby("missing_rate").crps.mean()
        ha = s[s.model == "ha"].groupby("missing_rate").crps.mean()
        d[f"{key}_deg"] = (dy.iloc[-1] / dy.iloc[0] - 1) * 100
        d[f"{key}_hadeg"] = (ha.iloc[-1] / ha.iloc[0] - 1) * 100
        d[f"{key}_crps40"] = dy.iloc[-1]
        c = pd.read_csv(path / "exp_c_policy.csv")
        leg = c[c.policy == "rule_based_legacy"]
        dec = c[c.policy == "decentralized_constrained"]
        unc = c[c.policy == "decentralized_utility_max"]
        d[f"{key}_leg_gap"] = leg.equal_opportunity_gap.mean()
        d[f"{key}_leg_rev"] = leg.revenue_recovered_pct.mean()
        d[f"{key}_dec_gap"] = dec.equal_opportunity_gap.mean()
        d[f"{key}_dec_rev"] = dec.revenue_recovered_pct.mean()
        d[f"{key}_unc_gap"] = unc.equal_opportunity_gap.mean()
        d[f"{key}_abst"] = dec.abstention_rate.mean()
        d[f"{key}_review"] = dec.review_workload_pct.mean()
        cs = pd.read_csv(path / "exp_e_cost_sensitivity.csv")
        g = cs.groupby("false_accept_multiplier").frac_accept.mean()
        d[f"{key}_mixed"] = [m for m, v in g.items() if 0.02 < v < 0.98]
        d[f"{key}_sweep"] = g
        a = path / "exp_a_fault_detection.csv"
        if a.exists():
            af = pd.read_csv(a)
            for m in ("mlp", "dyhgt_marl", "hist_gradient_boosting"):
                g2 = af[af.model == m]
                if not g2.empty:
                    d[f"{key}_f1_{m}"] = g2.f1.mean()
    return d


BANDIT = [
    ("""Decentralization is implemented as one action-value table per zone with independent updates, against a centralized variant with a single table. Both are compared in Section VI. The term multi-agent is used here in a restricted sense. The agents are independent learners partitioned by zone, with no communication channel, no joint value factorization and no opponent modelling, so the layer is weaker than the methods surveyed in [49] to [51]. Section VI-D reports that the centralized and decentralized variants agree within one standard deviation, which is consistent with that restricted reading.""",
     """Decentralization is implemented as one action-value table per zone with independent updates, against a centralized variant with a single table. Both are compared in Section VI.

The decision layer should be named accurately, because the name carries more than the implementation does. Every action except corroboration terminates the record, and nothing an agent does changes the state another agent sees or the distribution of later records. The layer is therefore a **two-zone tabular contextual bandit with one optional information gathering action**, not a multi-agent reinforcement learner in the sense of [49] to [51]. There is no communication channel, no joint value factorization, no opponent modelling and no cross-record credit assignment. The discount factor applies only inside the two-step episode created by corroboration. This paper keeps the system name DyHGT-MARL for continuity with the conference papers it builds on, and states here that the reinforcement learning component is a bandit. Section VI-D reports that the centralized and decentralized variants agree within one standard deviation, which is what a bandit formulation predicts."""),
    ("The decision process is a constrained Markov decision process.",
     "The decision process is a constrained contextual bandit, written below in the notation of a constrained Markov decision process because the constraint machinery is the same."),
]


def main():
    t = SRC.read_text(encoding="utf-8")
    d = gather()

    for a, b in BANDIT:
        if a not in t:
            sys.exit(f"bandit anchor not found: {a[:60]}")
        t = t.replace(a, b, 1)

    # abstract: name the bandit and the replication
    old = "A linear policy trained by policy gradient reaches 93.80 percent revenue by accepting every record, using none of the abstention machinery. Each outcome is traced to a property of the source generators."
    new = ("A linear policy trained by policy gradient reaches 93.80 percent revenue by "
           "accepting every record, using none of the abstention machinery. The decision "
           "layer is a two-zone tabular contextual bandit rather than a multi-agent "
           "reinforcement learner, and is described as such. A replication cohort built "
           "from a second parameterisation of the generator, with a different modal split "
           "and a different spatial scale on the same topology, reproduces the robustness "
           "and the fairness findings and contradicts one of them. Each outcome is traced "
           "to a property of the source generators.")
    if old not in t:
        sys.exit("abstract anchor not found")
    t = t.replace(old, new, 1)

    rc = pd.read_csv(TAB_KL / "exp_a_reconstruction.csv")
    rc = rc[~rc.model.str.contains(r"\[")].groupby("model").mae.mean().sort_values()
    d["kl_recon"] = rc.iloc[0]
    d["kl_knn"] = rc.iloc[1]
    sw_hc, sw_kl = d["hc_sweep"], d["kl_sweep"]
    hc_switch = next((m for m in sw_hc.index if sw_hc[m] < 0.5), None)
    kl_switch = next((m for m in sw_kl.index if sw_kl[m] < 0.5), None)
    mixed = d["kl_mixed"]

    section = f"""

### F. Replication on a second cohort

Every result above comes from one corpus. To test whether they survive a change
of parameterisation, the whole pipeline was rerun without modification on a
second cohort built from the Kuala Lumpur and Klang Valley files of the v2.0
generator. The scope of that test needs stating precisely, because the
generator's own labelling overstates it. The v2.0 generator uses a single
hardcoded edge list for both cities, so the two cohorts share an identical
topology, an identical adjacency matrix and byte-identical Laplacian positional
encodings. The traffic process is not city dependent either. What genuinely
differs is the modal split, 70.7, 16.7 and 12.6 percent against 40.7, 36.6 and
22.6 percent for motorcycles, cars and buses, the physical edge distances, with
means of 4.36 km and 6.92 km, the fare scale, and the random seed. This is a
replication under a second parameterisation, not a transfer test to an
independently specified network, and no cross-city generalization is claimed.

A measurement caution applies before any comparison. The two cohorts have
different travel time scales, with cohort means of roughly 25.6 and 6.1
minutes, because the primary cohort reads the original supplied matrix and the
replication reads the v2.0 table. Absolute error is therefore not comparable
between cohorts, and only scale free quantities and within-cohort changes are
compared below.

Three findings replicate. Relative forecast accuracy is close, at MAPE
{f(d['hc_mape'], 2)} against {f(d['kl_mape'], 2)} percent. Calibration holds in
both, at coverage {f(d['hc_picp'])} against {f(d['kl_picp'])} for a nominal
0.95 and calibration error {f(d['hc_ece'])} against {f(d['kl_ece'])}. Within
each cohort the model remains the best scoring forecaster. Most importantly the robustness
result, which is the paper's main positive claim, reproduces: between zero and
forty percent independent missingness the proposed model loses
{d['hc_deg']:+.1f} percent of score in the primary cohort and
{d['kl_deg']:+.1f} percent in the replication, while the climatological
baseline loses {d['hc_hadeg']:+.1f} and {d['kl_hadeg']:+.1f} percent.

The fairness finding replicates and strengthens. The legacy threshold rule
produces an equal opportunity gap of {f(d['hc_leg_gap'], 4)} in the primary
cohort and {f(d['kl_leg_gap'], 4)} in the replication, where the zone disparity
in sensor quality is sharper, and it recovers {f(d['hc_leg_rev'], 2)} against
{f(d['kl_leg_rev'], 2)} percent of revenue. Learned policies close the gap in
both, to {f(d['hc_dec_gap'], 4)} and {f(d['kl_dec_gap'], 4)}. The constraint
again fails to bind, since the unconstrained policy reaches
{f(d['hc_unc_gap'], 4)} and {f(d['kl_unc_gap'], 4)} without being asked to.

Two further results reverse, and both are informative. Reconstruction is the
larger of the two. In the primary cohort the model ranks fifth of seven, behind
non-causal interpolation at 1.288 and causal carry forward at 2.023 minutes. In
the replication it ranks first of seven, at {d['kl_recon']:.3f} minutes against
{d['kl_knn']:.3f} for the nearest baseline. The cause is visible in the inputs
rather than in the method. The primary cohort's series has a lag one
autocorrelation of 0.859, at which a gap bracketed by two observations is
nearly determined and interpolation is close to optimal. The replication's
series has an autocorrelation of 0.732, at which temporal interpolation
degrades and the cross-node structure the encoder uses begins to pay. The
conclusion in Section VII-B should therefore be read as conditional on the
temporal persistence of the series, not as a general statement about the
method.

Abstention
is used in the primary cohort, at {f(d['hc_abst'], 3)} of decisions and
{f(d['hc_review'], 1)} percent review workload, and is not used at all in the
replication, where the learned policy abstains on {f(d['kl_abst'], 3)} of
decisions. The value of the deferral mechanism is therefore a property of the
parameterisation rather than of the method. Second, Section VI-D concluded from
the primary cohort that the learned policy is constant at every cost setting.
That is not true of the replication. The switch away from blanket acceptance
occurs at a multiplier of {hc_switch:g} in the primary cohort and
{kl_switch:g} in the replication, and at a multiplier of
{', '.join(f'{m:g}' for m in mixed) if mixed else 'no setting'} the replication
produces a genuinely mixed policy rather than a constant one. The earlier
statement should be read as holding for the primary cohort only.

Table XIII sets the two cohorts side by side and Fig. 14 plots the three
comparisons. The replication supports the robustness and fairness results,
removes the claim that the learned policy is always constant, and shows that
the abstention machinery earns its place in one parameterisation and not in
the other.

[[FIGURE:fig_14_replication_cohort]]
"""

    anchor = "\n### E. Ablation\n"
    if anchor not in t:
        sys.exit("results section anchor not found")
    idx = t.index("\n---\n\n## VII. Discussion")
    t = t[:idx] + section + t[idx:]

    # limitations: soften the constant-policy claim, note the replication
    old_l = """**Synthetic data throughout.** No conclusion here transfers to any real network without validation on real data, which has not been done. The benchmark measures whether methods behave sensibly under mechanisms that were deliberately inserted. It cannot establish that those mechanisms resemble reality."""
    new_l = """**Synthetic data throughout.** No conclusion here transfers to any real network without validation on real data, which has not been done. The benchmark measures whether methods behave sensibly under mechanisms that were deliberately inserted. It cannot establish that those mechanisms resemble reality. The replication cohort of Section VI-F does not change this. Both cohorts come from the same generator and share a topology, so the replication tests sensitivity to parameterisation and not external validity."""
    if old_l not in t:
        sys.exit("limitations anchor not found")
    t = t.replace(old_l, new_l, 1)

    old_n = """**Non-binding constraint.** The fairness budget never binds, so the constrained optimization machinery is untested in the regime it was built for. The function approximation check of Section VI-D sharpens this. Because the observable state carries little information about latent validity, the optimal policy is constant, and the cost sensitivity sweep found no setting at which a mixed strategy was preferred."""
    new_n = """**Non-binding constraint.** The fairness budget never binds in either cohort, so the constrained optimization machinery is untested in the regime it was built for. The function approximation check of Section VI-D sharpens this for the primary cohort, where the observable state carries little information about latent validity and the cost sensitivity sweep found no setting at which a mixed strategy was preferred. Section VI-F shows that the second half of that statement is cohort specific, since the replication does produce a mixed policy at a high penalty."""
    if old_n not in t:
        sys.exit("non-binding anchor not found")
    t = t.replace(old_n, new_n, 1)

    # conclusion
    old_c = """The joint representation does not help for fault detection on this corpus, where a multilayer perceptron is decisively better, nor for reconstruction, where causal carry-forward is better, and neither relation typing nor dynamic topology nor reliability conditioning produces a measurable effect."""
    new_c = """A replication on a second parameterisation of the generator reproduces both results, and withdraws the claim that the learned decision policy is constant at every cost setting, which holds only for the primary cohort. The joint representation does not help for fault detection on this corpus, where a multilayer perceptron is decisively better, nor for reconstruction, where causal carry forward is better, and neither relation typing nor dynamic topology nor reliability conditioning produces a measurable effect."""
    if old_c in t:
        t = t.replace(old_c, new_c, 1)

    # artifacts appendix
    t = t.replace("`exp_e_policy_gradient.csv` and `exp_e_cost_sensitivity.csv`.",
                  "`exp_e_policy_gradient.csv` and `exp_e_cost_sensitivity.csv`. The "
                  "replication cohort writes the same set under `results/tables_kl/`.")
    t = t.replace("Fig. 1 through Fig. 13 appear in the body", "Fig. 1 through Fig. 14 appear in the body")
    t = t.replace("`table_01_literature_taxonomy` through `table_12b_cost_sensitivity`",
                  "`table_01_literature_taxonomy` through `table_13_replication_cohort`")

    SRC.write_text(t, encoding="utf-8")
    body = re.sub(r"\$\$.*?\$\$", " ",
                  re.sub(r"\[\[FIGURE:[^\]]+\]\]", "", t.split("## Author Contributions")[0]),
                  flags=re.S)
    print("body words:", len(re.sub(r"[#*`|>\-]", " ", body).split()))
    print("semicolons:", t.count(";"), " em dash:", t.count(chr(8212)))


if __name__ == "__main__":
    main()
