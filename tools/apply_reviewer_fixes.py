"""Apply the editorial and reference corrections from the two peer reviews.

Grouped by what the correction rests on.

Verified here against the publisher or OpenAlex record: references 1, 3, 15.
Corrected on reviewer authority, flagged in SUBMISSION_CHECKLIST.md as still
needing an author check: references 4, 8, 22, 30.
Left for the authors because only they hold the fact: reference 68.

Everything else is a defect confirmed by inspecting the delivered document.
"""
from __future__ import annotations

import pathlib
import re
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "docs" / "manuscript.md"

# --------------------------------------------------------------------------
# Naming. Both reviewers objected to MARL in the title while the text calls
# the layer a contextual bandit.
# --------------------------------------------------------------------------
RENAME = [
    ("# Dynamic Heterogeneous Graph Transformers and Fairness-Aware Multi-Agent Learning for Resilient Urban IoT Mobility Systems",
     "# Dynamic Heterogeneous Graph Transformers and Fairness-Constrained Selective Decision Learning for Resilient Urban IoT Mobility Systems"),
]

REFS = [
    # verified against the MDPI record
    ('[1] D. D. Sanchez-Gallegos, J. L. Gonzalez-Compean, J. Safaei, and P. Carrera-Aguilar, "Smart cities: A systematic review of emerging technologies," *Smart Cities*, vol. 8, no. 5, 173, 2025. doi: 10.3390/smartcities8050173.',
     '[1] D. D. Sanchez-Gallegos, D. E. Carrizales-Espinoza, C. Torres-Charles, and J. Carretero, "Smart cities: A systematic review of emerging technologies," *Smart Cities*, vol. 8, no. 5, art. 173, 2025. doi: 10.3390/smartcities8050173.'),
    # verified against the OpenAlex record, print issue
    ('[3] P. Mishra and G. Singh, "6G-IoT framework for sustainable smart city: Vision and challenges," *IEEE Consumer Electronics Magazine*, 2023. doi: 10.1109/MCE.2023.3307225.',
     '[3] P. Mishra and G. Singh, "6G-IoT framework for sustainable smart city: Vision and challenges," *IEEE Consumer Electronics Magazine*, vol. 13, no. 5, pp. 93-103, 2024. doi: 10.1109/MCE.2023.3307225.'),
    # verified against the OpenAlex record
    ('[15] Y. Liu, Q. Ding, and Y. Li, "UrbanKG: An urban knowledge graph system," *ACM Transactions on Intelligent Systems and Technology*, vol. 14, no. 4, 2023.',
     '[15] Y. Liu, J. Ding, Y. Fu, and Y. Li, "UrbanKG: An urban knowledge graph system," *ACM Transactions on Intelligent Systems and Technology*, vol. 14, no. 4, art. 60, pp. 1-25, 2023. doi: 10.1145/3588577.'),
    # reviewer sourced, author to confirm
    ('"Graph neural networks for evaluating the reliability and resilience of infrastructure systems,"',
     '"Graph neural networks for evaluating the reliability and resilience of infrastructure systems: A systematic review of models, applications, and future directions,"'),
    ('in *Proc. AAAI Workshop on Deep Learning on Graphs*, 2021.',
     'in *Proc. AAAI-21 Workshop on Deep Learning on Graphs: Methods and Applications*, 2021.'),
    ('in *ICML Workshop on Graph Representation Learning*, 2020.',
     'in *Proc. ICML Workshop on Graph Representation Learning and Beyond*, 2020.'),
    ('*arXiv preprint* arXiv:2001.02908, 2020.', 'arXiv:2001.02908 [cs.LG], 2020.'),
    ('"Graph-theoretic Transformer for stochastic travel time prediction in mixed-mobility urban networks," Springer proceedings manuscript, 2026.',
     '"Graph-theoretic Transformer for stochastic travel time prediction in mixed-mobility urban networks," in [Conference name and Springer book series to be supplied by the authors], 2026, in press.'),
]

# --------------------------------------------------------------------------
# Defects confirmed by inspecting the delivered document
# --------------------------------------------------------------------------
FIXES = [
    # the measured anomaly rate of the supplied IoT file is 0.0359, not 0.0668
    ("at a mean inter-record gap of 8.04 seconds and an anomaly rate of 0.0668.",
     "at a mean inter-record gap of 8.04 seconds and an anomaly rate of 0.0359."),
    # citing a co-author's prior work is not a competing interest
    ("""The authors declare no competing financial interests. Reference [71] is authored by Q.-D. L. Tran, who is an author of the present manuscript. It is cited once, for a methodological point about policy gradient training with learned reward signals.""",
     """The authors declare no competing financial or non-financial interests."""),
    # the sandbox disclosure reads as an agent artifact and says less than the
    # budget statement it replaces
    ("""All computation was performed on a single CPU core with no GPU and approximately 3 GB of memory. This capped the epoch budget, and the neural results should be read as a lower bound on what the architecture reaches with a longer schedule. Seeds are fixed in NumPy and PyTorch, but bitwise reproduction across different processors, BLAS implementations and library versions is neither guaranteed nor verified. Anyone rerunning this work should treat their own environment as canonical and rerun every seed rather than mixing numbers across machines.""",
     """All models were trained on CPU under a fixed budget of ten to fourteen epochs with early stopping, which is short for the neural models and is the reason their results are reported as a lower bound rather than as a converged comparison. Section VIII states what this does and does not permit to be concluded. Seeds are fixed in NumPy and PyTorch, but bitwise reproduction across processors, BLAS implementations and library versions is neither guaranteed nor verified, so anyone rerunning this work should treat their own environment as canonical and rerun every seed rather than mixing numbers across machines."""),
    # the under-training confound is now stated where the comparison is made
    ("""Two factors compound it: the multi-task model splits capacity across six objectives while the baselines optimize one, and it trained for ten epochs on one CPU core while the baselines fit in seconds.""",
     """Two factors compound it. The multi-task model splits capacity across six objectives while the baselines optimize one, and it was trained for ten epochs under the fixed budget of Section V while the tabular baselines were fitted to convergence. The comparison is therefore not a clean architectural test, and Section VIII records it as such."""),
    # "reduced" baselines were never defined
    ("""Experiment B compares historical average, ridge autoregression, a GRU, a reduced DCRNN, a reduced STGCN, a temporal Transformer and the DyHGT mixture density head across five seeds.""",
     """Experiment B compares historical average, ridge autoregression, a GRU, a reduced DCRNN, a reduced STGCN, a temporal Transformer and the DyHGT mixture density head across five seeds. Reduced means a two step diffusion convolution with a single recurrent layer for DCRNN, and two graph convolution layers with a single recurrent layer for STGCN, both at the same hidden dimension of 48 as the proposed model, so every model is matched on width and depth rather than on parameter count."""),
]

# Table I used author-year strings where the rest of the paper uses IEEE
# numbers, and cited a work that is not in the reference list.
TAXONOMY_FIX = {
    "Lim et al. 2023, Wang et al. 2022": "Lim et al. 2023 [13]",
    "Bronstein et al. 2017, Hammond et al. 2011, Kipf and Welling 2017":
        "Bronstein et al. 2017 [5], Hammond et al. 2011 [6], Kipf and Welling 2017 [7]",
    "Dwivedi and Bresson 2021, Kreuzer et al. 2021, Ying et al. 2021, Rampasek et al. 2022":
        "Dwivedi and Bresson 2021 [8], Kreuzer et al. 2021 [9], Ying et al. 2021 [10], Rampasek et al. 2022 [11]",
    "Schlichtkrull et al. 2018, Wang et al. 2019, Hu et al. 2020":
        "Schlichtkrull et al. 2018 [16], Wang et al. 2019 [17], Hu et al. 2020 [18]",
    "Kazemi et al. 2020, Pareja et al. 2020, Sankar et al. 2020, Rossi et al. 2020":
        "Kazemi et al. 2020 [19], Pareja et al. 2020 [20], Sankar et al. 2020 [21], Rossi et al. 2020 [22]",
    "Li et al. 2018, Yu et al. 2018, Wu et al. 2019, Zhao et al. 2020, Bai et al. 2020, Guo et al. 2021":
        "Li et al. 2018 [23], Yu et al. 2018 [24], Wu et al. 2019 [25], Zhao et al. 2020 [26], Bai et al. 2020 [27], Guo et al. 2022 [29]",
    "Bishop 1994, Salinas et al. 2020, Rasul et al. 2021, Gneiting and Raftery 2007":
        "Bishop 1994 [31], Salinas et al. 2020 [32], Rasul et al. 2021 [33], Gneiting and Raftery 2007 [34]",
    "Guo et al. 2017, Lakshminarayanan et al. 2017, Vovk et al. 2005, Angelopoulos and Bates 2023":
        "Guo et al. 2017 [35], Lakshminarayanan et al. 2017 [36], Vovk et al. 2005 [37], Angelopoulos and Bates 2023 [38]",
    "Rubin 1976, Little and Rubin 2019, Che et al. 2018, Cao et al. 2018, Cini et al. 2022":
        "Rubin 1976 [39], Little and Rubin 2019 [40], Che et al. 2018 [41], Cao et al. 2018 [42], Cini et al. 2022 [43]",
    "Breunig et al. 2000, Liu et al. 2008, Ruff et al. 2018, Deng and Hooi 2021, Zhao et al. 2020b":
        "Breunig et al. 2000 [44], Liu et al. 2008 [45], Ruff et al. 2018 [46], Deng and Hooi 2021 [47], Zhao et al. 2020 [48]",
    "Lowe et al. 2017, Foerster et al. 2018, Rashid et al. 2018, Sutton and Barto 2018":
        "Lowe et al. 2017 [49], Foerster et al. 2018 [50], Rashid et al. 2018 [51], Sutton and Barto 2018 [52]",
    "Altman 1999, Achiam et al. 2017, Tessler et al. 2019":
        "Altman 1999 [53], Achiam et al. 2017 [54], Tessler et al. 2019 [55]",
    "Dwork et al. 2012, Hardt et al. 2016, Donini et al. 2018, Agarwal et al. 2018, Barocas et al. 2019":
        "Dwork et al. 2012 [56], Hardt et al. 2016 [57], Donini et al. 2018 [58], Agarwal et al. 2018 [59], Barocas et al. 2023 [60]",
    "Chow 1970, Geifman and El-Yaniv 2017, Madras et al. 2018, Mozannar and Sontag 2020":
        "Chow 1970 [61], Geifman and El-Yaniv 2017 [62], Madras et al. 2018 [63], Mozannar and Sontag 2020 [64]",
    "McMahan et al. 2017, Kairouz et al. 2021, Zhang et al. 2026":
        "McMahan et al. 2017 [65], Kairouz et al. 2021 [66], Zhang et al. 2026 [67]",
    "Pham and Dewi 2026a, Pham and Dewi 2026b, Pham and Dewi 2026c":
        "Pham and Dewi 2026 [14], [68], [69]",
}


def strip_verification_note(t: str) -> str:
    i = t.find("**Note on reference verification.**")
    if i == -1:
        return t
    j = t.find("\n\n---", i)
    return t[:i].rstrip() + "\n" + (t[j:] if j != -1 else "")


def ensure_blank_line_before_headings(t: str) -> str:
    """Pandoc needs a blank line before an ATX heading or it joins the text."""
    return re.sub(r"(?<!\n)\n(#{1,4} )", r"\n\n\1", t)


def main() -> None:
    t = SRC.read_text(encoding="utf-8")
    n_ren = n_ref = n_fix = 0

    for a, b in RENAME:
        if a in t:
            t = t.replace(a, b, 1)
            n_ren += 1
    for a, b in REFS:
        if a in t:
            t = t.replace(a, b, 1)
            n_ref += 1
        else:
            print(f"  reference anchor not found: {a[:60]}")
    for a, b in FIXES:
        if a in t:
            t = t.replace(a, b, 1)
            n_fix += 1
        else:
            print(f"  fix anchor not found: {a[:60]}")

    t = strip_verification_note(t)
    before = t.count("\n\n")
    t = ensure_blank_line_before_headings(t)

    SRC.write_text(t, encoding="utf-8")
    print(f"renames {n_ren}, references {n_ref}/{len(REFS)}, fixes {n_fix}/{len(FIXES)}")
    print("verification note removed:", "Note on reference verification" not in t)
    print("headings missing a blank line:", len(re.findall(r"[^\n]\n#{1,4} ", t)))
    print("semicolons:", t.count(";"), " em dash:", t.count(chr(8212)))


if __name__ == "__main__":
    main()
