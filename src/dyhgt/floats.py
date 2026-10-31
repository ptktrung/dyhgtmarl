"""Renumber figures and tables into order of first citation.

IEEE and most journals require floats to be numbered in the order the text
first refers to them. The manuscript was written section by section, so the
numbers had drifted out of that order. Rather than hand-editing, the mapping
is derived from the text itself at build time, which means it stays correct
when sections move.

A lettered companion such as Table VIII b stays attached to its base table and
is not given an independent position.
"""
from __future__ import annotations

import re

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]
R2I = {r: i + 1 for i, r in enumerate(ROMAN)}

FIG_CITE = re.compile(r"\bFig\. (\d{1,2})\b")
FIG_MARK = re.compile(r"\[\[FIGURE:(fig_(\d{2})_[a-z0-9_]+)\]\]")
TAB_CITE = re.compile(r"\bTable ([IVX]{1,6})(\s?b)?\b")


def _first_order(keys):
    seen, order = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)
    return order


def renumber(text: str):
    """Return rewritten text, a figure map and a table map, both old to new."""
    body = text.split("## References")[0]

    # figures: a marker counts as a citation at its own position, because a
    # figure placed without a prior mention is still first seen there
    events = []
    for m in re.finditer(r"\bFig\. (\d{1,2})\b|\[\[FIGURE:fig_(\d{2})_", body):
        events.append(int(m.group(1) or m.group(2)))
    fig_order = _first_order(events)
    fig_map = {old: i + 1 for i, old in enumerate(fig_order)}

    tab_order = _first_order(R2I[m.group(1)] for m in TAB_CITE.finditer(body)
                             if m.group(1) in R2I)
    tab_map = {old: i + 1 for i, old in enumerate(tab_order)}

    def fig_cite_sub(m):
        n = int(m.group(1))
        return f"Fig. {fig_map.get(n, n)}"

    def tab_cite_sub(m):
        r = m.group(1)
        if r not in R2I:
            return m.group(0)
        new = tab_map.get(R2I[r], R2I[r])
        return f"Table {ROMAN[new - 1]}{m.group(2) or ''}"

    head, sep, tail = text.partition("## References")
    head = FIG_CITE.sub(fig_cite_sub, head)
    head = TAB_CITE.sub(tab_cite_sub, head)
    return head + sep + tail, fig_map, tab_map


def figure_number(filename: str, fig_map: dict) -> int:
    old = int(filename.split("_")[1])
    return fig_map.get(old, old)


def table_label(old_label: str, tab_map: dict) -> str:
    m = re.fullmatch(r"Table ([IVX]{1,6})(\s?b)?", old_label.strip())
    if not m or m.group(1) not in R2I:
        return old_label
    new = tab_map.get(R2I[m.group(1)], R2I[m.group(1)])
    return f"Table {ROMAN[new - 1]}{m.group(2) or ''}"
