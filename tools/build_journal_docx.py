"""Build the journal manuscript as a Word document.

Three things distinguish this from a plain conversion.

Equations. Display and inline mathematics are converted to Office Math Markup,
which is what Word calls a Standard Equation. They are live equation objects
that open in the equation editor, not pictures and not MathType fields.

Citations. Every bracketed citation becomes a Word CITATION field, and the
source list is written into the document as a custom XML part so that the
entries appear under References, Manage Sources. The bibliography is a
BIBLIOGRAPHY field inside a Bibliography content control, set to the IEEE
style, with the formatted entries cached so the list reads correctly before
any field update.

Layout. A reference document supplies journal manuscript styling, and the
numbered tables are inserted at the point of first citation.

Run: python scripts/build_journal_docx.py
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript.md"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
OUT = ROOT / "DyHGT_MARL_Resilient_Urban_IoT_Mobility_Systems_Full_Paper.docx"
BUILD = ROOT / "build"
sys.path.insert(0, str(ROOT / "src"))

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_B = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"
NS_DS = "http://schemas.openxmlformats.org/officeDocument/2006/customXml"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

FIG_CAPTIONS = {
    "fig_01_dyhgt_architecture": "DyHGT-MARL pipeline: data harmonization, dynamic heterogeneous encoder, multi-task heads and constrained decision layer.",
    "fig_02_heterogeneous_graph_schema": "Heterogeneous schema: five node types and seven relation types.",
    "fig_03_dataset_and_scenario_statistics": "Benchmark composition and scenario statistics.",
    "fig_04_dynamic_cascade_snapshots": "Snapshots of one cascade propagating over the travel graph.",
    "fig_05_fault_detection_performance": "Fault detection performance across models and the operating trade off.",
    "fig_06_reconstruction_example": "Reconstruction error by method, and one reconstructed series.",
    "fig_07_probabilistic_forecast": "Probabilistic travel time forecast with 95 percent bounds.",
    "fig_08_uncertainty_calibration": "Reliability diagram and probability integral transform histogram.",
    "fig_09_dynamic_graph_robustness": "Robustness across stress conditions and the independent missingness sweep.",
    "fig_10_fairness_utility_pareto": "Fairness and utility trade off for the constrained policy.",
    "fig_11_system_cost_latency": "Decision cost, review workload and training cost.",
    "fig_12_ablation_matrix": "Ablation of model side and policy side design choices.",
    "fig_13_function_approximation_policy": "Function approximation policy against the tabular policy, and the cost sensitivity sweep.",
    "fig_14_replication_cohort": "Replication cohort against the primary cohort: robustness to observation loss, the zone gap under each policy, and the cost sensitivity sweep.",
    "fig_15_stress_regime": "Revised stress regime against the primary benchmark: node correlation before and after decorrelation, fault detection under relational faults, and the fairness and utility trade off once latent validity depends on zone.",
}

# Numbered tables inserted at the paragraph that first cites them. Wide result
# files are narrowed to the columns a reader needs, and the full version stays
# in the CSV.
TABLE_INSERTS = [
    ("Table I",   "table_01_literature_taxonomy",  None,
     "Taxonomy of related work by the capability each line of research supplies and the capability it leaves open."),
    ("Table II",  "table_02_source_datasets",      None,
     "Source datasets and the integrated benchmark, with the limitation that matters most for each."),
    ("Table III", "table_03_node_edge_types",      None,
     "Node types, relation types and dynamic features of the benchmark graph."),
    ("Table IV",  "table_04_notation",             None,
     "Notation used throughout the paper."),
    ("Table V",   "table_05_scenarios_splits",     None,
     "Scenario definitions, generator parameters and chronological splits."),
    ("Table VIII", "table_08_forecast_calibration",
     ["model", "condition", "mae", "rmse", "crps", "picp95", "ece"],
     "Travel time forecasting and uncertainty calibration by evaluation condition, mean and standard deviation over five seeds."),
    ("Table X",   "table_10_system_cost",          None,
     "Computational cost of each component. Training times were measured on a single CPU core and are not a hardware benchmark."),
    ("Table XII", "table_12_function_approximation",
     ["policy", "revenue_recovered_pct", "decision_error_rate", "abstention_rate",
      "review_workload_pct", "equal_opportunity_gap", "n_seeds"],
     "Tabular action value policies against linear policies trained by policy gradient, on the same constrained process and seeds."),
    ("Table XIII", "table_13_replication_cohort", None,
     "Primary cohort against the Kuala Lumpur replication cohort. Both cohorts share the network topology and the scenario layer, and differ in modal split, physical edge distance, fare scale and random seed."),
    ("Table XV", "table_15_external_metr_la", None,
     "Probabilistic forecasting on METR-LA, 207 loop detectors on the Los Angeles freeway network, with the protocol of Section V. The last column is the share of continuous ranked probability score lost between zero and the highest added observation loss, which is the robustness claim of this paper tested on a corpus the authors did not generate."),
    ("Table XIV", "table_14_stress_regime", None,
     "The primary benchmark against the revised stress regime, in which faults show as neighbourhood disagreement rather than telemetry clamping, the node series are decorrelated, and latent validity and false accept cost differ by zone. Topology, splits and evaluation code are unchanged."),
]

# Captions for the tables already written into the manuscript text.
INLINE_TABLE_CAPTIONS = {
    "| Traffic node range |": None,          # mapping assumption, left unnumbered
    "| Model | Precision |": ("Table VI", "Fault detection on the chronological test block, mean over three seeds."),
    "| Method | MAE (min) |": ("Table VII", "Reconstruction of masked travel time observations on the shared artificial mask."),
    "| Missing rate |": ("Table VIII b", "Continuous ranked probability score under the independent missingness sweep."),
    "| Policy | Revenue (%) |": ("Table IX", "Decision policy outcomes on the held out transaction block."),
    "| Variant | F1 |": ("Table XI", "Ablation of model side design choices, mean over two seeds."),
}


# ---------------------------------------------------------------------------
# Markdown preparation
# ---------------------------------------------------------------------------
def load_table_md(stem: str, columns: list[str] | None) -> str:
    import pandas as pd
    df = pd.read_csv(TAB / f"{stem}.csv")
    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    df = df.fillna("")
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    header = "| " + " | ".join(str(c).replace("_", " ") for c in df.columns) + " |"
    sep = "|" + "|".join([" --- "] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(r) + " |" for r in df.astype(str).values]
    return "\n".join([header, sep] + rows)


def prepare_markdown(text: str) -> str:
    from dyhgt.floats import renumber, figure_number, table_label
    text, FIG_MAP, TAB_MAP = renumber(text)
    moved_f = {o: n for o, n in FIG_MAP.items() if o != n}
    moved_t = {o: n for o, n in TAB_MAP.items() if o != n}
    print(f"  floats renumbered into citation order: "
          f"{len(moved_f)} figures, {len(moved_t)} tables")

    # figures
    def fig_sub(m):
        name = m.group(1)
        num = figure_number(name, FIG_MAP)
        png = FIG / f"{name}.png"
        if not png.exists():
            return ""
        cap = FIG_CAPTIONS.get(name, name)
        return f"![Fig. {num}. {cap}]({png.as_posix()})"
    text = re.sub(r"\[\[FIGURE:([a-z0-9_]+)\]\]", fig_sub, text)

    # captions on tables already present in the text
    out_lines, lines = [], text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            matched = None
            for key, val in INLINE_TABLE_CAPTIONS.items():
                if line.startswith(key):
                    matched = val
                    break
            if matched:
                out_lines.append(f"**{table_label(matched[0], TAB_MAP)}.** {matched[1]}")
                out_lines.append("")
        out_lines.append(line)
        i += 1
    text = "\n".join(out_lines)

    # numbered tables at first citation
    for label0, stem, cols, caption in TABLE_INSERTS:
        label = table_label(label0, TAB_MAP)
        pat = re.compile(r"(^.*\b" + re.escape(label) + r"\b.*$)", re.M)
        m = pat.search(text)
        if not m:
            print(f"  note: no citation found for {label}, table appended to Appendix B")
            continue
        block = f"\n\n**{label}.** {caption}\n\n{load_table_md(stem, cols)}\n"
        # anchor to the end of the paragraph, not the end of the wrapped line,
        # so a table is never dropped into the middle of a sentence
        para_end = text.find("\n\n", m.end())
        end = len(text) if para_end == -1 else para_end
        text = text[:end] + block + text[end:]

    # the lettered companion follows its base table
    base12 = table_label("Table XII", TAB_MAP)
    if f"{base12}." in text:
        blk = f"\n\n**{base12} b.** Action mix chosen by the function approximation policy as the penalty for accepting an invalid record rises.\n\n{load_table_md('table_12b_cost_sensitivity', None)}\n"
        cols12 = next(c for lab, stem, c, _ in TABLE_INSERTS
                      if stem == "table_12_function_approximation")
        anchor = text.index(load_table_md("table_12_function_approximation", cols12))
        anchor_end = text.index("\n\n", anchor + 10)
        text = text[:anchor_end] + blk + text[anchor_end:]

    return text


# ---------------------------------------------------------------------------
# Reference parsing into Word source records
# ---------------------------------------------------------------------------
def split_authors(chunk: str) -> list[str]:
    chunk = chunk.replace(" and ", ", ").replace(" et al.", "")
    parts = [p.strip() for p in chunk.split(",") if p.strip()]
    out, buf = [], ""
    for p in parts:
        if re.fullmatch(r"(?:[A-Z]\.-?){1,4}", p.replace(" ", "")):
            buf = p
        else:
            out.append(f"{buf} {p}".strip() if buf else p)
            buf = ""
    return [a for a in out if a and not a.lower().startswith("et al")]


def parse_references(refs_text: str) -> list[dict]:
    srcs = []
    for m in re.finditer(r"^\[(\d+)\]\s+(.+?)(?=\n\n|\Z)", refs_text, re.S | re.M):
        num, body = int(m.group(1)), " ".join(m.group(2).split())
        rec = {"num": num, "raw": body}
        tm = re.search(r'"([^"]+),?"', body)
        if tm:
            rec["title"] = tm.group(1).rstrip(",")
            rec["authors"] = split_authors(body[:tm.start()].rstrip(", "))
            title_end = tm.end()
        else:
            # a book states its title in italics rather than in quotation marks
            bm = re.search(r"\*([^*]+)\*", body)
            if bm:
                rec["title"] = bm.group(1).strip().rstrip(".,")
                rec["authors"] = split_authors(body[:bm.start()].rstrip(", "))
                title_end = bm.end()
            else:
                rec["title"] = body[:80]
                rec["authors"] = []
                title_end = len(body)
        rec["_title_end"] = title_end
        ym = re.findall(r"\b(19|20)\d{2}\b", body)
        rec["year"] = re.findall(r"\b((?:19|20)\d{2})\b", body)[-1] if ym else ""
        rest = body[rec["_title_end"]:]
        cm = re.search(r"\*([^*]+)\*", rest)
        rec["container"] = cm.group(1).strip().rstrip(",") if cm else ""
        vm = re.search(r"vol\.\s*([0-9]+)", rest); rec["volume"] = vm.group(1) if vm else ""
        nm = re.search(r"no\.\s*([0-9]+)", rest); rec["issue"] = nm.group(1) if nm else ""
        pm = re.search(r"pp?\.\s*([0-9]+\s*-\s*[0-9]+|[0-9]+)", rest)
        rec["pages"] = pm.group(1).replace(" ", "") if pm else ""
        dm = re.search(r"doi:\s*([^\s,]+?)\.?$", rest, re.I)
        rec["doi"] = dm.group(1).rstrip(".") if dm else ""
        low = rest.lower()
        is_book = not re.search(r'"', body) and re.search(r"\*[^*]+\*", body)
        if is_book and ("press" in low or "wiley" in low or "springer" in low
                        or "crc" in low or "hall" in low):
            rec["type"] = "Book"
            rec["container"] = ""
        elif "in *proc" in low or rest.strip().startswith("in "):
            rec["type"] = "ConferenceProceedings"
        elif rec["container"] and (rec["volume"] or "journal" in low or
                                   "transactions" in low or "access" in low):
            rec["type"] = "JournalArticle"
        elif "press" in low or "wiley" in low or "springer" in low and not rec["volume"]:
            rec["type"] = "Book"
        elif rec["container"]:
            rec["type"] = "JournalArticle"
        else:
            rec["type"] = "Report"
        rec["tag"] = f"Ref{num:03d}"
        srcs.append(rec)
    return srcs


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def sources_xml(srcs: list[dict]) -> str:
    parts = [f'<b:Sources xmlns:b="{NS_B}" SelectedStyle="\\IEEE.XSL" '
             f'StyleName="IEEE" Version="6">']
    for s in srcs:
        a = "".join(
            f"<b:Person><b:Last>{esc(n.split()[-1])}</b:Last>"
            f"<b:First>{esc(' '.join(n.split()[:-1]) or n.split()[-1])}</b:First></b:Person>"
            for n in s["authors"][:12]) or \
            "<b:Person><b:Last>Anon</b:Last></b:Person>"
        fields = [
            f"<b:SourceType>{s['type']}</b:SourceType>",
            f"<b:Guid>{{{uuid.uuid4()}}}</b:Guid>",
            f"<b:Tag>{s['tag']}</b:Tag>",
            f"<b:Title>{esc(s['title'])}</b:Title>",
            f"<b:Year>{esc(s['year'])}</b:Year>",
            f"<b:Author><b:Author><b:NameList>{a}</b:NameList></b:Author></b:Author>",
        ]
        if s["container"]:
            tag = {"JournalArticle": "JournalName", "ConferenceProceedings": "ConferenceName",
                   "Book": "Title", "Report": "Publisher"}.get(s["type"], "JournalName")
            if tag != "Title":
                fields.append(f"<b:{tag}>{esc(s['container'])}</b:{tag}>")
        for key, tag in (("volume", "Volume"), ("issue", "Issue"), ("pages", "Pages"),
                         ("doi", "DOI")):
            if s[key]:
                fields.append(f"<b:{tag}>{esc(s[key])}</b:{tag}>")
        fields.append("<b:RefOrder>%d</b:RefOrder>" % s["num"])
        parts.append("<b:Source>" + "".join(fields) + "</b:Source>")
    parts.append("</b:Sources>")
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n' + "".join(parts)


# ---------------------------------------------------------------------------
# document.xml post-processing
# ---------------------------------------------------------------------------
def citation_runs(tag: str, shown: str) -> str:
    return (
        f'<w:r xmlns:w="{NS_W}"><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r xmlns:w="{NS_W}"><w:instrText xml:space="preserve"> CITATION {tag} \\l 1033 </w:instrText></w:r>'
        f'<w:r xmlns:w="{NS_W}"><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r xmlns:w="{NS_W}"><w:rPr><w:noProof/></w:rPr><w:t xml:space="preserve">{esc(shown)}</w:t></w:r>'
        f'<w:r xmlns:w="{NS_W}"><w:fldChar w:fldCharType="end"/></w:r>')


def inject_citations(doc_xml: bytes, srcs: list[dict]) -> tuple[bytes, int]:
    from lxml import etree
    root = etree.fromstring(doc_xml)
    W = "{%s}" % NS_W
    by_num = {s["num"]: s for s in srcs}

    # find the paragraph that holds the References heading, stop converting there
    stop = None
    for p in root.iter(W + "p"):
        txt = "".join(t.text or "" for t in p.iter(W + "t"))
        if re.fullmatch(r"\s*References\s*", txt):
            stop = p
            break

    reached, n = [False], 0
    pat = re.compile(r"\[(\d{1,2})\]")

    for p in list(root.iter(W + "p")):
        if p is stop:
            reached[0] = True
        if reached[0]:
            continue
        for r in list(p.iter(W + "r")):
            ts = r.findall(W + "t")
            if len(ts) != 1 or not ts[0].text:
                continue
            text = ts[0].text
            if not pat.search(text):
                continue
            pieces, last = [], 0
            for m in pat.finditer(text):
                num = int(m.group(1))
                if num not in by_num:
                    continue
                pieces.append(("t", text[last:m.start()]))
                pieces.append(("c", by_num[num]))
                last = m.end()
            if not pieces:
                continue
            pieces.append(("t", text[last:]))

            rpr = r.find(W + "rPr")
            rpr_xml = etree.tostring(rpr, encoding="unicode") if rpr is not None else ""
            parent = r.getparent()
            idx = list(parent).index(r)
            new_nodes = []
            for kind, val in pieces:
                if kind == "t":
                    if not val:
                        continue
                    frag = (f'<w:r xmlns:w="{NS_W}">{rpr_xml}'
                            f'<w:t xml:space="preserve">{esc(val)}</w:t></w:r>')
                    new_nodes.append(etree.fromstring(frag))
                else:
                    for node in etree.fromstring(
                            f'<root xmlns:w="{NS_W}">'
                            + citation_runs(val["tag"], f"[{val['num']}]") + "</root>"):
                        new_nodes.append(node)
                    n += 1
            parent.remove(r)
            for off, node in enumerate(new_nodes):
                parent.insert(idx + off, node)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8",
                          standalone=True), n


def number_equations(doc_xml: bytes) -> tuple[bytes, int]:
    """Number displayed equations consecutively, flush right, as IEEE requires.

    Pandoc centers a display equation with an oMathPara wrapper, which leaves no
    room for a number. Each such paragraph is rewritten with a centre tab stop
    and a right tab stop, so the equation stays centred and the number sits
    against the right margin, which is how the same layout is produced by hand
    in Word. Tab positions come from the section's own page size, so the number
    lands correctly whether the document is Letter or A4.
    """
    x = doc_xml.decode("utf-8")

    width = 12240
    m = re.search(r'<w:pgSz[^>]*w:w="(\d+)"', x)
    if m:
        width = int(m.group(1))
    left = right = 1440
    m = re.search(r"<w:pgMar[^>]*>", x)
    if m:
        ml = re.search(r'w:left="(\d+)"', m.group(0))
        mr = re.search(r'w:right="(\d+)"', m.group(0))
        left = int(ml.group(1)) if ml else left
        right = int(mr.group(1)) if mr else right
    text_w = max(width - left - right, 4000)
    tabs = (f'<w:tabs><w:tab w:val="center" w:pos="{text_w // 2}"/>'
            f'<w:tab w:val="right" w:pos="{text_w}"/></w:tabs>')

    para_re = re.compile(r"<w:p\b[^>]*>.*?</w:p>", re.S)
    math_re = re.compile(r"<m:oMathPara>\s*(?:<m:oMathParaPr>.*?</m:oMathParaPr>)?"
                         r"\s*(<m:oMath>.*?</m:oMath>)\s*</m:oMathPara>", re.S)
    count = 0

    def do_para(pm: re.Match) -> str:
        nonlocal count
        para = pm.group(0)
        mm = math_re.search(para)
        if not mm:
            return para
        count += 1
        math = mm.group(1)
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", para, re.S)
        if ppr:
            inner = ppr.group(0)
            if "<w:tabs>" not in inner:
                inner = inner.replace("</w:pPr>", tabs + "</w:pPr>")
            # a centred paragraph would fight the tab stops
            inner = re.sub(r'<w:jc w:val="[^"]*"/>', "", inner)
        else:
            inner = "<w:pPr>" + tabs + "</w:pPr>"
        body = ("<w:r><w:tab/></w:r>" + math +
                "<w:r><w:tab/></w:r>"
                f"<w:r><w:t xml:space=\"preserve\">({count})</w:t></w:r>")
        return "<w:p>" + inner + body + "</w:p>"

    out = para_re.sub(do_para, x)
    return out.encode("utf-8"), count


def wrap_bibliography(doc_xml: bytes) -> bytes:
    """Put the formatted reference list inside a Bibliography content control."""
    from lxml import etree
    root = etree.fromstring(doc_xml)
    W = "{%s}" % NS_W
    body = root.find(W + "body")
    kids = list(body)

    start = None
    for i, el in enumerate(kids):
        if el.tag != W + "p":
            continue
        txt = "".join(t.text or "" for t in el.iter(W + "t"))
        if re.fullmatch(r"\s*References\s*", txt):
            start = i
            break
    if start is None:
        return doc_xml

    end = len(kids)
    for j in range(start + 1, len(kids)):
        el = kids[j]
        if el.tag == W + "p":
            txt = "".join(t.text or "" for t in el.iter(W + "t")).strip()
            if txt.startswith("Appendix A"):
                end = j
                break

    entries = kids[start + 1:end]
    sdt = etree.fromstring(
        f'<w:sdt xmlns:w="{NS_W}"><w:sdtPr><w:docPartObj>'
        f'<w:docPartGallery w:val="Bibliographies"/><w:docPartUnique/>'
        f'</w:docPartObj></w:sdtPr><w:sdtContent/></w:sdt>')
    content = sdt.find(W + "sdtContent")

    first = etree.fromstring(
        f'<w:p xmlns:w="{NS_W}">'
        f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> BIBLIOGRAPHY \\l 1033 </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r></w:p>')
    content.append(first)
    for el in entries:
        body.remove(el)
        content.append(el)
    content.append(etree.fromstring(
        f'<w:p xmlns:w="{NS_W}"><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'))

    body.insert(start + 1, sdt)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def add_bibliography_part(docx_path: pathlib.Path, sources: str) -> None:
    """Add the custom XML part that populates References, Manage Sources."""
    from lxml import etree
    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(docx_path) as z:
        z.extractall(tmp)

    (tmp / "customXml" / "_rels").mkdir(parents=True, exist_ok=True)
    (tmp / "customXml" / "item1.xml").write_text(sources, encoding="utf-8")
    item_id = "{%s}" % uuid.uuid4()
    (tmp / "customXml" / "itemProps1.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<ds:datastoreItem xmlns:ds="{NS_DS}" ds:itemID="{item_id}">'
        f'<ds:schemaRefs><ds:schemaRef ds:uri="{NS_B}"/></ds:schemaRefs>'
        '</ds:datastoreItem>', encoding="utf-8")
    (tmp / "customXml" / "_rels" / "item1.xml.rels").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Relationships xmlns="{NS_R}/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{NS_R}/customXmlProps" '
        'Target="itemProps1.xml"/></Relationships>'.replace(
            f"{NS_R}/package", "http://schemas.openxmlformats.org/package"),
        encoding="utf-8")

    rels_path = tmp / "word" / "_rels" / "document.xml.rels"
    rels = etree.parse(str(rels_path))
    rroot = rels.getroot()
    used = {r.get("Id") for r in rroot}
    rid = next(f"rId{i}" for i in range(900, 999) if f"rId{i}" not in used)
    etree.SubElement(rroot, "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship",
                     Id=rid, Type=f"{NS_R}/customXml", Target="../customXml/item1.xml")
    rels.write(str(rels_path), xml_declaration=True, encoding="UTF-8", standalone=True)

    ct_path = tmp / "[Content_Types].xml"
    ct = etree.parse(str(ct_path))
    croot = ct.getroot()
    CT = "{http://schemas.openxmlformats.org/package/2006/content-types}"
    have = {o.get("PartName") for o in croot.iter(CT + "Override")}
    if "/customXml/itemProps1.xml" not in have:
        etree.SubElement(croot, CT + "Override", PartName="/customXml/itemProps1.xml",
                         ContentType="application/vnd.openxmlformats-officedocument.customXmlProperties+xml")
    ct.write(str(ct_path), xml_declaration=True, encoding="UTF-8", standalone=True)

    docx_path.unlink()
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        ctf = tmp / "[Content_Types].xml"
        z.write(ctf, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ctf:
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)


# OMML and settings child order required by the schema. Pandoc emits a few of
# these out of sequence. Word tolerates it, but a strict consumer may not.
MATH_DPR_ORDER = ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"]
# m:scr and m:sty are what Word itself writes inside m:rPr, but the strict ISO
# schema omits them, so that element is left exactly as produced.
PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr",
    "widowControl", "numPr", "suppressLineNumbers", "pBdr", "shd", "tabs",
    "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct", "topLinePunct",
    "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing",
    "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
    "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId",
    "cnfStyle", "rPr", "sectPr", "pPrChange"]
SETTINGS_ORDER = ["writeProtection", "view", "zoom", "removePersonalInformation",
    "removeDateAndTime", "doNotDisplayPageBoundaries", "displayBackgroundShape",
    "printPostScriptOverText", "printFractionalCharacterWidth", "printFormsData",
    "embedTrueTypeFonts", "embedSystemFonts", "saveSubsetFonts", "saveFormsData",
    "mirrorMargins", "alignBordersAndEdges", "bordersDoNotSurroundHeader",
    "bordersDoNotSurroundFooter", "gutterAtTop", "hideSpellingErrors",
    "hideGrammaticalErrors", "activeWritingStyle", "proofState", "formsDesign",
    "attachedTemplate", "linkStyles", "stylePaneFormatFilter", "stylePaneSortMethod",
    "documentType", "mailMerge", "revisionView", "trackRevisions", "doNotTrackMoves",
    "doNotTrackFormatting", "documentProtection", "autoFormatOverride",
    "styleLockTheme", "styleLockQFSet", "defaultTabStop", "autoHyphenation",
    "consecutiveHyphenLimit", "hyphenationZone", "doNotHyphenateCaps", "showEnvelope",
    "summaryLength", "clickAndTypeStyle", "defaultTableStyle", "evenAndOddHeaders",
    "bookFoldRevPrinting", "bookFoldPrinting", "bookFoldPrintingSheets",
    "drawingGridHorizontalSpacing", "drawingGridVerticalSpacing",
    "displayHorizontalDrawingGridEvery", "displayVerticalDrawingGridEvery",
    "doNotUseMarginsForDrawingGridOrigin", "drawingGridHorizontalOrigin",
    "drawingGridVerticalOrigin", "doNotShadeFormData", "noPunctuationKerning",
    "characterSpacingControl", "printTwoOnOne", "strictFirstAndLastChars",
    "noLineBreaksAfter", "noLineBreaksBefore", "savePreviewPicture",
    "doNotValidateAgainstSchema", "saveInvalidXml", "ignoreMixedContent",
    "alwaysShowPlaceholderText", "doNotDemarcateInvalidXml", "saveXmlDataOnly",
    "useXSLTWhenSaving", "saveThroughXslt", "showXMLTags",
    "alwaysMergeEmptyNamespace", "updateFields", "hdrShapeDefaults", "footnotePr",
    "endnotePr", "compat", "docVars", "rsids", "attachedSchema", "themeFontLang",
    "clrSchemeMapping", "doNotIncludeSubdocsInStats", "doNotAutoCompressPictures",
    "forceUpgrade", "captions", "readModeInkLockDown", "smartTagType",
    "shapeDefaults", "doNotEmbedSmartTags", "decimalSymbol", "listSeparator"]


def fix_schema_quirks(docx_path: pathlib.Path) -> None:
    from lxml import etree
    NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    W, M = "{%s}" % NS_W, "{%s}" % NS_M

    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(docx_path) as z:
        z.extractall(tmp)

    def reorder(parent, order, ns):
        """Stable sort by schema position, leaving unknown elements at the end."""
        rank = {f"{ns}{n}": i for i, n in enumerate(order)}
        big = len(order) + 1
        kids = list(parent)
        for k in sorted(kids, key=lambda e: rank.get(e.tag, big)):
            parent.append(k)

    dp = tmp / "word" / "document.xml"
    tree = etree.parse(str(dp))
    root = tree.getroot()
    for el in root.iter(M + "dPr"):
        reorder(el, MATH_DPR_ORDER, M)
    for el in root.iter(W + "pPr"):
        reorder(el, PPR_ORDER, W)
    for el in root.iter(W + "pgMar"):
        for attr, val in (("gutter", "0"), ("header", "720"), ("footer", "720")):
            if W + attr not in el.attrib:
                el.set(W + attr, val)
    tree.write(str(dp), xml_declaration=True, encoding="UTF-8", standalone=True)

    np_ = tmp / "word" / "numbering.xml"
    if np_.exists():
        tree = etree.parse(str(np_))
        for el in tree.getroot().iter(W + "nsid"):
            v = (el.get(W + "val") or "0").upper()
            el.set(W + "val", v.rjust(8, "0")[-8:])
        tree.write(str(np_), xml_declaration=True, encoding="UTF-8", standalone=True)

    sp = tmp / "word" / "settings.xml"
    if sp.exists():
        tree = etree.parse(str(sp))
        sroot = tree.getroot()
        reorder(sroot, SETTINGS_ORDER, W)
        tree.write(str(sp), xml_declaration=True, encoding="UTF-8", standalone=True)

    docx_path.unlink()
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        ctf = tmp / "[Content_Types].xml"
        z.write(ctf, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ctf:
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reference document with journal styling
# ---------------------------------------------------------------------------
def make_reference_doc(path: pathlib.Path) -> None:
    subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                   stdout=open(path, "wb"), check=True)
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor, Inches

    d = Document(str(path))
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Inches(1.0)
        sec.top_margin = sec.bottom_margin = Inches(1.0)

    def style(name):
        try:
            return d.styles[name]
        except KeyError:
            return None

    body = style("Normal")
    body.font.name = "Times New Roman"
    body.font.size = Pt(10.5)
    body.paragraph_format.space_after = Pt(6)
    body.paragraph_format.line_spacing = 1.15

    bt = style("Body Text")
    if bt is not None:
        bt.font.name = "Times New Roman"
        bt.font.size = Pt(10.5)
        bt.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        bt.paragraph_format.space_after = Pt(6)
        bt.paragraph_format.line_spacing = 1.15

    for i, size in [(1, 13), (2, 11.5), (3, 10.5), (4, 10.5)]:
        h = style(f"Heading {i}")
        if h is None:
            continue
        h.font.name = "Times New Roman"
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.italic = (i >= 3)
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.paragraph_format.space_before = Pt(12 if i == 1 else 9)
        h.paragraph_format.space_after = Pt(5)
        h.paragraph_format.keep_with_next = True

    t = style("Title")
    if t is not None:
        t.font.name = "Times New Roman"
        t.font.size = Pt(17)
        t.font.bold = True
        t.font.color.rgb = RGBColor(0, 0, 0)

    cap = style("Image Caption")
    if cap is not None:
        cap.font.name = "Times New Roman"
        cap.font.size = Pt(8.5)
        cap.font.italic = True
        cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(10)

    for nm in ("Table Caption", "Compact", "Source Code", "Verbatim Char"):
        s = style(nm)
        if s is not None and hasattr(s, "font"):
            s.font.name = "Times New Roman"
    d.save(str(path))
    force_black_headings(path)


def force_black_headings(path: pathlib.Path) -> None:
    """Pandoc's default styles colour headings from the theme.

    A theme colour overrides an explicit RGB value, so the attribute has to be
    removed rather than painted over.
    """
    from lxml import etree
    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    sp = tmp / "word" / "styles.xml"
    tree = etree.parse(str(sp))
    W = "{%s}" % NS_W
    for color in tree.getroot().iter(W + "color"):
        for attr in ("themeColor", "themeTint", "themeShade"):
            color.attrib.pop(W + attr, None)
        color.set(W + "val", "000000")
    tree.write(str(sp), xml_declaration=True, encoding="UTF-8", standalone=True)
    path.unlink()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        ctf = tmp / "[Content_Types].xml"
        z.write(ctf, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ctf:
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)


def shrink_table_text(docx_path: pathlib.Path, pt: float = 7.5,
                      page_in: float = 6.5, column_in: float | None = None,
                      fit_chars: int = 74) -> list[bool]:
    """Set table type size and share the available width out by column content.

    Pandoc leaves column widths to the renderer, which gives a long text column
    the same width as a column holding the word yes. Width is therefore
    allocated in proportion to the mean content length of each column, within
    bounds that keep a narrow column readable and stop one column taking the
    whole table.

    `page_in` is the width available to a table that spans the text block, and
    `column_in` the width available to one inside a two column body. Passing
    `column_in` turns on the narrow case: a table whose natural width fits
    inside `fit_chars` characters is sized to one column and left to flow with
    the text, which avoids the section break a full width float needs and the
    whitespace that break costs. Everything wider is sized to `page_in`.

    Returns one flag per table in document order, true where the table was
    sized to the full text block, so the caller can span exactly those and keep
    widths and section breaks in agreement. An earlier version sized every
    table to the journal's single column measure of 6.5 inches whatever document
    it was building, which left a blank strip beside every full width table in
    the Access layout and overflowed every table that sat in a column.
    """
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, Emu, Inches

    d = Document(str(docx_path))
    wide_flags: list[bool] = []
    for tb in d.tables:
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        tb.autofit = False
        ncol = len(tb.columns)
        if ncol == 0:
            wide_flags.append(True)
            continue

        # natural width: the widest cell in each column, added up
        natural = 0
        for j in range(ncol):
            w = 0
            for row in tb.rows:
                try:
                    w = max(w, len(row.cells[j].text.strip()))
                except IndexError:
                    pass
            natural += w
        wide = column_in is None or natural > fit_chars or ncol >= 6
        wide_flags.append(wide)
        page = Inches(page_in if wide else column_in)

        lens = []
        for j in range(ncol):
            vals = []
            for row in tb.rows:
                try:
                    vals.append(len(row.cells[j].text.strip()))
                except IndexError:
                    vals.append(0)
            body = vals[1:] or vals
            lens.append(max(sum(body) / max(len(body), 1), vals[0] * 0.55, 3.0))
        lo, hi = 0.45, 3.2
        mean = sum(lens) / ncol
        weights = [min(max(v / mean, lo), hi) for v in lens]
        total = sum(weights)
        widths = [Emu(int(page * w / total)) for w in weights]

        # The grid has to agree with the cells. Pandoc writes a tblGrid sized
        # for its own default page, and setting cell widths alone leaves that
        # grid in place. Under a fixed table layout the grid is what a renderer
        # lays out against, so a table sized for one column but still carrying
        # a 5.5 inch grid overflows and collapses into a stack of one word
        # lines, which is what produced the whitespace beside the narrow
        # tables. Grid, table width and cell widths are therefore all written
        # from the same numbers here.
        twips = [max(int(int(w) / 635), 120) for w in widths]
        grid = tb._tbl.find(qn("w:tblGrid"))
        if grid is not None:
            for col in list(grid.findall(qn("w:gridCol"))):
                grid.remove(col)
            for tw in twips:
                col = OxmlElement("w:gridCol")
                col.set(qn("w:w"), str(tw))
                grid.append(col)
        tbl_pr = tb._tbl.tblPr
        for tag in ("w:tblW", "w:tblLayout"):
            for el in tbl_pr.findall(qn(tag)):
                tbl_pr.remove(el)
        tw_el = OxmlElement("w:tblW")
        tw_el.set(qn("w:w"), str(sum(twips)))
        tw_el.set(qn("w:type"), "dxa")
        tbl_pr.append(tw_el)
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)

        for j, w in enumerate(widths):
            for row in tb.rows:
                try:
                    row.cells[j].width = w
                except IndexError:
                    pass
        for row in tb.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)
                    for r in p.runs:
                        r.font.size = Pt(pt)
                        r.font.name = "Times New Roman"
    d.save(str(docx_path))
    return wide_flags


# ---------------------------------------------------------------------------
def main() -> None:
    BUILD.mkdir(exist_ok=True)
    text = SRC.read_text(encoding="utf-8")

    body_text, refs_text = text.split("## References", 1)
    srcs = parse_references(refs_text)
    print(f"parsed {len(srcs)} reference entries")
    missing = [s["num"] for s in srcs if not s["authors"] or not s["year"]]
    if missing:
        print(f"  note: entries with incomplete metadata for Manage Sources: {missing}")

    prepared = prepare_markdown(text)
    md_path = BUILD / "manuscript_pandoc.md"
    md_path.write_text(prepared, encoding="utf-8")

    ref_doc = BUILD / "reference.docx"
    make_reference_doc(ref_doc)

    tmp_docx = BUILD / "stage1.docx"
    subprocess.run(["pandoc", str(md_path), "-o", str(tmp_docx),
                    "--reference-doc", str(ref_doc),
                    "-f", "markdown+pipe_tables+tex_math_dollars",
                    "--wrap=none"], check=True)

    shrink_table_text(tmp_docx)

    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(tmp_docx) as z:
        z.extractall(tmp)
    doc_path = tmp / "word" / "document.xml"
    xml = doc_path.read_bytes()
    xml, n_cites = inject_citations(xml, srcs)
    xml, n_eqs = number_equations(xml)
    xml = wrap_bibliography(xml)
    # The markdown source separates sections with a rule, which pandoc draws as
    # a horizontal line. A journal paper does not use them.
    rules = len(re.findall(rb"<w:p\b(?:(?!</w:p>).)*?o:hr=\"t\".*?</w:p>", xml, re.S))
    xml = re.sub(rb"<w:p\b(?:(?!</w:p>).)*?o:hr=\"t\".*?</w:p>", b"", xml, flags=re.S)
    if rules:
        print(f"  removed {rules} horizontal rules")
    doc_path.write_bytes(xml)

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        ctf = tmp / "[Content_Types].xml"
        z.write(ctf, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ctf:
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)

    fix_schema_quirks(OUT)
    add_bibliography_part(OUT, sources_xml(srcs))
    print(f"citation fields inserted: {n_cites}")
    print(f"displayed equations numbered: {n_eqs}")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
