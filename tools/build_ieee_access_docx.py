"""Build the IEEE Access submission file from the manuscript source.

The journal version and this one share their content and differ in presentation.
This script produces the layout the IEEE Access template defines: the page size
and margins of that template, a single column title block, a two column body,
the template's own paragraph styles, numbered equations with the number at the
template's tab stop, figures and wide tables spanning both columns, and author
biographies with photographs after the references.

The approach is to let pandoc convert the body with the template supplied as a
reference document, so the template's styles, theme and fonts travel into the
output, and then to rewrite the parts pandoc has no way to know about. Those
are the title block, the section structure that gives a two column body with
full width floats, the mapping from pandoc's style names onto the template's,
the equation numbering, and the biographies.

Run build_journal_docx.py first or alongside it. The two outputs are built from
the same `docs/manuscript.md` and the same result files, so their content stays
identical by construction.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import build_journal_docx as J

TEMPLATE = ROOT / "assets" / "Access-Template-2024.docx"
BIOS = ROOT / "assets" / "author_bios.docx"
BUILD = ROOT / "build"
OUT = ROOT / "DyHGT_IEEE_Access_Submission.docx"

NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

# pandoc's style names on the left, the style used in the output on the right.
# The body starts at a level two markdown heading, so pandoc's Heading2 is a
# numbered section of the paper.
STYLE_MAP = {
    "Heading2": "AccSec", "Heading3": "AccSub", "Heading4": "AccSubSub",
    "Heading1": "AccSec",
    "FirstParagraph": "PARA", "BodyText": "PARAIndent",
    "ImageCaption": "FigureCaption", "Caption": "FigureCaption",
    "CaptionedFigure": "FigureCaption", "SourceCode": "Text", "BlockText": "PARA",
}

# The template's own heading styles carry list numbering, which would run
# alongside the numbers already written into the headings and would also break
# every cross reference in the text. These three are the template's styles with
# the numbering switched off, so what the heading says is what appears.
EXTRA_STYLES = (
    '<w:style w:type="paragraph" w:customStyle="1" w:styleId="AccSec">'
    '<w:name w:val="Acc Section"/><w:basedOn w:val="H1"/>'
    '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr></w:pPr>'
    '<w:rPr><w:caps/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:customStyle="1" w:styleId="AccSub">'
    '<w:name w:val="Acc Subsection"/><w:basedOn w:val="H2First"/>'
    '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr></w:pPr>'
    '</w:style>'
    '<w:style w:type="paragraph" w:customStyle="1" w:styleId="AccSubSub">'
    '<w:name w:val="Acc Sub subsection"/><w:basedOn w:val="H3"/>'
    '<w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr></w:pPr>'
    '</w:style>'
)

# page geometry copied from the template section properties
PG = ('<w:pgSz w:w="11520" w:h="15660" w:code="1"/>'
      '<w:pgMar w:top="1300" w:right="740" w:bottom="1040" w:left="740" '
      'w:header="360" w:footer="640" w:gutter="0"/>')
SECT_ONE = f'<w:sectPr><w:type w:val="continuous"/>{PG}<w:cols w:space="720"/></w:sectPr>'
SECT_TWO = (f'<w:sectPr><w:type w:val="continuous"/>{PG}'
            f'<w:cols w:num="2" w:space="400"/></w:sectPr>')


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def para(style: str, runs: str, sect: str = "") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{sect}</w:pPr>{runs}</w:p>'


def run(text: str, *, bold=False, sup=False, italic=False) -> str:
    rpr = ""
    if bold:
        rpr += "<w:b/>"
    if italic:
        rpr += "<w:i/>"
    if sup:
        rpr += '<w:vertAlign w:val="superscript"/>'
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


# ---------------------------------------------------------------------------
# front matter
# ---------------------------------------------------------------------------
def parse_front(md: str) -> dict:
    """Pull the title, authors, affiliations, abstract and keywords apart."""
    head = md.split("## Abstract")[0]
    title = re.search(r"^# (.+)$", head, re.M).group(1).strip()
    authors_line = re.search(r"^\*\*(.+)$", head, re.M).group(0)
    authors = []
    for m in re.finditer(r"\*\*([^*]+)\*\*\s*\(([\d,\s]+)\)", authors_line):
        authors.append((m.group(1).strip(), [a.strip() for a in m.group(2).split(",")]))
    affs = re.findall(r"^\((\d)\) (.+)$", head, re.M)
    abstract = md.split("## Abstract")[1].split("**Keywords:**")[0].strip()
    keywords = md.split("**Keywords:**")[1].split("\n")[0].strip()
    # Read the corresponding author from the manuscript rather than repeating it
    # here. An earlier version hardcoded the line and its address drifted from
    # the one in the manuscript, which is the kind of divergence a reader finds
    # and an author does not.
    cm = re.search(r"^Corresponding author:\s*(.+?)\s*\(([^)]+)\)\s*$", head, re.M)
    if not cm:
        sys.exit("no corresponding author line found in the manuscript front matter")
    corresponding = f"{cm.group(1)} (e-mail: {cm.group(2)})"
    return {"title": title, "authors": authors, "affs": affs,
            "abstract": abstract, "keywords": keywords,
            "corresponding": corresponding}


def front_xml(f: dict, corresponding: str) -> str:
    out = [
        para("DOP", run("Date of publication xxxx 00, 0000, date of current version "
                        "xxxx 00, 0000.")),
        para("DOI", run("Digital Object Identifier 10.1109/ACCESS.2024.Doi Number")),
        para("PaperTitle", run(f["title"])),
    ]

    runs = []
    for i, (name, marks) in enumerate(f["authors"]):
        if i:
            runs.append(run(", " if i < len(f["authors"]) - 1 else ", and "))
        runs.append(run(name.upper()))
        runs.append(run(",".join(marks), sup=True))
    out.append(para("AU", "".join(runs)))

    for i, (num, text) in enumerate(f["affs"]):
        style = "PINoSpace" if i < len(f["affs"]) - 1 else "PI"
        out.append(para(style, run(num, sup=True) + run(text)))
    out.append(para("PI", run(f"Corresponding author: {corresponding}")))

    out.append(para("Abstract", run("ABSTRACT ", bold=True) + run(f["abstract"])))
    out.append(para("IT", run("INDEX TERMS ", bold=True) + run(f["keywords"]),
                    sect=SECT_ONE))
    return "".join(out)


# ---------------------------------------------------------------------------
# biographies
# ---------------------------------------------------------------------------
def bios_xml(tmp: pathlib.Path, next_rid: int) -> tuple[str, list[tuple[str, bytes, str]]]:
    """Return biography paragraphs plus the image parts they reference."""
    if not BIOS.exists():
        return "", []
    import docx

    d = docx.Document(str(BIOS))
    z = zipfile.ZipFile(BIOS)
    media = sorted(n for n in z.namelist() if n.startswith("word/media/"))

    blocks, current = [], []
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        # a new biography starts where a surname in capitals leads the line
        if re.match(r"^\{?[A-Z][A-Za-z.\-]*(\s+[A-Z][A-Za-z.\-]*)*\}?\s+(received|was born)", t):
            if current:
                blocks.append(current)
            current = [t]
        elif current:
            current.append(t)
    if current:
        blocks.append(current)

    parts, xml = [], []
    for i, block in enumerate(blocks):
        rid = f"rIdBio{i}"
        if i < len(media):
            data = z.read(media[i])
            ext = pathlib.Path(media[i]).suffix
            name = f"media/bio{i}{ext}"
            parts.append((name, data, rid))
            xml.append(image_para(rid, i))
        first = block[0].replace("{", "").replace("}", "")
        m = re.match(r"^(.+?)\s+(received|was born)", first)
        if m:
            lead, rest = m.group(1), first[len(m.group(1)):].lstrip()
            body = run(lead.upper() + " ", bold=True) + run(rest)
        else:
            body = run(first)
        xml.append(para("AUBios", body))
        for extra in block[1:]:
            xml.append(para("AUBiosNoSpace", run(extra)))
    return "".join(xml), parts


def image_para(rid: str, idx: int) -> str:
    cx, cy = 914400, 1143000          # one inch by one and a quarter, IEEE bio size
    return (
        '<w:p><w:pPr><w:pStyle w:val="AUBios"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0" '
        f'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{900 + idx}" name="bio{idx}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{900 + idx}" name="bio{idx}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" r:embed="{rid}"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )


# ---------------------------------------------------------------------------
# body rewriting
# ---------------------------------------------------------------------------
def remap_styles(x: str) -> str:
    # pandoc writes the attribute with a trailing space, so match loosely
    def one(m):
        return f'<w:pStyle w:val="{STYLE_MAP.get(m.group(1), m.group(1))}"/>'
    return re.sub(r'<w:pStyle w:val="([^"]+)"\s*/>', one, x)


def uppercase_sections(xml: bytes) -> bytes:
    """Set section headings in capitals, as the template does.

    This runs last. Citation injection and the bibliography both locate the
    reference list by its heading text, so changing that text earlier would
    send citation fields into the reference list itself.
    """
    x = xml.decode("utf-8")

    def upper(m):
        return re.sub(r"(<w:t[^>]*>)([^<]*)(</w:t>)",
                      lambda r: r.group(1) + r.group(2).upper() + r.group(3), m.group(0))

    return re.sub(r"<w:p\b[^>]*>(?:(?!</w:p>).)*?AccSec(?:(?!</w:p>).)*?</w:p>",
                  upper, x, flags=re.S).encode("utf-8")


def style_float_captions(x: str) -> str:
    """A paragraph that opens with a bold Table label becomes a table caption."""
    def sub(m):
        body = m.group(0)
        if re.search(r"<w:b\s*/>.{0,80}?<w:t[^>]*>\s*Table [IVX]+", body, re.S):
            return re.sub(r'<w:pStyle w:val="[^"]+"\s*/>',
                          '<w:pStyle w:val="TableCaption"/>', body, count=1)
        return body
    return re.sub(r"<w:p\b[^>]*>.*?</w:p>", sub, x, flags=re.S)


def keep_floats_together(x: str) -> tuple[str, int]:
    """Stop a caption separating from the float it belongs to.

    Word breaks a column wherever it runs out of room, and with nothing holding
    a caption to its table or figure it will set the caption at the foot of one
    column and the float at the head of the next. What the reader sees is a
    block of white space where the float should have been. Three such gaps were
    reported on the author's copy, below Table III, beside Table XIV and below
    Table XV.

    The fix is the mechanism Word provides: keepNext binds a paragraph to the
    one after it, and keepLines stops a paragraph splitting across a break. A
    table caption is written above its table, so the caption keeps with what
    follows. A figure caption is written below its image, so the image keeps
    with what follows and the caption is the thing kept to. Both are applied by
    rewriting the paragraph properties in place.
    """
    keep = "<w:keepNext/><w:keepLines/>"
    chunks = re.findall(r"<w:p\b[^>]*>.*?</w:p>|<w:tbl>.*?</w:tbl>", x, re.S)
    n = 0

    def add(p: str) -> str:
        if "<w:keepNext/>" in p:
            return p
        if "<w:pPr>" in p:
            return p.replace("<w:pPr>", "<w:pPr>" + keep, 1)
        return re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + keep + "</w:pPr>", p, count=1)

    out = list(chunks)
    for i, c in enumerate(chunks):
        is_tbl_cap = 'w:val="TableCaption"' in c
        is_fig_cap = 'w:val="FigureCaption"' in c
        has_image = "<w:drawing>" in c
        # a caption above its table, and the image above its caption
        if is_tbl_cap or has_image:
            out[i] = add(c)
            n += 1
        # a figure caption should not split across a column break either
        elif is_fig_cap:
            out[i] = add(c).replace(keep, "<w:keepLines/>", 1)
            n += 1
    for old, new in zip(chunks, out):
        if old != new:
            x = x.replace(old, new, 1)
    return x, n


def number_equations(x: str) -> tuple[str, int]:
    """Template style Equation, with the number at the style's own tab stop."""
    math_re = re.compile(r"<m:oMathPara>\s*(?:<m:oMathParaPr>.*?</m:oMathParaPr>)?"
                         r"\s*(<m:oMath>.*?</m:oMath>)\s*</m:oMathPara>", re.S)
    count = 0

    def do(pm):
        nonlocal count
        p = pm.group(0)
        mm = math_re.search(p)
        if not mm:
            return p
        count += 1
        return ('<w:p><w:pPr><w:pStyle w:val="Equation"/></w:pPr>'
                + mm.group(1)
                + '<w:r><w:tab/></w:r>'
                + f'<w:r><w:t xml:space="preserve">({count})</w:t></w:r></w:p>')

    return re.sub(r"<w:p\b[^>]*>.*?</w:p>", do, x, flags=re.S), count


def strip_filler(x: str) -> tuple[str, int, int]:
    """Remove the blank paragraphs and rules that pandoc leaves behind.

    The manuscript separates its sections with a markdown rule, which pandoc
    turns into a horizontal line, and leaves an empty paragraph wherever the
    source had a blank line inside a block. Neither belongs in an IEEE layout,
    and in a two column body each one shows as a gap. Paragraphs carrying a
    section break or a figure are left alone, because removing those would
    change the layout rather than tidy it.
    """
    rules = len(re.findall(r"<w:p\b[^>]*>(?:(?!</w:p>).)*?o:hr=\"t\".*?</w:p>", x, re.S))
    x = re.sub(r"<w:p\b[^>]*>(?:(?!</w:p>).)*?o:hr=\"t\".*?</w:p>", "", x, flags=re.S)

    blanks = 0
    out = []
    for chunk in re.findall(r"<w:p\b[^>]*>.*?</w:p>|<w:p\b[^>]*/>|<w:tbl>.*?</w:tbl>|[^<]+|<[^>]+>", x, re.S):
        if chunk.startswith("<w:p") and chunk.endswith("</w:p>"):
            text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", chunk, re.S)).strip()
            has_keep = ("<w:sectPr" in chunk or "<w:drawing>" in chunk
                        or "<m:oMath" in chunk or "<w:fldChar" in chunk)
            if not text and not has_keep:
                blanks += 1
                continue
        out.append(chunk)
    return "".join(out), rules, blanks


def span_floats(x: str, table_wide: list[bool] | None = None) -> tuple[str, int]:
    """Give figures and wide tables the full page width.

    A two column body is one continuous section. A float that must cross the
    column boundary sits in a one column section of its own, which is the
    mechanism the template itself uses for its wide figures. A section break
    lives on a paragraph, so where a float is preceded by a table an empty
    paragraph is inserted to carry the break. Consecutive floats share a single
    section rather than switching back and forth, which keeps the number of
    breaks, and the whitespace they cost, to a minimum.
    """
    chunks = re.findall(r"<w:p\b[^>]*>.*?</w:p>|<w:tbl>.*?</w:tbl>", x, re.S)
    if not chunks:
        return x, 0
    head = x[:x.index(chunks[0])]
    tail = x[x.index(chunks[-1]) + len(chunks[-1]):]

    def is_caption(c):
        return 'w:val="FigureCaption"' in c or 'w:val="TableCaption"' in c

    # Tables are spanned on the same test that decided their width, so a table
    # is never sized for one measure and placed in the other. The flag is
    # computed once per chunk, because the walk below inspects a chunk more
    # than once and a running counter would drift.
    float_flag: list[bool] = []
    tbl_i = 0
    for c in chunks:
        if "<w:drawing>" in c:
            float_flag.append(True)
        elif c.startswith("<w:tbl>"):
            if table_wide is not None:
                float_flag.append(tbl_i < len(table_wide) and table_wide[tbl_i])
            else:
                first = re.search(r"<w:tr\b.*?</w:tr>", c, re.S)
                float_flag.append(bool(first) and first.group(0).count("<w:tc>") >= 4)
            tbl_i += 1
        else:
            float_flag.append(False)

    def is_float(k):
        return float_flag[k]

    def breaker(sect):
        return ('<w:p><w:pPr><w:pStyle w:val="PARA"/>'
                '<w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
                f'<w:rPr><w:sz w:val="2"/></w:rPr>{sect}</w:pPr></w:p>')

    def carry(p, sect):
        if not p.startswith("<w:p"):
            return None
        if "<w:pPr>" in p:
            return p.replace("</w:pPr>", sect + "</w:pPr>", 1)
        return re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + sect + "</w:pPr>", p, count=1)

    out, spans, i = [], 0, 0
    while i < len(chunks):
        if not is_float(i):
            out.append(chunks[i])
            i += 1
            continue

        # the whole run of floats and their captions goes in one section
        group, j = [], i
        while j < len(chunks) and (is_float(j) or
                                   (group and is_caption(chunks[j]))):
            group.append(chunks[j])
            j += 1

        # A table caption is written before its table, so without this the
        # caption stays in the two column flow while the table spans, which
        # separates them and leaves a gap where the caption used to sit.
        while out and is_caption(out[-1]):
            group.insert(0, out.pop())

        moved = carry(out[-1], SECT_TWO) if out else None
        if moved is not None:
            out[-1] = moved
        else:
            out.append(breaker(SECT_TWO))
        out.extend(group)
        out.append(breaker(SECT_ONE))
        spans += 1
        i = j
    return head + "".join(out) + tail, spans


def scale_images(x: str, max_cx: int = 6350000) -> str:
    """Hold every figure to the printable width of the page."""
    def fix(m):
        cx, cy = int(m.group(1)), int(m.group(2))
        if cx <= max_cx:
            return m.group(0)
        k = max_cx / cx
        return m.group(0).replace(f'cx="{cx}"', f'cx="{max_cx}"') \
                         .replace(f'cy="{cy}"', f'cy="{int(cy * k)}"')
    return re.sub(r'cx="(\d+)" cy="(\d+)"', fix, x)


# ---------------------------------------------------------------------------
def main() -> None:
    if not TEMPLATE.exists():
        sys.exit(f"template not found at {TEMPLATE}")
    BUILD.mkdir(exist_ok=True)
    text = (ROOT / "docs" / "manuscript.md").read_text(encoding="utf-8")
    body_text, refs_text = text.split("## References", 1)
    srcs = J.parse_references(refs_text)
    print(f"parsed {len(srcs)} reference entries")

    prepared = J.prepare_markdown(text)
    front = parse_front(prepared)
    corresponding = front["corresponding"]

    # pandoc converts from the first section heading onward
    cut = prepared.index("## I. Introduction")
    body_md = prepared[cut:]
    # keep the contribution list as plain numbered paragraphs, because a Word
    # list would renumber itself against the template's own definitions
    body_md = re.sub(r"^(\d+)\. ", lambda m: f"{m.group(1)}\\) ", body_md, flags=re.M)
    md_path = BUILD / "access_body.md"
    md_path.write_text(body_md, encoding="utf-8")

    stage = BUILD / "access_stage.docx"
    subprocess.run(["pandoc", str(md_path), "-o", str(stage),
                    "--reference-doc", str(TEMPLATE),
                    "-f", "markdown+pipe_tables+tex_math_dollars",
                    "--wrap=none"], check=True)

    # Access geometry: text block 10040 twips wide, two columns of 4820 with a
    # 400 twip gutter. Tables are sized to whichever of the two they will sit
    # in, and the flags say which that was.
    table_wide = J.shrink_table_text(stage, pt=7.0, page_in=6.95, column_in=3.33, fit_chars=88)
    print(f"tables: {sum(table_wide)} full width, "
          f"{len(table_wide) - sum(table_wide)} inside a column")

    tmp = pathlib.Path(tempfile.mkdtemp())
    with zipfile.ZipFile(stage) as z:
        z.extractall(tmp)
    styles_path = tmp / "word" / "styles.xml"
    sx = styles_path.read_text(encoding="utf-8")
    if "AccSec" not in sx:
        styles_path.write_text(sx.replace("</w:styles>", EXTRA_STYLES + "</w:styles>"),
                               encoding="utf-8")

    doc = tmp / "word" / "document.xml"
    x = doc.read_text(encoding="utf-8")

    x = remap_styles(x)
    x = style_float_captions(x)
    x, n_eq = number_equations(x)
    # after equation numbering, so an equation paragraph is never mistaken for
    # a blank one, and before spanning, so removed blanks cannot carry a break
    x, n_rules, n_blanks = strip_filler(x)
    print(f"removed {n_rules} horizontal rules and {n_blanks} blank paragraphs")
    x = scale_images(x)

    bios, bio_parts = bios_xml(tmp, 0)

    start = x.index(">", x.index("<w:body")) + 1
    end = x.index("</w:body>")
    inner = x[start:end]
    inner = re.sub(r"<w:sectPr\b.*?</w:sectPr>", "", inner, flags=re.S)
    if os.environ.get("ACCESS_SPAN_FLOATS", "1") == "1":
        inner, n_span = span_floats(inner, table_wide)
    else:
        inner, n_span = inner, 0

    # Applied after the biographies are joined on, so a portrait is held with
    # the biography it belongs to as well.
    inner, n_keep = keep_floats_together(inner + bios)
    bios = ""

    final = (front_xml(front, corresponding) + inner + bios
             + f"<w:sectPr>{PG}<w:cols w:num=\"2\" w:space=\"400\"/></w:sectPr>")
    x = x[:start] + final + x[end:]

    xml, n_cites = J.inject_citations(x.encode("utf-8"), srcs)
    xml = J.wrap_bibliography(xml)
    xml = uppercase_sections(xml)
    doc.write_bytes(xml)

    # biography photographs
    if bio_parts:
        rels_path = tmp / "word" / "_rels" / "document.xml.rels"
        rels = rels_path.read_text(encoding="utf-8")
        add = ""
        for name, data, rid in bio_parts:
            (tmp / "word" / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp / "word" / name).write_bytes(data)
            add += (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                    f'officeDocument/2006/relationships/image" Target="{name}"/>')
        rels_path.write_text(rels.replace("</Relationships>", add + "</Relationships>"),
                             encoding="utf-8")
        ct = tmp / "[Content_Types].xml"
        c = ct.read_text(encoding="utf-8")
        if 'Extension="png"' not in c:
            c = c.replace("</Types>", '<Default Extension="png" '
                          'ContentType="image/png"/></Types>')
        ct.write_text(c, encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        ctf = tmp / "[Content_Types].xml"
        z.write(ctf, "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f != ctf:
                z.write(f, f.relative_to(tmp).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)

    J.fix_schema_quirks(OUT)
    J.add_bibliography_part(OUT, J.sources_xml(srcs))
    print(f"equations numbered: {n_eq}")
    print(f"full width floats: {n_span}")
    print(f"floats bound to their captions: {n_keep}")
    print(f"citation fields inserted: {n_cites}")
    print(f"biographies: {len(bio_parts)} with photographs")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
