"""Render a reference PDF whose equations are certain to be visible.

The submission artifact is the .docx, where equations are Office Math Markup so
that Word opens them in the equation editor. That markup cannot be checked by
eye in this environment: the LibreOffice build available here drops OMML on
import, so a PDF converted from the .docx shows the equations as blank gaps.
The same happens for a one line document produced by pandoc alone, so it is a
limitation of that converter rather than a defect in the document.

This script therefore renders the same manuscript through LaTeX, where the
mathematics is typeset from the original sources. Use it to check what every
equation says. It is not a proof of how Word lays the document out, and the
.docx remains the file to submit.

Displayed equations are wrapped in an equation environment so the reference
numbering matches the numbering written into the .docx.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
SRC_MD = BUILD / "manuscript_pandoc.md"
OUT = ROOT / "DyHGT_MARL_Math_Reference.pdf"


def have(package: str) -> bool:
    r = subprocess.run(["kpsewhich", f"{package}.sty"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def main() -> None:
    if not SRC_MD.exists():
        sys.exit(f"{SRC_MD} not found, run build_journal_docx.py first")
    work = BUILD / "latex"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    tex = work / "reference.tex"
    subprocess.run(["pandoc", str(SRC_MD), "-o", str(tex), "-s",
                    "-V", "geometry:margin=1in", "-V", "fontsize=10pt"], check=True)
    t = tex.read_text(encoding="utf-8")

    # drop preamble packages this TeX installation does not carry. Without
    # lmodern the default fonts are bitmap, and microtype's font expansion is a
    # fatal error on those, so microtype goes too whenever lmodern is absent.
    scalable = have("lmodern")
    for pkg in ("lmodern", "microtype"):
        if not have(pkg) or (pkg == "microtype" and not scalable):
            t = re.sub(rf"^.*\\usepackage(\[[^\]]*\])?\{{{pkg}\}}.*$", "", t, flags=re.M)
    if not have("microtype") or not scalable:
        t = t.replace("\\IfFileExists{microtype.sty}{% use microtype if available", "\\iffalse")
        t = t.replace("  \\UseMicrotypeSet[protrusion]{basicmath} % disable protrusion for tt fonts\n}{}",
                      "\\fi")

    # number displayed equations, matching the .docx
    t = t.replace("\\[", "\\begin{equation}").replace("\\]", "\\end{equation}")

    tex.write_text(t, encoding="utf-8")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name],
                       cwd=work, capture_output=True)

    pdf = work / "reference.pdf"
    if not pdf.exists():
        log = (work / "reference.log").read_text(errors="ignore")
        bad = [l for l in log.splitlines() if l.startswith("!")][:5]
        sys.exit("pdflatex produced no output:\n  " + "\n  ".join(bad))
    shutil.copy(pdf, OUT)

    import pypdf
    r = pypdf.PdfReader(OUT)
    text = "\n".join(p.extract_text() for p in r.pages)
    # Probes are tokens that genuinely occur in the seven displayed equations,
    # so a miss means the math did not render rather than that the probe is
    # stale. An earlier list tested for argmax, which no equation uses.
    probes = ("(V", "softmax", "exp", "log")
    rendered = sum(1 for probe in probes if probe in text)
    print(f"wrote {OUT.name}: {len(r.pages)} pages, "
          f"{rendered} of {len(probes)} equation probes found in the rendered text")


if __name__ == "__main__":
    main()
