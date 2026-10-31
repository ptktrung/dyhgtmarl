"""Report the page count of a built .docx.

An earlier version of this project claimed the available converter could not
be trusted for pagination. That was tested rather than assumed once a file
paginated by Word itself became available: Word reported 29 pages for it and
the local converter reported 30, so the converter is accurate to within about
one page and is used here as the working measure. It still drops Office Math
Markup on import, which costs a fraction of a page across seven displayed
equations, so a reported count is a slight underestimate.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def word_pages(docx: pathlib.Path) -> int | None:
    """The count Word itself stored, when the file has been through Word."""
    try:
        app = zipfile.ZipFile(docx).read("docProps/app.xml").decode("utf8")
    except KeyError:
        return None
    import re
    m = re.search(r"<Pages>(\d+)</Pages>", app)
    return int(m.group(1)) if m else None


def converted_pages(docx: pathlib.Path) -> int | None:
    import pypdf
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                            "--outdir", td, str(docx)],
                           capture_output=True, timeout=900)
        pdf = pathlib.Path(td) / (docx.stem + ".pdf")
        if not pdf.exists():
            print(r.stderr.decode()[-400:], file=sys.stderr)
            return None
        return len(pypdf.PdfReader(str(pdf)).pages)


def main() -> None:
    targets = [pathlib.Path(a) for a in sys.argv[1:]] or \
              [ROOT / "DyHGT_IEEE_Access_Submission.docx"]
    for t in targets:
        if not t.exists():
            print(f"{t.name}: not found")
            continue
        w, c = word_pages(t), converted_pages(t)
        parts = [f"{t.name}:"]
        if c is not None:
            parts.append(f"{c} pages (converted, about +1 against Word)")
        if w:
            parts.append(f"Word stored {w}")
        print(" ".join(parts))


if __name__ == "__main__":
    main()
