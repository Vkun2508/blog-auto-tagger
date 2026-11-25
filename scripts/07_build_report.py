"""Step 7: turn reports/report.md into a styled HTML page and a PDF.

Run it with:   python scripts/07_build_report.py
Output:        reports/report.html and reports/Project_Report.pdf (about 10 seconds)

The PDF is printed by Microsoft Edge, which is built into Windows, so no extra software is needed.
Charts are the PNG files in reports/figures/, referenced from the Markdown.
"""

import subprocess
import sys
from pathlib import Path

import markdown

PROJECT = Path(__file__).parent.parent
REPORTS = PROJECT / "reports"
SOURCE = REPORTS / "report.md"
HTML_OUT = REPORTS / "report.html"
PDF_OUT = REPORTS / "Project_Report.pdf"

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]

# Same "editor's red pencil" look as the app: paper, ink, one red accent.
CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
:root { --paper:#FFFFFF; --ink:#1F2328; --muted:#5B6168; --rule:#E3DDD2; --tint:#F6F2EA; --red:#B3261E; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { margin:0; background:var(--paper); color:var(--ink);
       font: 10.5pt/1.55 "IBM Plex Sans", "Segoe UI", Arial, sans-serif; }
h1, h2, h3 { font-family: "Source Serif 4", Cambria, Georgia, serif; color: var(--ink); line-height: 1.2; }
h1 { font-size: 22pt; margin: 0 0 6pt; }
h2 { font-size: 15pt; margin: 20pt 0 6pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt; padding-left: 16pt; }
a { color: var(--red); text-decoration: none; }
code { font-family: "IBM Plex Mono", Consolas, monospace; font-size: 9pt; background: var(--tint); padding: 0 3px; border-radius: 3px; }
pre { background: var(--tint); padding: 8pt 10pt; border-radius: 6px; font-size: 8.5pt; line-height: 1.4; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; break-inside: avoid;
        font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4pt 6pt; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; border-bottom: 1.5px solid var(--ink); }
img { max-width: 100%; display: block; margin: 6pt auto 2pt; break-inside: avoid; }
img + em, p > em:only-child { display: block; text-align: center; color: var(--muted); font-size: 8.5pt; margin-bottom: 8pt; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: var(--tint); border-radius: 6px; }
blockquote p:last-child { margin: 0; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 14pt 0; }
.cover { height: 255mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }
.cover .kicker { font-family: "IBM Plex Mono", Consolas, monospace; font-size: 9pt; letter-spacing: .08em;
                 text-transform: uppercase; color: var(--red); }
.cover h1 { font-size: 34pt; margin: 10pt 0 12pt; max-width: 15ch; }
.cover .lede { font-size: 12.5pt; color: var(--muted); max-width: 52ch; }
.cover .byline { font-size: 11pt; border-top: 1px solid var(--rule); padding-top: 10pt; }
.cover .byline strong { font-weight: 600; }
"""

COVER = """
<section class="cover">
  <div>
    <div class="kicker">Project report</div>
    <h1>Automated Content Tagging for Blogs</h1>
    <p class="lede">A multi-label system that reads a blog post, suggests up to five tags from an approved
    taxonomy with a score for each, and routes uncertain drafts to an editor.</p>
  </div>
  <div class="byline"><strong>Vinochan R</strong> &nbsp;·&nbsp; Persevex AI Internship</div>
</section>
"""


def build_html():
    """Markdown -> one HTML page with the cover and the print stylesheet."""
    body = markdown.markdown(
        SOURCE.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    fonts = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
             "&family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:wght@600;700&display=swap")
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Automated Content Tagging for Blogs - Project Report</title>"
        f"<link rel='stylesheet' href='{fonts}'><style>{CSS}</style></head>"
        f"<body>{COVER}<main>{body}</main></body></html>"
    )
    HTML_OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {HTML_OUT}")


def print_pdf():
    """Ask Edge (headless, no window) to print the HTML page to PDF."""
    edge = next((p for p in EDGE_PATHS if p.exists()), None)
    if edge is None:
        sys.exit("Microsoft Edge was not found. Open reports/report.html in a browser and print it to PDF.")
    subprocess.run(
        [str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--virtual-time-budget=15000", f"--print-to-pdf={PDF_OUT}", HTML_OUT.resolve().as_uri()],
        check=True, capture_output=True, timeout=120,
    )
    print(f"Wrote {PDF_OUT} ({PDF_OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build_html()
    print_pdf()
