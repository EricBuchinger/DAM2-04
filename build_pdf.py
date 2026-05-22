"""Render ue4_solution.md -> ue4_solution.pdf via markdown -> HTML -> headless Chrome."""
import re
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).parent
MD = HERE / "ue4_solution.md"
HTML = HERE / "ue4_solution.html"
PDF = HERE / "ue4_solution.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

text = MD.read_text()
# strip pandoc-style title block (% title / % author / % date) - we render our own H1
text = re.sub(r"^(%.*\n)+", "", text, count=1)

html_body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "toc"],
)

css = """
@page { size: A4; margin: 22mm 18mm; }
body {
  font-family: -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #222;
  max-width: 100%;
}
h1 { font-size: 22pt; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 15pt; border-bottom: 1px solid #bbb; padding-bottom: 3px; margin-top: 24px; }
h3 { font-size: 12pt; margin-top: 18px; }
p, li { text-align: justify; }
table { border-collapse: collapse; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #999; padding: 5px 8px; }
th { background: #eee; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 10pt; }
pre { background: #f3f3f3; padding: 10px; border-radius: 4px; font-size: 10pt; overflow-x: auto; }
pre code { background: none; padding: 0; }
img { max-width: 100%; height: auto; display: block; margin: 10px auto; }
hr { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; color: #555; }
"""

mathjax = """
<script>
window.MathJax = {
  tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] },
  svg: { fontCache: 'global' }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
"""

HTML.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Assignment 4 - Pokemon Analysis</title>
<style>{css}</style>
{mathjax}
</head><body>
{html_body}
</body></html>
""")

subprocess.run([
    CHROME,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--virtual-time-budget=10000",
    f"--print-to-pdf={PDF}",
    HTML.as_uri(),
], check=True)

print(f"Wrote {PDF} ({PDF.stat().st_size // 1024} KB)")
