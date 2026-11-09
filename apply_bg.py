"""Point the site background at the sharp field photo."""

import re
from pathlib import Path

page = Path("src/site.html")
html = page.read_text(encoding="utf-8")
style = (
    '<style id="field-bg">'
    'body{background:#c9d7b8 url("/field-bg.jpg") center 30% / cover no-repeat !important;}'
    "</style>"
)
if re.search(r'(?s)<style id="field-bg">.*?</style>', html):
    html = re.sub(r'(?s)<style id="field-bg">.*?</style>', style, html, count=1)
elif "</head>" in html:
    html = html.replace("</head>", style + "</head>", 1)
else:
    raise SystemExit("Could not find the page head in src/site.html")
page.write_text(html, encoding="utf-8")
print("Background uses src/field-bg.jpg")
