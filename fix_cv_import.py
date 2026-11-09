from pathlib import Path
import re

root = Path(".").resolve()
app_path = root / "app.py"
app = app_path.read_text(encoding="utf-8")
imported = re.search(r"from src\.config import \((.*?)\)", app, re.S)
if not imported or "CV_METRICS_PATH" not in imported.group(1):
    if "from src.config import CV_METRICS_PATH" not in app:
        app = app.replace(
            "import streamlit as st\n",
            "import streamlit as st\nfrom src.config import CV_METRICS_PATH\n",
            1,
        )
        app_path.write_text(app, encoding="utf-8")

config_path = root / "src" / "config.py"
cfg = config_path.read_text(encoding="utf-8")
if "CV_METRICS_PATH" not in cfg:
    line = 'CV_METRICS_PATH = REPORTS_DIR / "cv_metrics.json"\n'
    anchor = 'REPORTS_DIR = ROOT / "reports"\n'
    cfg = cfg.replace(anchor, anchor + line, 1) if anchor in cfg else cfg + "\n" + line
    config_path.write_text(cfg, encoding="utf-8")
print("fixed")