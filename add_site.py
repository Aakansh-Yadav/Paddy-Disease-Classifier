from pathlib import Path
p = Path("src/api.py")
text = p.read_text(encoding="utf-8-sig")
if "site.html" not in text:
    text = text.replace(
        "from fastapi import FastAPI, File, HTTPException, UploadFile\n",
        "from pathlib import Path\n\nfrom fastapi import FastAPI, File, HTTPException, UploadFile\nfrom fastapi.responses import HTMLResponse\n",
        1,
    )
    route = """SITE_PATH = Path(__file__).with_name(\"site.html\")\n\n@app.get(\"/\", response_class=HTMLResponse)\ndef home() -> str:\n    return SITE_PATH.read_text(encoding=\"utf-8\")\n\n"""
    needle = '@app.get("/api/health")\n'
    if needle not in text:
        raise SystemExit("Could not update src/api.py")
    text = text.replace(needle, route + needle, 1)
    p.write_text(text, encoding="utf-8")
print("ready")
