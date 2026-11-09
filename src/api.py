"""HTTP API for the YOLO11s-cls paddy leaf classifier."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.predict import load_backend, predict_image

BACKEND = "yolo11s-cls"
MAX_BYTES = 12_000_000


def public_result(result: dict) -> dict:
    stage = result["two_stage"]
    disease = stage.get("disease")
    check = result["input_check"]
    return {
        "model": "YOLO11s-cls",
        "label": result["label"],
        "title": result["title"],
        "probability": result["probability"],
        "cause": result["cause"],
        "signs": result["signs"],
        "action": result["action"],
        "ranking": [
            {
                "label": row["label"],
                "title": row["title"],
                "probability": row["probability"],
            }
            for row in result["ranking"]
        ],
        "two_stage": {
            "health": stage["health"],
            "p_healthy": stage["p_healthy"],
            "p_unhealthy": stage["p_unhealthy"],
            "disease": None
            if not disease
            else {
                "label": disease["label"],
                "title": disease["title"],
                "probability": disease["probability"],
                "cause": disease["cause"],
                "signs": disease["signs"],
                "action": disease["action"],
                "ranking": [
                    {
                        "label": row["label"],
                        "title": row["title"],
                        "probability": row["probability"],
                    }
                    for row in disease["ranking"]
                ],
            },
        },
        "input_check": {
            "typical_leaf_photo": check["typical_leaf_photo"],
            "reasons": check["reasons"],
            "guidance": check["guidance"],
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, device, _spec = load_backend(BACKEND)
    app.state.model = model
    app.state.device = device
    yield


app = FastAPI(title="Paddy Doctor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:47321", "http://localhost:47321"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


SITE_PATH = Path(__file__).with_name("site.html")
LOGIN_PATH = Path(__file__).with_name("login.html")
GUIDE_PATH = Path(__file__).with_name("guide.html")
FIELD_BG_PATH = Path(__file__).with_name("field-bg.jpg")

@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return LOGIN_PATH.read_text(encoding="utf-8")

@app.get("/login", response_class=HTMLResponse)
def login_page() -> str:
    return LOGIN_PATH.read_text(encoding="utf-8")

@app.get("/classify", response_class=HTMLResponse)
def classifier_page() -> str:
    return SITE_PATH.read_text(encoding="utf-8")

@app.get("/help", response_class=HTMLResponse)
@app.get("/about", response_class=HTMLResponse)
def guide_page() -> str:
    return GUIDE_PATH.read_text(encoding="utf-8")

@app.get("/field-bg.jpg")
def field_background() -> FileResponse:
    if not FIELD_BG_PATH.is_file():
        raise HTTPException(status_code=404, detail="Background photo is missing.")
    return FileResponse(FIELD_BG_PATH, media_type="image/jpeg")

@app.get("/api/health")
def health() -> dict:
    ready = getattr(app.state, "model", None) is not None
    return {"ready": ready, "model": "YOLO11s-cls"}


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Choose a leaf photo first.")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="That photo is larger than 12 MB.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="Upload a JPEG or PNG photo of a rice leaf."
        ) from exc
    result = predict_image(
        image,
        model=app.state.model,
        device=app.state.device,
        backend=BACKEND,
        topk=10,
    )
    return public_result(result)
