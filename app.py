"""Streamlit app: explore the dataset, inspect errors, and classify a leaf."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import (
    CLASS_DIST_PATH,
    CLASS_INFO,
    CLASSES,
    CLASS_TO_IDX,
    COMPARISON_PATH,
    MODEL_REGISTRY,
    SAMPLES_PATH,
    TRAIN_CSV,
    TRAIN_IMAGES,
)
from src.dataset import load_metadata
from src.explain import maybe_heatmap
from src.predict import available_backends, load_backend, predict_image

st.set_page_config(
    page_title="Paddy Disease Classifier",
    page_icon="🌾",
    layout="wide",
)

ROADMAP = [
    (
        "1. Problem definition",
        "Classify a paddy (rice) leaf photo into one of ten labels: nine diseases plus a healthy leaf. The output is a class name and a confidence score a field worker can act on.",
    ),
    (
        "2. Dataset intake",
        "Use the Kaggle Paddy Doctor folder: train.csv (image_id, label, variety, age) and train_images/<label>/*.jpg. Point PADDY_DATA_DIR at your Downloads copy, or prepare a public subset for this environment.",
    ),
    (
        "3. Exploratory analysis",
        "Count images per class, check variety and plant age, and inspect a sample grid. This shows the imbalance (blast, hispa, and normal dominate; panicle blight is rare).",
    ),
    (
        "4. Preprocessing",
        "Resize to 224×224, ImageNet normalize, and apply flips, crops, rotation, and color jitter only on the training split.",
    ),
    (
        "5. Train / validation split",
        "Hold out 20% of images with a stratified split so every disease stays in both sets.",
    ),
    (
        "6. Model",
        "Fine-tune MobileNetV3-Small as the CPU baseline, then YOLO11n-cls on the same split. YOLO11s-cls is optional if you want a larger classifier. Detection YOLO is not used: this dataset has image-level labels, not lesion boxes.",
    ),
    (
        "7. Training",
        "Warm up the classifier head with a frozen backbone, then unfreeze. AdamW, label smoothing, class-balanced sampling, cosine decay, keep best validation macro F1.",
    ),
    (
        "8. Evaluation",
        "Report accuracy, per-class F1, a confusion matrix, and a table of validation mistakes.",
    ),
    (
        "9. Inference app",
        "Upload a leaf image, see the predicted disease, Grad-CAM overlay, confidence, and a short field note.",
    ),
    (
        "10. Full-dataset rerun",
        "When the complete local folder is available, set PADDY_DATA_DIR and retrain with more epochs for production-quality accuracy.",
    ),
]


@st.cache_resource
def cached_backend(key: str):
    return load_backend(key)


def _fmt_label(name: str) -> str:
    return name.replace("_", " ")


def page_roadmap() -> None:
    st.title("Paddy disease classification")
    st.markdown(
        "A computer-vision project that reads a rice-leaf photograph and names "
        "the disease. The steps below are the full build sequence used in this repo."
    )
    for title, body in ROADMAP:
        st.markdown(f"**{title}**  \n{body}")
    st.info(
        "Your Windows folder `Downloads/paddy-disease-classification` is the "
        "intended full dataset. This workspace cannot see that path, so training "
        "uses a public subset. Copy the folder next to the repo or set "
        "`PADDY_DATA_DIR` to train on every image."
    )


def page_explore() -> None:
    st.title("Dataset")
    if not TRAIN_CSV.exists():
        st.warning(
            "No dataset on disk yet. From the project root run "
            "`python -m src.prepare_data` then `python -m src.train`."
        )
        return
    try:
        df = load_metadata()
    except Exception as exc:
        st.error(str(exc))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Images found", f"{len(df):,}")
    c2.metric("Classes", df["label"].nunique())
    if "variety" in df.columns:
        c3.metric("Varieties", df["variety"].nunique())
    else:
        c3.metric("Images missing files", 0)

    left, right = st.columns(2)
    with left:
        if CLASS_DIST_PATH.exists():
            st.image(str(CLASS_DIST_PATH), caption="Class counts")
        else:
            st.bar_chart(df["label"].value_counts())
    with right:
        if SAMPLES_PATH.exists():
            st.image(str(SAMPLES_PATH), caption="One example per class")
        if "age" in df.columns:
            st.markdown("**Plant age (days)**")
            st.dataframe(
                pd.to_numeric(df["age"], errors="coerce")
                .describe()
                .to_frame("age")
                .T,
                hide_index=True,
            )

    st.subheader("What each label means")
    rows = [
        {
            "class": _fmt_label(name),
            "cause": info["cause"],
            "field signs": info["signs"],
            "suggested next step": info["action"],
        }
        for name, info in CLASS_INFO.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Browse a class")
    choice = st.selectbox("Class", CLASSES, format_func=_fmt_label)
    subset = df[df["label"] == choice]
    st.caption(f"{len(subset)} images")
    cols = st.columns(5)
    for i, (_, row) in enumerate(subset.head(10).iterrows()):
        path = Path(row["path"])
        if path.exists():
            cols[i % 5].image(str(path), caption=row["image_id"], use_container_width=True)


def page_metrics() -> None:
    st.title("Model performance")
    if COMPARISON_PATH.exists():
        st.subheader("MobileNet vs YOLO")
        st.dataframe(pd.read_json(COMPARISON_PATH), hide_index=True, use_container_width=True)
        st.caption(
            "Same stratified 80/20 split. Scores are on the current subset, not the full Kaggle set."
        )
    spec = MODEL_REGISTRY[st.session_state["backend"]]
    metrics_path = spec["metrics"]
    if not metrics_path.exists():
        st.warning(f"No metrics for {spec['title']}. Train it first.")
        return
    metrics = json.loads(metrics_path.read_text())
    st.subheader(spec["title"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Validation accuracy", f"{metrics['val_accuracy'] * 100:.1f}%")
    c2.metric("Macro F1", f"{metrics['val_macro_f1']:.3f}")
    c3.metric("Val images", metrics.get("n_val", "—"))
    c4.metric("Mistakes", metrics.get("val_errors", "—"))
    cols = st.columns(2)
    history = spec.get("history")
    if history and Path(history).exists():
        cols[0].image(str(history), caption="Loss and accuracy by epoch")
    cm = spec.get("cm")
    if cm and Path(cm).exists():
        cols[1].image(str(cm), caption="Confusion matrix")
    per_class = metrics.get("classification_report", {})
    table = []
    for name in CLASSES:
        row = per_class.get(name)
        if not row:
            continue
        table.append(
            {
                "class": _fmt_label(name),
                "precision": round(row.get("precision", 0), 3),
                "recall": round(row.get("recall", 0), 3),
                "f1": round(row.get("f1-score", 0), 3),
                "support": int(row.get("support", 0)),
            }
        )
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)


def page_errors() -> None:
    st.title(f"Validation mistakes · {MODEL_REGISTRY[st.session_state['backend']]['title']}")
    st.markdown(
        "Images the model got wrong on the held-out split. Use this to see "
        "which diseases are confused (for example blight vs streak)."
    )
    errors_path = MODEL_REGISTRY[st.session_state["backend"]]["errors"]
    if not errors_path.exists():
        st.warning("No error table yet. Train this backend first.")
        return
    frame = pd.read_csv(errors_path)
    mistakes = frame.loc[~frame["correct"]].copy()
    c1, c2 = st.columns(2)
    c1.metric("Checked", len(frame))
    c2.metric("Wrong", len(mistakes))
    if mistakes.empty:
        st.success("No mistakes on the validation split.")
        return
    st.dataframe(
        mistakes.drop(columns=["correct"]),
        hide_index=True,
        use_container_width=True,
    )
    try:
        meta = load_metadata()
        paths = dict(zip(meta["image_id"].astype(str), meta["path"]))
    except Exception:
        paths = {}
    show = mistakes.head(8)
    cols = st.columns(4)
    for i, (_, row) in enumerate(show.iterrows()):
        path = paths.get(str(row["image_id"]))
        box = cols[i % 4]
        if path and Path(path).exists():
            box.image(path, use_container_width=True)
        box.caption(
            f"true: {_fmt_label(row['true_label'])} · "
            f"pred: {_fmt_label(row['predicted'])}"
        )


def _resolve_image() -> Image.Image | None:
    uploaded = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png"])
    demo_paths = []
    if TRAIN_IMAGES.exists():
        for label in CLASSES:
            hits = sorted((TRAIN_IMAGES / label).glob("*.jpg"))
            if hits:
                demo_paths.append(hits[min(5, len(hits) - 1)])
    demo_names = ["(choose a sample)"] + [p.parent.name for p in demo_paths]
    sample = st.selectbox("Or try a sample from the training folder", demo_names)
    if uploaded is not None:
        return Image.open(uploaded).convert("RGB")
    if sample != "(choose a sample)":
        return Image.open(demo_paths[demo_names.index(sample) - 1]).convert("RGB")
    return None


def _render_prediction(image: Image.Image, backend: str, col_img, col_out) -> None:
    model, device, spec = cached_backend(backend)
    result = predict_image(image, model=model, device=device, backend=backend)
    heatmap = maybe_heatmap(backend, model, image, CLASS_TO_IDX[result["label"]], device)
    col_img.image(image, caption=f"Input · {spec['title']}", use_container_width=True)
    if heatmap is not None:
        col_img.image(heatmap, caption="Where the model looked (Grad-CAM)", use_container_width=True)
    col_out.success(
        f"**{spec['title']} → {result['title']}**  \n"
        f"Confidence {result['probability'] * 100:.1f}%"
    )
    col_out.markdown(f"**Cause:** {result['cause']}")
    col_out.markdown(f"**What to look for:** {result['signs']}")
    col_out.markdown(f"**Suggested next step:** {result['action']}")
    if result["probability"] < 0.45:
        col_out.warning(
            "Confidence is low. Take another photo in even light, or treat this "
            "as a hint rather than a diagnosis."
        )
    chart = pd.DataFrame(
        {
            "class": [_fmt_label(r["label"]) for r in result["ranking"]],
            "probability": [r["probability"] for r in result["ranking"]],
        }
    ).set_index("class")
    col_out.bar_chart(chart)


def page_classify() -> None:
    st.title("Classify a leaf")
    st.markdown(
        "Upload a close-up of a rice leaf. YOLO11n-cls is the primary YOLO model; "
        "MobileNetV3-Small stays as the baseline. Compare both on the same photo."
    )
    ready = available_backends()
    if not ready:
        st.error("No trained weights yet. Run `python -m src.train` and `python -m src.yolo_train`.")
        return

    image = _resolve_image()
    if image is None:
        st.info("Waiting for an image.")
        return

    backends = ready if st.session_state.get("compare") and len(ready) > 1 else [st.session_state["backend"]]
    for backend in backends:
        col_img, col_out = st.columns([1, 1.2])
        with st.spinner(f"Running {MODEL_REGISTRY[backend]['title']}…"):
            _render_prediction(image, backend, col_img, col_out)
        st.divider()


pages = {
    "Roadmap": page_roadmap,
    "Dataset": page_explore,
    "Metrics": page_metrics,
    "Mistakes": page_errors,
    "Classify": page_classify,
}

with st.sidebar:
    st.header("Paddy Doctor")
    ready = available_backends()
    labels = {key: MODEL_REGISTRY[key]["title"] for key in MODEL_REGISTRY}
    if ready:
        default = "yolo11n-cls" if "yolo11n-cls" in ready else ready[0]
        backend = st.radio(
            "Classifier",
            ready,
            index=ready.index(default),
            format_func=lambda key: labels[key],
        )
        st.session_state["backend"] = backend
        st.session_state["compare"] = st.checkbox(
            "Compare all trained models",
            value=len(ready) > 1,
        )
        st.caption(MODEL_REGISTRY[backend]["why"])
    else:
        st.session_state["backend"] = "mobilenet_v3_small"
        st.session_state["compare"] = False
        st.warning("Train a model to enable inference.")
    choice = st.radio("Page", list(pages))

pages[choice]()
