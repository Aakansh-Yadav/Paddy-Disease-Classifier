import json
from pathlib import Path

root = Path(".").resolve()
(reports := root / "reports").mkdir(exist_ok=True)
(reports / "cv_metrics.json").write_text(r"""
{
  "n_splits": 5,
  "seed": 42,
  "epochs": 12,
  "note": "Stratified 5-fold cross-validation on all labeled images. Each image is held out once. Reported scores are the mean across folds.",
  "models": {
    "mobilenet_v3_small": {
      "title": "MobileNetV3-Small",
      "n_splits": 5,
      "complete": true,
      "mean": {"accuracy": 0.9009324564549759, "precision_macro": 0.893016957679819, "recall_macro": 0.9124571646901074, "f1_macro": 0.9010067003838003, "precision_weighted": 0.9045813702495483, "recall_weighted": 0.9009324564549759, "f1_weighted": 0.9010776921638108},
      "std": {"accuracy": 0.007275967168215576, "precision_macro": 0.004923185322822561, "recall_macro": 0.008643163798130585, "f1_macro": 0.006073328115414369, "precision_weighted": 0.007446226379318896, "recall_weighted": 0.007275967168215576, "f1_weighted": 0.007411618381686486},
      "folds": [
        {"fold": 1, "n_train": 8325, "n_val": 2082, "errors": 215, "accuracy": 0.8967339097022095, "precision_macro": 0.8853152068315708, "recall_macro": 0.9077388201762304, "f1_macro": 0.8947086705979105},
        {"fold": 2, "n_train": 8325, "n_val": 2082, "errors": 206, "accuracy": 0.9010566762728146, "precision_macro": 0.8923478468862618, "recall_macro": 0.9082376813428968, "f1_macro": 0.8989343193498517},
        {"fold": 3, "n_train": 8326, "n_val": 2081, "errors": 225, "accuracy": 0.8918789043728976, "precision_macro": 0.8937250072943655, "recall_macro": 0.9059957386202988, "f1_macro": 0.8981595359244444},
        {"fold": 4, "n_train": 8326, "n_val": 2081, "errors": 200, "accuracy": 0.9038923594425757, "precision_macro": 0.8949345114042601, "recall_macro": 0.9131413882663508, "f1_macro": 0.9025807654663132},
        {"fold": 5, "n_train": 8326, "n_val": 2081, "errors": 185, "accuracy": 0.9111004324843825, "precision_macro": 0.898762215982637, "recall_macro": 0.9271721950447601, "f1_macro": 0.9106502105804818}
      ]
    },
    "yolo11n-cls": {
      "title": "YOLO11n-cls",
      "n_splits": 5,
      "complete": true,
      "mean": {"accuracy": 0.9631982979438412, "precision_macro": 0.9600545675557702, "recall_macro": 0.9585199079907569, "f1_macro": 0.959055142060793, "precision_weighted": 0.9634444401331917, "recall_weighted": 0.9631982979438412, "f1_weighted": 0.9631777800757583},
      "std": {"accuracy": 0.004710883614796614, "precision_macro": 0.006002093401716845, "recall_macro": 0.006376183262752328, "f1_macro": 0.0057410153783638085, "precision_weighted": 0.004684954637099071, "recall_weighted": 0.004710883614796614, "f1_weighted": 0.004704210491836203},
      "folds": [
        {"fold": 1, "n_train": 8325, "n_val": 2082, "errors": 84, "accuracy": 0.9596541786743515, "precision_macro": 0.9546320299267175, "recall_macro": 0.9548636909859095, "f1_macro": 0.9543955182599391},
        {"fold": 2, "n_train": 8325, "n_val": 2082, "errors": 79, "accuracy": 0.9620557156580212, "precision_macro": 0.9542169284430303, "recall_macro": 0.9562987366248054, "f1_macro": 0.9551068477532884},
        {"fold": 3, "n_train": 8326, "n_val": 2081, "errors": 87, "accuracy": 0.9581931763575204, "precision_macro": 0.9593565091220061, "recall_macro": 0.9513649265830258, "f1_macro": 0.9551016415494649},
        {"fold": 4, "n_train": 8326, "n_val": 2081, "errors": 64, "accuracy": 0.9692455550216242, "precision_macro": 0.9681111326433351, "recall_macro": 0.9630104772933168, "f1_macro": 0.9652612751287674},
        {"fold": 5, "n_train": 8326, "n_val": 2081, "errors": 69, "accuracy": 0.9668428640076886, "precision_macro": 0.9639562376437618, "recall_macro": 0.967061708466727, "f1_macro": 0.9654104276125054}
      ]
    },
    "yolo11s-cls": {
      "title": "YOLO11s-cls",
      "n_splits": 5,
      "complete": true,
      "mean": {"accuracy": 0.9710776473108094, "precision_macro": 0.9690469624964482, "recall_macro": 0.969342077292794, "f1_macro": 0.969068534413782, "precision_weighted": 0.9712331581597484, "recall_weighted": 0.9710776473108094, "f1_weighted": 0.9710520085239578},
      "std": {"accuracy": 0.0029823980336272917, "precision_macro": 0.004298020068140111, "recall_macro": 0.004705543393596033, "f1_macro": 0.00430197349536037, "precision_weighted": 0.002998589955811605, "recall_weighted": 0.0029823980336272917, "f1_weighted": 0.0029774005533340763},
      "folds": [
        {"fold": 1, "n_train": 8325, "n_val": 2082, "errors": 71, "accuracy": 0.9658981748318924, "precision_macro": 0.963003253799803, "recall_macro": 0.962294467308932, "f1_macro": 0.9625455212116277},
        {"fold": 2, "n_train": 8325, "n_val": 2082, "errors": 60, "accuracy": 0.9711815561959655, "precision_macro": 0.9666731133383744, "recall_macro": 0.9674108744119263, "f1_macro": 0.9668217394335654},
        {"fold": 3, "n_train": 8326, "n_val": 2081, "errors": 57, "accuracy": 0.9726093224411341, "precision_macro": 0.9716132378379833, "recall_macro": 0.9712470829708824, "f1_macro": 0.9712735785015674},
        {"fold": 4, "n_train": 8326, "n_val": 2081, "errors": 57, "accuracy": 0.9726093224411341, "precision_macro": 0.97397582724734, "recall_macro": 0.9710731870476714, "f1_macro": 0.9724504947284467},
        {"fold": 5, "n_train": 8326, "n_val": 2081, "errors": 56, "accuracy": 0.9730898606439212, "precision_macro": 0.969969380258741, "recall_macro": 0.974684774724558, "f1_macro": 0.9722513381937036}
      ]
    }
  }
}
""".strip() + "\n", encoding="utf-8")

config = root / "src" / "config.py"
text = config.read_text(encoding="utf-8")
if "CV_METRICS_PATH" not in text:
    line = 'CV_METRICS_PATH = REPORTS_DIR / "cv_metrics.json"\n'
    anchor = 'SPLIT_METRICS_PATH = REPORTS_DIR / "split_metrics.json"\n'
    other = 'COMPARISON_PATH = REPORTS_DIR / "model_comparison.json"\n'
    if anchor in text:
        text = text.replace(anchor, anchor + line, 1)
    elif other in text:
        text = text.replace(other, other + line, 1)
    else:
        text += "\n" + line
    config.write_text(text, encoding="utf-8")

app = root / "app.py"
app_text = app.read_text(encoding="utf-8")
if "CV_METRICS_PATH" not in app_text:
    app_text = app_text.replace("SPLIT_METRICS_PATH,", "SPLIT_METRICS_PATH,\n    CV_METRICS_PATH,", 1)
block = """
    if CV_METRICS_PATH.exists():
        cv_payload = json.loads(CV_METRICS_PATH.read_text())
        st.subheader("5-fold cross-validation (mean)")
        cv_rows = []
        fold_rows = []
        for block in cv_payload.get("models", {}).values():
            mean = block.get("mean") or {}
            std = block.get("std") or {}
            if not mean:
                continue
            cv_rows.append({
                "model": block["title"],
                "folds": f"{len(block.get('folds', []))}/{block.get('n_splits', 5)}",
                "accuracy": f"{mean['accuracy']:.3f} ± {std['accuracy']:.3f}",
                "precision_macro": f"{mean['precision_macro']:.3f} ± {std['precision_macro']:.3f}",
                "recall_macro": f"{mean['recall_macro']:.3f} ± {std['recall_macro']:.3f}",
                "f1_macro": f"{mean['f1_macro']:.3f} ± {std['f1_macro']:.3f}",
                "precision_weighted": f"{mean['precision_weighted']:.3f} ± {std['precision_weighted']:.3f}",
                "recall_weighted": f"{mean['recall_weighted']:.3f} ± {std['recall_weighted']:.3f}",
                "f1_weighted": f"{mean['f1_weighted']:.3f} ± {std['f1_weighted']:.3f}",
            })
            for row in block.get("folds", []):
                fold_rows.append({
                    "model": block["title"],
                    "fold": row["fold"],
                    "n_val": row["n_val"],
                    "accuracy": round(row["accuracy"], 4),
                    "precision_macro": round(row["precision_macro"], 4),
                    "recall_macro": round(row["recall_macro"], 4),
                    "f1_macro": round(row["f1_macro"], 4),
                    "errors": row["errors"],
                })
        if cv_rows:
            st.dataframe(pd.DataFrame(cv_rows), hide_index=True, use_container_width=True)
            st.caption(cv_payload.get("note", ""))
        if fold_rows:
            with st.expander("Per-fold scores"):
                st.dataframe(pd.DataFrame(fold_rows), hide_index=True, use_container_width=True)
"""
if "5-fold cross-validation" not in app_text:
    marker = "    if COMPARISON_PATH.exists():"
    if marker not in app_text:
        raise SystemExit("Could not update app.py")
    app_text = app_text.replace(marker, block + "\n" + marker, 1)
app.write_text(app_text, encoding="utf-8")
print("ready", reports / "cv_metrics.json")