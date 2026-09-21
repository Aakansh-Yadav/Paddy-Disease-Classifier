# Paddy disease classification

A computer-vision project that classifies a rice (paddy) leaf photograph into **one of ten labels**: nine diseases plus a healthy leaf. It follows the [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) dataset layout (`train.csv` + `train_images/<label>/*.jpg`).

This README is the project roadmap. The Streamlit app repeats the same steps on the **Roadmap** page.

## Roadmap

1. **Problem definition**  
   Input: a close-up photo of a paddy leaf. Output: a class name and confidence. Classes are `bacterial_leaf_blight`, `bacterial_leaf_streak`, `bacterial_panicle_blight`, `blast`, `brown_spot`, `dead_heart`, `downy_mildew`, `hispa`, `normal`, `tungro`.

2. **Dataset intake**  
   Expected folder (the one on your machine at `Downloads/paddy-disease-classification`):

   ```
   paddy-disease-classification/
     train.csv          # image_id, label, variety, age
     train_images/
       blast/*.jpg
       hispa/*.jpg
       ...
     test_images/       # optional unlabeled photos
   ```

   Point the trainer at it:

   ```bash
   export PADDY_DATA_DIR="/path/to/paddy-disease-classification"
   ```

   In this cloud workspace that Windows path is not visible, so `python -m src.prepare_data --full` downloads the public Hugging Face copy of all ~10,407 train images. `--per-class 80` is only a small demo sample.

3. **Exploratory data analysis**  
   `python -m src.eda` (also runs at the start of training) writes class counts and a sample grid to `reports/`. Use these plots to confirm imbalance and image quality before modeling.

4. **Preprocessing and augmentation**  
   Every image is resized to 224×224 and normalized with ImageNet mean/std. Training-only transforms: random crop, horizontal flip, small rotation, color jitter.

5. **Train / validation split**  
   Stratified 80/20 split so rare classes (for example bacterial panicle blight) still appear in validation.

6. **Model choice**  
   **MobileNetV3-Small** is the CPU baseline. **YOLO11n-cls** is the YOLO model we actually integrate (nano, ImageNet-pretrained, whole-image labels). **YOLO11s-cls** is optional if nano underfits (`python -m src.yolo_train --model yolo11s-cls`). Detection YOLO (`yolo11n.pt`, boxes) is out of scope until lesion boxes exist.

7. **Training loop**  
   Freeze the ImageNet backbone for two epochs while the new head learns, then unfreeze at a lower backbone learning rate. Cross-entropy with label smoothing, AdamW, cosine decay, and a **weighted sampler**. The checkpoint with the best validation **macro F1** is kept. YOLO uses Ultralytics classify training on the **same** stratified split.

8. **Evaluation**  
   Accuracy, per-class precision/recall/F1, training curves, a confusion matrix, and `reports/val_predictions.csv` listing every validation image (including mistakes).

9. **Inference application**  
   Streamlit app (`app.py`): upload a leaf photo, see Grad-CAM (where the network looked), confidence, cause, and a suggested next step. The **Mistakes** page shows held-out errors.

10. **Full-dataset rerun**  
    When you have the complete local Kaggle folder, set `PADDY_DATA_DIR` and train longer (for example `--epochs 15`). That is the step that turns the demo checkpoint into a field-quality model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands

```bash
# 1. Full train set (~10,407 images)
python -m src.prepare_data --full

# 2. Train the MobileNet baseline
python -m src.train --epochs 12

# 2b. Train YOLO11n-cls on the same split (recommended YOLO)
python -m src.yolo_train --model yolo11n-cls --epochs 12

# Optional larger YOLO classifier
# python -m src.yolo_train --model yolo11s-cls --epochs 12

# 3. Recompute metrics and the validation error table
python -m src.evaluate

# 4. Classify one file
python -m src.predict path/to/leaf.jpg --backend yolo11n-cls
python -m src.batch_predict path/to/test_images --backend yolo11n-cls

# 5. Classify a whole folder (test_images or a field dump)
python -m src.batch_predict path/to/test_images -o reports/batch_predictions.csv

# 6. Open the app
streamlit run app.py --server.port 48621

# 7. Unit tests
pytest tests -q
```

Use your full download:

```bash
set PADDY_DATA_DIR=C:\Users\Aakansh\Downloads\paddy-disease-classification
python -m src.train --epochs 15
```

## Project layout

```
app.py                 Streamlit UI (roadmap, dataset, metrics, mistakes, classify)
src/config.py          Paths, class names, hyperparameters
src/prepare_data.py    Local folder check or Hugging Face subset
src/dataset.py         Dataset, stratified split, dataloaders
src/eda.py             Class-count and sample plots
src/model.py           MobileNetV3-Small
src/train.py           Two-phase training loop
src/evaluate.py        Metrics, confusion matrix, val_predictions.csv
src/yolo_data.py       ImageFolder export for Ultralytics
src/yolo_train.py      YOLO11-cls training + comparison JSON
src/explain.py         Grad-CAM heatmaps
src/predict.py         Single-image inference
src/batch_predict.py   Folder inference to CSV
tests/                 Unit tests for classes, transforms, and model head
reports/               Plots and metrics.json
models/                paddy_mobilenetv3.pt after training
```

## Notes

- The demo checkpoint trained on a subset will not match a model trained on all ~10k Kaggle images. Use your Downloads folder for that.
- This is a decision-support tool, not a replacement for a plant pathologist.
