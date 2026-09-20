from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    __import__("os").environ.get(
        "PADDY_DATA_DIR", ROOT / "data" / "paddy-disease-classification"
    )
)
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_IMAGES = DATA_DIR / "train_images"
TEST_IMAGES = DATA_DIR / "test_images"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

CLASSES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

CLASS_INFO = {
    "bacterial_leaf_blight": {
        "title": "Bacterial leaf blight",
        "cause": "Xanthomonas oryzae pv. oryzae",
        "signs": "Yellow-to-white lesions along leaf edges that later dry and gray.",
        "action": "Use certified seed, avoid field-to-field flood water, and ask an agronomist about copper or recommended bactericides.",
    },
    "bacterial_leaf_streak": {
        "title": "Bacterial leaf streak",
        "cause": "Xanthomonas oryzae pv. oryzicola",
        "signs": "Narrow, water-soaked streaks between leaf veins.",
        "action": "Reduce dense planting, keep nitrogen moderate, and remove volunteer rice around the field.",
    },
    "bacterial_panicle_blight": {
        "title": "Bacterial panicle blight",
        "cause": "Burkholderia glumae",
        "signs": "Florets on the panicle brown and fail to fill grain.",
        "action": "Avoid excess nitrogen at heading and inspect panicles early in warm, humid weather.",
    },
    "blast": {
        "title": "Rice blast",
        "cause": "Magnaporthe oryzae",
        "signs": "Diamond-shaped leaf spots with gray centers and brown borders.",
        "action": "Choose a blast-tolerant variety if possible and time fungicide with an extension officer.",
    },
    "brown_spot": {
        "title": "Brown spot",
        "cause": "Bipolaris oryzae",
        "signs": "Circular brown lesions, often on older leaves, linked to nutrient stress.",
        "action": "Check soil fertility, especially potassium, and improve drainage.",
    },
    "dead_heart": {
        "title": "Dead heart",
        "cause": "Stem borer damage",
        "signs": "Central tiller dries and pulls out easily; young plants look wilted.",
        "action": "Scout for stem borers, keep the field weed-free, and follow local IPM thresholds.",
    },
    "downy_mildew": {
        "title": "Downy mildew",
        "cause": "Sclerophthora macrospora",
        "signs": "Pale, twisted young leaves and stunted growth.",
        "action": "Rogue severely stunted plants and avoid planting in poorly drained nursery beds.",
    },
    "hispa": {
        "title": "Rice hispa",
        "cause": "Dicladispa armigera feeding",
        "signs": "White scraping streaks and holes from beetle larvae and adults.",
        "action": "Clip and destroy heavily mined leaves; use insecticide only if the local threshold is crossed.",
    },
    "normal": {
        "title": "Healthy leaf",
        "cause": "No disease",
        "signs": "Uniform green leaf with no lesions, streaks, or dead tissue.",
        "action": "Keep routine scouting. No disease treatment is indicated from this photo.",
    },
    "tungro": {
        "title": "Rice tungro",
        "cause": "Tungro virus complex, spread by leafhoppers",
        "signs": "Stunting, yellow-orange leaves, and delayed flowering.",
        "action": "Control green leafhoppers in the nursery and avoid staggered planting in the same block.",
    },
}

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2
VAL_FRACTION = 0.2
SEED = 42
EPOCHS = 12
FREEZE_EPOCHS = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
PRETRAINED = True
BACKBONE = "mobilenet_v3_small"
CHECKPOINT = MODELS_DIR / "paddy_mobilenetv3.pt"
METRICS_PATH = REPORTS_DIR / "metrics.json"
CM_PATH = REPORTS_DIR / "confusion_matrix.png"
HISTORY_PATH = REPORTS_DIR / "training_history.png"
CLASS_DIST_PATH = REPORTS_DIR / "class_distribution.png"
SAMPLES_PATH = REPORTS_DIR / "sample_grid.png"
ERRORS_PATH = REPORTS_DIR / "val_predictions.csv"

SUBSET_PER_CLASS = 80
