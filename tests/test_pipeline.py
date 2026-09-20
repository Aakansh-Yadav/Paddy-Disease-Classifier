from src.config import CLASS_INFO, CLASSES, CLASS_TO_IDX, MODEL_REGISTRY
from src.dataset import image_transforms
from src.model import build_model
from src.predict import ranking_from_scores


def test_ten_classes_and_actions():
    assert len(CLASSES) == 10
    assert CLASS_TO_IDX["normal"] == 8
    for name in CLASSES:
        info = CLASS_INFO[name]
        assert info["title"]
        assert info["action"]


def test_eval_transforms_output_shape():
    from PIL import Image

    image = Image.new("RGB", (640, 480), (40, 120, 50))
    tensor = image_transforms(train=False)(image)
    assert tuple(tensor.shape) == (3, 224, 224)


def test_model_head_matches_classes():
    model = build_model(pretrained=False)
    last = model.classifier[-1]
    assert last.out_features == len(CLASSES)


def test_yolo11n_is_the_default_yolo_backend():
    assert MODEL_REGISTRY["yolo11n-cls"]["kind"] == "yolo"
    assert MODEL_REGISTRY["yolo11n-cls"]["pretrained"] == "yolo11n-cls.pt"
    assert MODEL_REGISTRY["yolo11s-cls"]["kind"] == "yolo"


def test_ranking_puts_highest_score_first():
    ranking = ranking_from_scores({"blast": 0.2, "normal": 0.7, "hispa": 0.1}, topk=2)
    assert ranking[0]["label"] == "normal"
    assert ranking[1]["label"] == "blast"
