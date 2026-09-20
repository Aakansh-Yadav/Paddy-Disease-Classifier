from src.config import CLASS_INFO, CLASSES, CLASS_TO_IDX
from src.dataset import image_transforms
from src.model import build_model


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
