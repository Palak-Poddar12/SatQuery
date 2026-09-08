from rs_llava_ben.evaluate import evaluate
from rs_llava_ben.prepare_data import build_records
from rs_llava_ben.change_detection import encode_mask


def test_build_records_creates_multilabel_and_presence_questions():
    records = list(build_records(iter([{"image": "patch.jpg", "labels": ["Forest", "Water bodies"]}])))
    assert len(records) == 4
    assert records[0]["task"] == "multilabel"
    assert records[0]["answer"] == "This image contains Forest, and Water bodies land cover classes."
    assert next(row for row in records if row.get("category") == "forest")["answer"].startswith("Yes")
    assert next(row for row in records if row.get("category") == "urban")["answer"].startswith("No")


def test_evaluate_returns_oa_and_aa():
    result = evaluate([
        {"true_labels": ["Forest"], "pred_labels": ["Forest"]},
        {"true_labels": ["Water"], "pred_labels": ["Forest"]},
    ])
    assert result["overall_accuracy"] == 0.5
    assert result["average_accuracy"] == 0.5
    assert result["per_class_accuracy"] == {"Forest": 0.5, "Water": 0.5}


def test_change_mask_is_png_base64():
    import base64
    from io import BytesIO

    from PIL import Image

    encoded = encode_mask(Image.new("L", (2, 2), 255))
    decoded = Image.open(BytesIO(base64.b64decode(encoded)))
    assert decoded.format == "PNG"
    assert decoded.size == (2, 2)
