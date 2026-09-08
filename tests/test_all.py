import base64
import io

from PIL import Image


def make_test_image() -> str:
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_vqa():
    from models.vqa import run_vqa

    result = run_vqa(make_test_image(), "What is visible?")
    assert "answer" in result
    assert "confidence" in result
    assert result["task_type"] == "vqa"
    print("VQA test passed!")


def test_caption():
    from models.captioning import run_caption

    result = run_caption(make_test_image(), "brief")
    assert "caption" in result
    assert result["task_type"] == "caption"
    print("Caption test passed!")


def test_grounding():
    from models.grounding import run_grounding

    result = run_grounding(make_test_image(), "water body")
    assert "bboxes" in result
    assert result["task_type"] == "grounding"
    print("Grounding test passed!")


def test_change():
    from models.change_detection import run_change_vqa

    img = make_test_image()
    result = run_change_vqa(img, img, "What changed?")
    assert "change_description" in result
    assert result["task_type"] == "change_vqa"
    print("Change detection test passed!")


def test_sar():
    from models.sar_fusion import run_sar_fusion

    img = make_test_image()
    result = run_sar_fusion(img, img, "What is visible?")
    assert "answer" in result
    assert result["task_type"] == "sar_fusion"
    print("SAR fusion test passed!")


if __name__ == "__main__":
    test_vqa()
    test_caption()
    test_grounding()
    test_change()
    test_sar()
    print("ALL TESTS PASSED!")
