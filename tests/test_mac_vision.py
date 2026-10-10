"""Contract checks for the optional local Vision adapter; no OS OCR is invoked."""
from unittest.mock import patch

from PIL import Image

from desicrew.mac_vision import LocalMacVisionRecognizer


def test_bank_account_requires_two_scale_agreement():
    recognizer = LocalMacVisionRecognizer.__new__(LocalMacVisionRecognizer)
    recognizer.max_crop_pixels = 1_000_000
    crop = Image.new("RGB", (200, 40), "white")
    with patch.object(recognizer, "_read_once", side_effect=[["123456789"], ["1234567891"]]):
        result = recognizer.recognize(crop, field="bank_account_number")
    assert result["text"] == ""
    assert result["raw_candidates"] == ["123456789"]
    assert result["verification_candidates"] == ["1234567891"]
    assert "bank_account_ocr_scales_disagree_or_unavailable" in result["validation_issues"]

    with patch.object(recognizer, "_read_once", side_effect=[["123456789"], ["123456789"]]):
        agreed = recognizer.recognize(crop, field="bank_account_number")
    assert agreed["text"] == "123456789"
    assert agreed["needs_review"]
    assert agreed["confidence"] == .35
