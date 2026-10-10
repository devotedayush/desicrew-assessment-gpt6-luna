"""Optional offline macOS Vision recognizer for explicit, single-line image ROIs.

Vision is a local OS framework. This adapter never invokes a hosted OCR API,
but its OCR confidence is not calibrated on the supplied insurance documents.
Every output remains review-required. Linux users can use the base Tesseract
pipeline or the separately guarded local GOT backend.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any


class LocalMacVisionRecognizer:
    def __init__(self, *, max_crop_pixels: int = 1_000_000):
        if sys.platform != "darwin":
            raise RuntimeError("The optional Vision recognizer requires macOS")
        if not 1 <= max_crop_pixels <= 1_000_000:
            raise ValueError("max_crop_pixels must be between 1 and 1000000")
        try:
            import Foundation  # noqa: F401
            import Vision  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install requirements-macos-vision.txt for the optional local Vision backend") from exc
        self.max_crop_pixels = max_crop_pixels

    @staticmethod
    def _read_once(image: Any) -> list[str]:
        import Foundation
        import Vision
        with tempfile.TemporaryDirectory(prefix="desicrew_vision_") as temporary:
            path = Path(temporary) / "line.png"
            image.convert("RGB").save(path)
            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(False)
            handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
                Foundation.NSURL.fileURLWithPath_(str(path)), {})
            success, error = handler.performRequests_error_([request], None)
            if not success:
                raise RuntimeError(f"Local Vision text request failed: {error}")
            observations = request.results() or []
            return [str(observation.topCandidates_(1)[0].string())
                    for observation in observations if observation.topCandidates_(1)]

    def recognize(self, image: Any, *, field: str | None = None,
                  page: int = 1, bbox: list[int] | None = None) -> dict[str, Any]:
        from PIL import Image
        if not isinstance(image, Image.Image):
            raise TypeError("recognize expects a PIL image")
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > self.max_crop_pixels or height > width or height > 500:
            raise ValueError("Expected one bounded horizontal value-line crop")
        candidates = self._read_once(image)
        # An ROI with multiple text lines cannot be called an unambiguous value.
        text = candidates[0] if len(candidates) == 1 else ""
        issues = ["macos_vision_ocr_not_calibrated", "handwritten_form_requires_review"]
        if len(candidates) != 1:
            issues.append("multiple_or_no_text_lines_in_explicit_roi")
        # A bank account number has no useful checksum here. Require a second
        # independent image scale to agree before emitting its unverified digits.
        verification_candidates = []
        if field == "bank_account_number":
            if width * height * 4 <= self.max_crop_pixels:
                larger = image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
                verification_candidates = self._read_once(larger)
            if len(verification_candidates) != 1 or verification_candidates[0].replace(" ", "") != text.replace(" ", ""):
                text = ""
                issues.append("bank_account_ocr_scales_disagree_or_unavailable")
        return {"text": text, "confidence": .35 if text else 0.,
                "confidence_kind": "heuristic_quality_score_not_probability",
                "needs_review": True, "validation_issues": issues,
                "raw_candidates": candidates, "verification_candidates": verification_candidates,
                "terminated_early": False,
                "model": "local_macos_vision_accurate",
                "provenance": {"method": "local_macos_vision_accurate_explicit_crop",
                               "field": field, "page": page, "bbox": bbox,
                               "crop_size": [width, height],
                               "language_correction": False,
                               "external_image_uploads": 0}}
