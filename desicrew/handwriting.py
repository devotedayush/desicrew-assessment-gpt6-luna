"""Offline-only, cropped-single-line TrOCR with conservative review metadata.

The optional model is never downloaded by this module. Install optional dependencies
and explicitly obtain the public model first. Scores are heuristics, not calibrated
probabilities that a field is correct; every handwriting result requires review.
"""
from __future__ import annotations

import math
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

MODEL_ID = "microsoft/trocr-small-handwritten"
MODEL_REVISION = "b4648cfa171985a6745f37ddd637e98c0da958ac"
BASE_MODEL_ID = "microsoft/trocr-base-handwritten"
BASE_MODEL_REVISION = "eaacaf452b06415df8f10bb6fad3a4c11e609406"
CONFIDENCE_KIND = "heuristic_quality_score_not_probability"


class LocalHandwritingRecognizer:
    """Recognize one explicitly selected text-line crop, on CPU and offline.

    Token/time/pixel/thread budgets limit work. Transformers' max_time is a soft
    generation limit (it checks after a decoding step), not a hard process timeout.
    Model loading/encoding can exceed that limit. Use a process timeout for a hard
    batch deadline, as the evaluation script does.
    """

    def __init__(self, model_path: str | Path, *, max_new_tokens: int = 64,
                 max_seconds: float = 15, threads: int = 2,
                 max_crop_pixels: int = 1_000_000, preprocessing: str = "none"):
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise ValueError("model_path must be an existing local model directory")
        if not (self.model_path / "model.safetensors").is_file():
            raise ValueError("Offline model requires model.safetensors; unsafe pickle fallback is disabled")
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be an integer between 1 and 128")
        if not isinstance(threads, int) or not 1 <= threads <= 8:
            raise ValueError("threads must be an integer between 1 and 8")
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= 120:
            raise ValueError("max_seconds must be greater than zero and at most 120")
        if not isinstance(max_crop_pixels, int) or not 1 <= max_crop_pixels <= 4_000_000:
            raise ValueError("max_crop_pixels must be between 1 and 4000000")
        if preprocessing not in {"none", "contrast_rule_cleanup"}:
            raise ValueError("preprocessing must be none or contrast_rule_cleanup")
        metadata_path = self.model_path / "DOWNLOAD_MANIFEST.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        self.model_id = metadata.get("model_id", "local_checkpoint_unverified_source")
        self.model_revision = metadata.get("revision")
        self.expected_weights_hash = metadata.get("sha256", {}).get("model.safetensors")
        self.weights_hash_verified = False
        self.preprocessing = preprocessing
        self.max_new_tokens = max_new_tokens
        self.max_seconds = float(max_seconds)
        self.threads = threads
        self.max_crop_pixels = max_crop_pixels
        self._processor = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self.expected_weights_hash:
            with (self.model_path / "model.safetensors").open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != self.expected_weights_hash:
                raise RuntimeError("Local safetensors hash no longer matches the download manifest")
            self.weights_hash_verified = True
        os.environ.setdefault("HF_HOME", str(self.model_path / ".hf-cache"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError("Optional handwriting dependencies are missing; install requirements-handwriting.txt in an isolated environment") from exc
        torch.set_num_threads(self.threads)
        # Deliberately use a local directory, no token, no remote code, safe weights.
        options = dict(local_files_only=True, trust_remote_code=False, token=False)
        self._processor = TrOCRProcessor.from_pretrained(str(self.model_path), use_fast=False, **options)
        self._model = VisionEncoderDecoderModel.from_pretrained(
            str(self.model_path), use_safetensors=True, **options
        ).to("cpu").eval()
        self._torch = torch

    @staticmethod
    def _quality_score(token_score: float, text: str, terminated_early: bool) -> float:
        # Decoding certainty does not measure field correctness or domain shift.
        if not text.strip() or not math.isfinite(token_score):
            return 0.0
        score = min(.72, max(0.0, token_score) * .72)
        if terminated_early:
            score = min(score, .25)
        return round(score, 3)

    @staticmethod
    def _prepare_crop(image: Any, mode: str) -> tuple[Any, dict[str, Any]]:
        from PIL import Image, ImageOps
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        removed = 0
        if mode == "contrast_rule_cleanup":
            import numpy as np
            gray = np.asarray(ImageOps.autocontrast(ImageOps.grayscale(prepared))).copy()
            ink = gray < 110
            for y, row in enumerate(ink):
                changes = np.diff(np.pad(row.astype(np.int8), (1, 1)))
                for start, end in zip(np.where(changes == 1)[0], np.where(changes == -1)[0]):
                    if end - start > gray.shape[1] * .58:
                        gray[y, start:end] = 255
                        removed += int(end - start)
            prepared = Image.fromarray(gray).convert("RGB")
        return prepared, {"mode": mode, "removed_long_rule_pixels": removed,
                          "destructive_cleanup_not_validated": mode != "none"}

    def recognize(self, image: Any, *, field: str | None = None, page: int = 1,
                  bbox: list[int] | None = None) -> dict[str, Any]:
        from PIL import Image, ImageOps
        if not isinstance(image, Image.Image):
            raise TypeError("recognize expects a PIL image of one text-line crop")
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > self.max_crop_pixels:
            raise ValueError("Crop is empty or exceeds the configured pixel budget")
        # A geometric guard rejects obvious full pages. It cannot prove that a
        # wide region contains one line; explicit crop inspection is still needed.
        if height > width or height > 500:
            raise ValueError("Expected one horizontal text-line crop, not a full page")
        if page < 1:
            raise ValueError("page must be positive")
        if bbox is not None and (len(bbox) != 4 or any(not isinstance(x, int) for x in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            raise ValueError("bbox must contain four ordered integer xyxy coordinates")
        self._load()
        started = time.monotonic()
        prepared, preprocessing_info = self._prepare_crop(image, self.preprocessing)
        inputs = self._processor(images=prepared, return_tensors="pt").pixel_values.to("cpu")
        with self._torch.inference_mode():
            generated = self._model.generate(
                inputs, max_new_tokens=self.max_new_tokens, max_time=self.max_seconds,
                do_sample=False, num_beams=1, return_dict_in_generate=True,
                output_scores=True, use_cache=True,
            )
        elapsed = time.monotonic() - started
        text = self._processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
        count = len(generated.scores)
        eos = self._model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        tail = generated.sequences[0, -count:].tolist() if count else []
        ended = bool(tail and tail[-1] in eos_ids)
        truncated = not ended
        token_score = 0.0
        if count:
            pad = self._model.generation_config.pad_token_id
            # Gather selected-token log softmax directly. The Transformers 4.46
            # compute_transition_scores helper assumes a top-level vocab_size,
            # which VisionEncoderDecoderConfig does not expose.
            visible = [float(self._torch.log_softmax(step.float(), dim=-1)[0, token].item())
                       for step, token in zip(generated.scores, tail)
                       if token not in eos_ids and token != pad]
            if visible:
                token_score = math.exp(sum(visible) / len(visible))
        issues = ["handwritten_model_not_calibrated", "handwritten_form_requires_review"]
        if not self.weights_hash_verified:
            issues.append("local_checkpoint_source_not_hash_verified")
        if self.preprocessing != "none":
            issues.append("diagnostic_rule_cleanup_can_damage_touching_handwriting")
        if truncated:
            issues.append("handwriting_generation_incomplete_or_budget_exhausted")
        if not text:
            issues.append("empty_handwriting_model_output")
        return {
            "text": text,
            "confidence": self._quality_score(token_score, text, truncated),
            "confidence_kind": CONFIDENCE_KIND,
            "token_score": round(token_score, 6),
            "token_score_kind": "geometric_mean_decoder_token_softmax_not_field_correctness",
            "token_count": count,
            "model": self.model_id,
            "model_revision": self.model_revision,
            "weights_hash_verified": self.weights_hash_verified,
            "elapsed_seconds": round(elapsed, 3),
            "needs_review": True,
            "validation_issues": issues,
            "terminated_early": truncated,
            "preprocessing": preprocessing_info,
            "provenance": {"method": "local_trocr_single_line", "field": field,
                           "page": page, "bbox": bbox, "crop_size": [width, height]},
        }
