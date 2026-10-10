"""Experimental native GOT-OCR2, CPU-only and offline, for inspected line crops.

No image is transmitted and this module never downloads a model or runs model
repository code. Every prediction requires human review. This alternative is not
automatically selected by the application.
"""
from __future__ import annotations

import hashlib
import ctypes
import gc
import json
import math
import os
from pathlib import Path
import time
import sys
from typing import Any

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
MODEL_REVISION = "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
WEIGHTS_SHA256 = "6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4"
CONFIDENCE_KIND = "heuristic_quality_score_not_probability"


def _current_rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    except ImportError:
        return None


def release_cpu_allocator() -> dict[str, Any]:
    """Collect unreachable objects and return already-free glibc pages only.

    This does not touch live model tensors or change model math. malloc_trim is a
    standard GNU libc function; unsupported platforms simply skip that step.
    """
    before = _current_rss_mb()
    collected = gc.collect()
    available = False
    trimmed = False
    if sys.platform.startswith("linux"):
        try:
            trim = ctypes.CDLL(None).malloc_trim
            trim.argtypes = [ctypes.c_size_t]
            trim.restype = ctypes.c_int
            available = True
            trimmed = bool(trim(0))
        except (AttributeError, OSError):
            pass
    return {"garbage_objects_collected": collected, "malloc_trim_available": available,
            "malloc_trim_released_free_pages": trimmed,
            "rss_before_cleanup_mb": before, "rss_after_cleanup_mb": _current_rss_mb()}


class LocalGotOcrRecognizer:
    """One bounded alternative checkpoint, never an unattended acceptance path."""

    def __init__(self, model_path: str | Path, *, max_new_tokens: int = 64,
                 max_seconds: float = 20, threads: int = 2,
                 max_crop_pixels: int = 1_000_000):
        self.model_path = Path(model_path)
        if not self.model_path.is_dir() or not (self.model_path / "model.safetensors").is_file():
            raise ValueError("An existing local safetensors model directory is required")
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        if not isinstance(threads, int) or not 1 <= threads <= 2:
            raise ValueError("This diagnostic allows only one or two CPU threads")
        if not math.isfinite(max_seconds) or not 0 < max_seconds <= 60:
            raise ValueError("max_seconds must be positive and at most 60")
        if not isinstance(max_crop_pixels, int) or not 1 <= max_crop_pixels <= 1_000_000:
            raise ValueError("max_crop_pixels must be between 1 and 1000000")
        self.max_new_tokens = max_new_tokens
        self.max_seconds = float(max_seconds)
        self.threads = threads
        self.max_crop_pixels = max_crop_pixels
        self.weights_hash_verified = False
        self._processor = self._model = self._torch = None

    def _verify_files(self) -> None:
        metadata_path = self.model_path / "DOWNLOAD_MANIFEST.json"
        if not metadata_path.is_file():
            raise RuntimeError("A verified pinned public download manifest is required")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("model_id") != MODEL_ID or metadata.get("revision") != MODEL_REVISION:
            raise RuntimeError("Only the explicitly approved official pinned checkpoint is supported")
        expected = metadata.get("sha256", {})
        if expected.get("model.safetensors") != WEIGHTS_SHA256:
            raise RuntimeError("Manifest does not record the official pinned weights hash")
        for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                     "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            path = self.model_path / name
            if not path.is_file() or name not in expected:
                raise RuntimeError(f"Missing verified local model file: {name}")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected[name]:
                raise RuntimeError(f"Local model file hash mismatch: {name}")
        self.weights_hash_verified = True

    def _load(self) -> None:
        if self._model is not None:
            return
        self._verify_files()
        os.environ.setdefault("HF_HOME", str(self.model_path / ".hf-cache"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        try:
            import torch
            from transformers import AutoProcessor, GotOcr2ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install requirements-got-ocr.txt in its separate isolated environment") from exc
        torch.set_num_threads(self.threads)
        options = dict(local_files_only=True, trust_remote_code=False, token=False)
        self._processor = AutoProcessor.from_pretrained(str(self.model_path), use_fast=False, **options)
        self._model = GotOcr2ForConditionalGeneration.from_pretrained(
            str(self.model_path), use_safetensors=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, attn_implementation="sdpa", **options,
        ).to("cpu").eval()
        self._torch = torch

    def recognize(self, image: Any, *, field: str | None = None,
                  page: int = 1, bbox: list[int] | None = None) -> dict[str, Any]:
        from PIL import Image, ImageOps
        if not isinstance(image, Image.Image):
            raise TypeError("recognize expects an inspected PIL value-line crop")
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > self.max_crop_pixels:
            raise ValueError("Crop exceeds the pixel budget or is empty")
        if height > width or height > 500:
            raise ValueError("Expected an inspected horizontal line crop, not a whole page")
        if not isinstance(page, int) or page < 1:
            raise ValueError("page must be a positive integer")
        if bbox is not None and (len(bbox) != 4 or any(not isinstance(x, int) for x in bbox)
                                 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            raise ValueError("bbox must contain four ordered integer xyxy coordinates")
        self._load()
        started = time.monotonic()
        rss_before_crop = _current_rss_mb()
        prepared = inputs = generated = output_ids = None
        try:
            prepared = ImageOps.exif_transpose(image).convert("RGB")
            inputs = self._processor(images=prepared, return_tensors="pt")
            # Cast only image tensors. Token ids retain their integer dtype.
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._torch.bfloat16, device="cpu")
            prompt_length = inputs["input_ids"].shape[1]
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, max_time=self.max_seconds,
                    do_sample=False, num_beams=1, tokenizer=self._processor.tokenizer,
                    stop_strings="<|im_end|>", return_dict_in_generate=False,
                    output_scores=False, output_logits=False, output_attentions=False,
                    output_hidden_states=False, use_cache=True,
                )
            # A plain token tensor avoids returning/preserving generation KV
            # cache metadata. Scores/logits are not needed for this heuristic.
            output_ids = generated[0, prompt_length:]
            text = self._processor.decode(output_ids, skip_special_tokens=True).strip()
            token_count = int(output_ids.numel())
            eos = self._model.generation_config.eos_token_id
            eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
            im_end_id = self._processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(im_end_id, int) and im_end_id >= 0:
                eos_ids.add(im_end_id)
            ended = bool(token_count and output_ids[-1].item() in eos_ids)
        finally:
            # Release all per-crop tensor/PIL references BEFORE collecting and
            # trimming, including when decoding or inference raises an error.
            prepared = inputs = generated = output_ids = None
            cleanup = release_cpu_allocator()
        truncated = not ended
        elapsed = time.monotonic() - started
        issues = ["experimental_general_ocr_model_not_calibrated", "handwritten_form_requires_review"]
        if truncated:
            issues.append("generation_incomplete_or_budget_exhausted")
        if not text:
            issues.append("empty_general_ocr_output")
        return {
            "text": text, "confidence": .35 if text and ended else (.15 if text else 0.),
            "confidence_kind": CONFIDENCE_KIND, "needs_review": True,
            "validation_issues": issues, "token_count": token_count,
            "terminated_early": truncated, "elapsed_seconds": round(elapsed, 3),
            "model": MODEL_ID, "model_revision": MODEL_REVISION,
            "weights_hash_verified": self.weights_hash_verified,
            "device": "cpu", "dtype": "bfloat16", "cpu_threads": self.threads,
            "memory_management": {"rss_before_crop_mb": rss_before_crop,
                                  "generation_metadata_returned": False,
                                  "decoder_scores_or_logits_retained": False, **cleanup},
            "provenance": {"method": "experimental_local_got_ocr2_inspected_line",
                           "field": field, "page": page, "bbox": bbox,
                           "crop_size": [width, height], "preprocessing": "RGB_only"},
        }
