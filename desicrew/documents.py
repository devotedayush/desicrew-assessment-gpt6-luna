"""Local document classification, extraction and explicitly conservative review flags.

No network client or external OCR service is used. Confidence values are heuristic
quality scores, not calibrated correctness probabilities. See docs/OCR_METHOD.md.
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DOCUMENT_SCHEMAS = {
    "aadhaar_card": ("aadhaar_number", "full_name", "date_of_birth", "address"),
    "pan_card": ("pan_number", "full_name", "father_name", "date_of_birth"),
    "driving_licence": ("dl_number", "name", "date_of_issue", "valid_till_date"),
    "passport": ("passport_number", "date_of_birth", "date_of_expiry", "mrz_line_2"),
    "nach_ecs_mandate": ("bank_account_number", "ifsc_code", "bank_name", "amount_figures", "frequency"),
    "fatca_annexure_form": ("policy_number", "tin_pan", "father_name", "place_of_birth", "nationality"),
    "benefit_illustration_declaration": ("application_number", "policyholder_name", "date", "place"),
    "moral_hazard_questionnaire": ("application_number", "name_of_life_assured", "nominee_relationship", "date", "place"),
    "multiple_policies_consent_form": ("proposer_name", "reason_for_multiple_policies", "date", "place"),
    "suitability_profiler_declaration": ("application_number", "name_of_life_assured", "name_of_agent_sp", "date", "place"),
}
DOCUMENT_LABELS = {
    "aadhaar_card": "Aadhaar Card", "pan_card": "PAN Card",
    "driving_licence": "Driving Licence", "passport": "Passport",
    "nach_ecs_mandate": "NACH / ECS Mandate",
    "fatca_annexure_form": "FATCA Annexure Form",
    "benefit_illustration_declaration": "Benefit Illustration Declaration",
    "moral_hazard_questionnaire": "Moral Hazard Questionnaire",
    "multiple_policies_consent_form": "Multiple Policies Consent Form",
    "suitability_profiler_declaration": "Suitability Profiler Declaration",
}
HANDWRITTEN_TYPES = frozenset(list(DOCUMENT_SCHEMAS)[4:])
ALIASES = {
    "aadhaar_number": ("aadhaar number", "aadhar number", "aadhaar no", "aadhar no", "uid number", "uid"),
    "full_name": ("full name", "name"),
    "name": ("full name", "name"),
    "date_of_birth": ("date of birth", "birth date", "dob", "d.o.b", "d.o.b."),
    "address": ("address", "residential address", "permanent address"),
    "pan_number": ("permanent account number", "pan number", "pan no", "pan"),
    "father_name": ("father's name", "fathers name", "father name", "name of father"),
    "dl_number": ("driving licence number", "driving license number", "licence number", "license number", "dl number", "dl no", "d.l. no"),
    "date_of_issue": ("date of issue", "issue date", "issued on", "doi"),
    "valid_till_date": ("valid till date", "valid till", "valid upto", "valid up to", "validity", "expiry date"),
    "passport_number": ("passport number", "passport no", "passport no."),
    "date_of_expiry": ("date of expiry", "expiry date", "expiration date"),
    "mrz_line_2": ("mrz line 2", "mrz line ii"),
    "bank_account_number": ("bank account number", "bank a/c number", "account number", "account no", "a/c no", "a/c number"),
    "ifsc_code": ("ifsc code", "ifsc"),
    "bank_name": ("bank name", "name of bank", "with bank", "bank"),
    "amount_figures": ("amount (figures)", "amount in figures", "amount figures", "amount", "rupees"),
    "frequency": ("frequency", "periodicity"),
    "policy_number": ("policy number", "policy no"),
    "tin_pan": ("tin / pan", "tin/pan", "tin or pan", "tax identification number", "tin", "pan number", "pan"),
    "place_of_birth": ("place of birth", "birth place", "birthplace"),
    "nationality": ("nationality", "citizenship"),
    "application_number": ("application/proposal form number", "application / proposal form number", "application number", "application no", "proposal number", "proposal no"),
    "policyholder_name": ("policyholder name", "policy holder name", "name of policyholder", "name of policy holder"),
    "name_of_life_assured": ("name of life assured", "name of the life to be assured", "life assured name", "life assured"),
    "nominee_relationship": ("nominee relationship", "relationship with nominee", "relationship of nominee", "relationship"),
    "proposer_name": ("proposer/life assured name", "proposer / life assured name", "proposer name", "name of proposer"),
    "reason_for_multiple_policies": ("reason for multiple policies", "reason for buying multiple policies", "reasons for multiple policies", "reason"),
    "name_of_agent_sp": ("name of agent/sp", "name of agent / sp", "agent/sp name", "agent name", "name of agent", "name of sp"),
    "date": ("date", "dated"), "place": ("place",),
}
CLASSIFY_RULES = {
    "aadhaar_card": ((r"\b(?:aadhaar|aadhar|unique identification authority)\b", 5), (r"\bgovernment of india\b", 1)),
    "pan_card": ((r"\bpermanent account number\b", 5), (r"\bincome tax department\b", 5), (r"\bpan card\b", 5)),
    "driving_licence": ((r"\bdriving licen[cs]e\b", 5), (r"\b(?:transport department|motor vehicles)\b", 1)),
    "passport": ((r"\bpassport\b", 5), (r"\brepublic of india\b", 1), (r"(?m)^P<[A-Z<]{20,}", 6)),
    "nach_ecs_mandate": ((r"\bnach\b|\becs\b", 5), (r"\bmandate\b", 3), (r"\bifsc\b", 1)),
    "fatca_annexure_form": ((r"\bfatca\b", 6), (r"\b(?:foreign account tax|tax residency)\b", 2)),
    "benefit_illustration_declaration": ((r"\bbenefit illustration\b", 6), (r"\bdeclaration\b", 1)),
    "moral_hazard_questionnaire": ((r"\bm?oral hazard\b", 6), (r"\bquestionnaire\b", 1)),
    "multiple_policies_consent_form": ((r"\bmultiple polic(?:y|ies)\b", 6), (r"\bconsent\b", 1)),
    "suitability_profiler_declaration": ((r"\bsuitability profil(?:er|ing)\b", 6), (r"\bdeclaration\b", 1)),
}
DATE_FIELDS = frozenset(("date", "date_of_birth", "date_of_issue", "valid_till_date", "date_of_expiry"))
IDENTIFIER_FIELDS = frozenset(("aadhaar_number", "pan_number", "dl_number", "passport_number", "bank_account_number", "ifsc_code", "policy_number", "tin_pan", "application_number"))
HIGH_RISK_HANDWRITTEN = frozenset(("ifsc_code", "tin_pan", "bank_account_number", "place", "place_of_birth")) | DATE_FIELDS


@dataclass
class _Line:
    text: str
    page: int = 1
    confidence: float = 1.0
    bbox: list[int] | None = None
    index: int = 0
    contributing_lines: list[_Line] | None = None


def _combine_lines(lines: Iterable[_Line]) -> _Line:
    """Keep every source line and never let a weak continuation disappear."""
    flattened = []
    seen = set()
    for line in lines:
        for part in line.contributing_lines or [line]:
            if id(part) not in seen:
                flattened.append(part)
                seen.add(id(part))
    first = flattened[0]
    boxes = [line.bbox for line in flattened if line.bbox is not None]
    bbox = None
    if len(boxes) == len(flattened) and len({line.page for line in flattened}) == 1:
        bbox = [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]
    return _Line("\n".join(line.text for line in flattened), first.page, min(line.confidence for line in flattened), bbox, first.index, flattened)


def _source_evidence(line: _Line, raw: str, method: str) -> dict[str, Any]:
    parts = line.contributing_lines or [line]
    return {"page": line.page, "line": line.index, "text": line.text, "candidate_text": raw, "method": method, "bbox": line.bbox, "ocr_confidence": round(line.confidence, 3), "lines": [{"page": part.page, "line": part.index, "text": part.text, "bbox": part.bbox, "ocr_confidence": round(part.confidence, 3)} for part in parts]}


def _refresh_field_review_reasons(result: dict[str, Any]) -> dict[str, Any]:
    fields = result.get("fields", {})
    retained = [reason for reason in result.get("review_reasons", []) if reason.split(":", 1)[0] not in fields]
    retained.extend(f"{field}:{item['status']}" for field, item in fields.items() if item.get("needs_review"))
    result["review_reasons"] = list(dict.fromkeys(retained))
    return result


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t:;|_=")


def _label_re(alias: str) -> str:
    # Literal spaces vary after OCR; slash-separated labels may gain spaces.
    result = re.escape(alias).replace(r"\ ", r"\s*").replace("/", r"\s*/\s*").replace("'", "['’‘]")
    return result + (r"\.?" if alias.endswith(" no") else "")


def _checksum_mrz(text: str) -> int:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return sum((0 if ch == "<" else alphabet.index(ch)) * (7, 3, 1)[i % 3] for i, ch in enumerate(text)) % 10


def _valid_date(raw: str) -> tuple[str | None, list[str]]:
    value = raw.strip()
    value = re.sub(r"\s*([/.-])\s*", r"\1", value)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt.endswith("%y"):
                return None, ["two_digit_year_century_ambiguous"]
            if parsed.year < 1900 or parsed.year > 2150:
                return None, ["date_year_out_of_supported_range"]
            issues = []
            if fmt.startswith("%d") and parsed.day <= 12 and parsed.month <= 12 and parsed.day != parsed.month:
                issues.append("numeric_date_assumed_day_first")
            return parsed.strftime("%Y-%m-%d"), issues
        except ValueError:
            pass
    return None, ["invalid_or_ambiguous_date"]


class DocumentProcessor:
    """Classify and extract the 44 brief targets, always returning missing fields.

    `process_text` is for supplied text/OCR transcripts and reproducible tests, not
    evidence of image OCR accuracy. `document_type` is an optional human hint; it
    does not turn uncertain OCR or extracted values into high-confidence values.
    """

    def __init__(self, review_threshold: float = .90, *, tesseract_command: str | None = None, language: str = "eng", timeout: float = 45, max_pages: int = 20, max_pixels: int = 40_000_000, max_total_pixels: int = 60_000_000, handwriting_backend: Any = None, roi_manifest_path: str | Path | None = None, region_templates: dict[str, Any] | None = None, use_printed_region_ocr: bool = False, use_handwritten_region_ocr: bool = False, region_preprocessing: str = "none", use_checkbox_detection: bool = False):
        if not 0 < review_threshold <= 1:
            raise ValueError("review_threshold must be greater than 0 and at most 1")
        self.review_threshold = float(review_threshold)
        self.tesseract_command = tesseract_command or shutil.which("tesseract") or "/usr/bin/tesseract"
        self.language = language
        self.timeout = timeout
        self.max_pages = max_pages
        self.max_pixels = max_pixels
        self.max_total_pixels = max_total_pixels
        if min(max_pages, max_pixels, max_total_pixels, timeout) <= 0:
            raise ValueError("Page, pixel and timeout limits must be positive")
        self.handwriting_backend = handwriting_backend
        self.region_templates = {self._type_hint(key): value for key, value in (region_templates or {}).items()}
        self.use_printed_region_ocr = use_printed_region_ocr
        self.use_handwritten_region_ocr = use_handwritten_region_ocr
        if region_preprocessing not in {"none", "contrast_rule_cleanup"}:
            raise ValueError("region_preprocessing must be none or contrast_rule_cleanup")
        self.region_preprocessing = region_preprocessing
        self.use_checkbox_detection = use_checkbox_detection
        self.roi_documents = []
        if roi_manifest_path is not None:
            manifest = json.loads(Path(roi_manifest_path).read_text(encoding="utf-8"))
            self.roi_documents = [{**entry, "document_type": self._type_hint(entry.get("document_type"))} for entry in manifest.get("documents", [])]
            if manifest.get("coordinate_space") != "original_image_pixels":
                raise ValueError("ROI manifest coordinates must use original_image_pixels")

    def _type_hint(self, document_type: str | None) -> str | None:
        if document_type is None:
            return None
        normalized = re.sub(r"[^a-z0-9]+", "_", document_type.lower()).strip("_")
        candidates = {**{key: key for key in DOCUMENT_SCHEMAS}, **{re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"): key for key, name in DOCUMENT_LABELS.items()}}
        candidates.update({"aadhaar": "aadhaar_card", "pan": "pan_card", "nach": "nach_ecs_mandate", "fatca": "fatca_annexure_form", "driving_license": "driving_licence"})
        if normalized not in candidates:
            raise ValueError(f"Unsupported document_type {document_type!r}; choose one of {list(DOCUMENT_SCHEMAS)}")
        return candidates[normalized]

    def classify(self, text: str) -> tuple[str | None, float, list[dict[str, Any]]]:
        # Distinct out-of-scope headers must not become a supported form because
        # its terms/footnotes happen to mention ECS, FATCA or another product.
        for pattern, label in ((r"\bassignment request form\b", "assignment_request_form"), (r"\bcustomer\s+declaration\s*[-–,:]?\s*(?:application|proposal)", "customer_declaration_application_proposal_form")):
            if re.search(pattern, text[:5000], re.I):
                return None, 0.0, [{"document_type": "out_of_scope", "observed_title": label, "score": 0, "matched_rules": [pattern]}]
        evidence = []
        for doc_type, rules in CLASSIFY_RULES.items():
            matched = [pattern for pattern, _ in rules if re.search(pattern, text, re.I)]
            score = sum(weight for pattern, weight in rules if re.search(pattern, text, re.I))
            evidence.append({"document_type": doc_type, "score": score, "matched_rules": matched})
        evidence.sort(key=lambda item: item["score"], reverse=True)
        top, runner = evidence[0]["score"], evidence[1]["score"]
        if top < 4:
            return None, 0.0, evidence[:3]
        if top == runner:
            return None, .45, evidence[:3]
        confidence = min(.97, .68 + .035 * top + .012 * (top - runner))
        if runner >= 4:
            confidence = min(confidence, .75)
        return evidence[0]["document_type"], round(confidence, 3), evidence[:3]

    def process_text(self, text: str, document_type: str | None = None, *, source_name: str = "text_input", handwritten: bool | None = None, ocr_confidence: float = 1.0) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not 0 <= ocr_confidence <= 1:
            raise ValueError("ocr_confidence must be between 0 and 1")
        lines = [_Line(line, index=i + 1, confidence=ocr_confidence) for i, line in enumerate(text.splitlines()) if line.strip()]
        return self._process(lines, self._type_hint(document_type), source_name, handwritten, {"engine": "provided_text", "pages": 1, "mean_word_confidence": ocr_confidence, "is_image_ocr": False, "warnings": ["Text-input tests do not measure image OCR or handwriting accuracy"], "passes": []})

    def process_file(self, path: str | Path, document_type: str | None = None) -> dict[str, Any]:
        path = Path(path)
        hint = self._type_hint(document_type)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".text"}:
            return self.process_text(path.read_text(encoding="utf-8"), hint, source_name=path.name)
        if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            raise ValueError("Supported files: PDF, PNG, JPEG, TIFF, BMP, WebP, or TXT")
        try:
            lines, ocr = self._read_document(path)
        except (ImportError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return self._process([], hint, path.name, None, {"engine": "local_tesseract", "pages": 0, "mean_word_confidence": 0., "is_image_ocr": True, "warnings": [f"read_or_ocr_failed: {type(exc).__name__}: {exc}"], "passes": []})
        result = self._process(lines, hint, path.name, None, ocr)
        if self.use_printed_region_ocr or self.use_handwritten_region_ocr:
            result = self._augment_printed_regions(path, result)
        result = self._augment_handwriting(path, result)
        return self._augment_checkboxes(path, result) if self.use_checkbox_detection else result

    def _augment_checkboxes(self, path: Path, result: dict[str, Any]) -> dict[str, Any]:
        """Inspect explicit option-box pixels; labels are layout metadata only."""
        template = self.region_templates.get(result.get("document_type"))
        if not template or not template.get("checkbox_fields") or path.suffix.lower() == ".pdf":
            return result
        from PIL import Image
        from desicrew.checkboxes import inspect_box, selected_option
        with Image.open(path) as original:
            expected_ratio = template.get("aspect_ratio", 0)
            if expected_ratio <= 0 or abs((original.width / original.height) / expected_ratio - 1) > .025:
                result["ocr"].setdefault("warnings", []).append("Checkbox layout aspect ratio differs; pixel detection skipped")
                return result
            for field, options in template["checkbox_fields"].items():
                if field not in result["fields"] or len(options) > 12:
                    continue
                evidence = []
                for option in options:
                    normalized = option.get("bbox_normalized", [])
                    if len(normalized) != 4 or not all(isinstance(n, (int, float)) and 0 <= n <= 1 for n in normalized):
                        continue
                    bbox = [round(normalized[0] * original.width), round(normalized[1] * original.height), round(normalized[2] * original.width), round(normalized[3] * original.height)]
                    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                        continue
                    evidence.append({**inspect_box(original.crop(bbox)), "label": option["label"], "description": option.get("description", option["label"]), "page": 1, "bbox": bbox})
                value, issues = selected_option(evidence)
                if len(evidence) != len(options):
                    value = None
                    issues.append("checkbox_option_region_missing")
                previous = result["fields"][field]
                if previous.get("value") is not None and value is not None and previous["value"].casefold() != value.casefold():
                    value = None
                    issues.append("text_and_pixel_checkbox_candidates_disagree")
                selected = next((option for option in evidence if option["status"] == "selected"), None)
                result["fields"][field] = {"value": value, "confidence": .55 if value is not None else 0., "status": "needs_review" if value is not None else "unreadable", "needs_review": True, "validation_issues": list(dict.fromkeys(issues + ["checkbox_pixel_heuristic_not_calibrated", "checkbox_template_alignment_requires_review", "handwritten_form_requires_review"])), "source": {"method": "local_checkbox_interior_heuristic", "page": 1, "bbox": selected["bbox"] if selected else None, "candidate_text": selected["label"] if selected else None, "label_source": "geometry_only_template_option_labels", "previous_text_source": previous.get("source"), "option_evidence": evidence}, "checkbox_candidates": evidence}
        result["needs_review"] = True
        result["review_reasons"] = list(dict.fromkeys(result["review_reasons"] + ["checkbox_pixels_and_layout_require_review"]))
        return result

    def _regions_for_file(self, path: Path, doc_type: str, image_size: tuple[int, int]) -> list[dict[str, Any]]:
        """Exact paths, or basename+type+dimensions, then explicit normalized layouts."""
        documents = [entry for entry in self.roi_documents if entry.get("document_type") == doc_type and Path(entry.get("file", "")).resolve() == path.resolve()]
        if not documents:
            documents = [entry for entry in self.roi_documents if entry.get("document_type") == doc_type and Path(entry.get("file", "")).name == path.name and entry.get("image_size") == list(image_size)]
        if len(documents) == 1:
            return documents[0].get("regions", [])[:44]
        if len(documents) > 1:
            return []  # Ambiguous manifests cannot authorize an arbitrary crop.
        template = self.region_templates.get(doc_type)
        if not template:
            return []
        expected_ratio = template.get("aspect_ratio")
        actual_ratio = image_size[0] / image_size[1]
        if not isinstance(expected_ratio, (float, int)) or expected_ratio <= 0 or abs(actual_ratio / expected_ratio - 1) > .025:
            return []
        regions = []
        for region in template.get("regions", [])[:44]:
            normalized = region.get("bbox_normalized")
            if not isinstance(normalized, list) or len(normalized) != 4 or not all(isinstance(n, (int, float)) and 0 <= n <= 1 for n in normalized):
                continue
            bbox = [round(normalized[0] * image_size[0]), round(normalized[1] * image_size[1]), round(normalized[2] * image_size[0]), round(normalized[3] * image_size[1])]
            regions.append({"field": region.get("field"), "page": 1, "bbox": bbox, "region_source": "normalized_supplied_layout_template_requires_alignment_review", "template_id": template.get("template_id", doc_type)})
        return regions

    def _augment_printed_regions(self, path: Path, result: dict[str, Any]) -> dict[str, Any]:
        is_handwritten = result.get("document_type") in HANDWRITTEN_TYPES
        if result.get("document_type") == "unknown" or path.suffix.lower() == ".pdf" or (is_handwritten and not self.use_handwritten_region_ocr) or (not is_handwritten and not self.use_printed_region_ocr):
            return result
        from PIL import Image, ImageOps
        with Image.open(path) as original:
            regions = self._regions_for_file(path, result["document_type"], original.size)
            for region in regions:
                field = region.get("field")
                bbox = region.get("bbox")
                if field in {"frequency", "reason_for_multiple_policies"} or field not in result["fields"] or not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(n, int) for n in bbox) or not (0 <= bbox[0] < bbox[2] <= original.width and 0 <= bbox[1] < bbox[3] <= original.height):
                    continue
                candidates = []
                evidence = []
                with tempfile.TemporaryDirectory(prefix="desicrew_region_") as temp:
                    crop_path = Path(temp) / "crop.png"
                    cropped = original.crop(bbox).convert("RGB")
                    preprocessing_info = {"mode": "none", "destructive_cleanup_not_validated": False}
                    if self.region_preprocessing != "none":
                        from desicrew.handwriting import LocalHandwritingRecognizer
                        cropped, preprocessing_info = LocalHandwritingRecognizer._prepare_crop(cropped, self.region_preprocessing)
                    cropped = ImageOps.autocontrast(ImageOps.grayscale(cropped))
                    scale = min(3., max(2. if field == "address" else 1., 64 / cropped.height))
                    cropped = cropped.resize((round(cropped.width * scale), round(cropped.height * scale)), Image.Resampling.LANCZOS)
                    scale_x = cropped.width / (bbox[2] - bbox[0])
                    scale_y = cropped.height / (bbox[3] - bbox[1])
                    cropped.save(crop_path)
                    for psm in ((6, 4) if field == "address" else (7, 6)):
                        try:
                            command = [self.tesseract_command, str(crop_path), "stdout", "-l", self.language, "--oem", "1", "--psm", str(psm), "tsv"]
                            completed = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout, check=True)
                            crop_lines, _ = self._parse_tsv(completed.stdout, 1)
                            raw = " ".join(line.text for line in crop_lines)
                            confidence = min((line.confidence for line in crop_lines), default=0.)
                            evidence.append({"text": raw, "ocr_confidence": confidence, "psm": psm, "bbox": bbox, "region_source": region.get("region_source", "explicit_printed_region"), "preprocessing": preprocessing_info})
                            if raw:
                                for crop_line in crop_lines:
                                    if crop_line.bbox:
                                        crop_line.bbox = [round(crop_line.bbox[0] / scale_x) + bbox[0], round(crop_line.bbox[1] / scale_y) + bbox[1], round(crop_line.bbox[2] / scale_x) + bbox[0], round(crop_line.bbox[3] / scale_y) + bbox[1]]
                                candidates.append((raw, _combine_lines(crop_lines), "local_region_ocr"))
                        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                            result["ocr"].setdefault("warnings", []).append(f"Local region OCR failed for {field}: {type(exc).__name__}: {exc}")
                roi_candidates = list(candidates)
                previous = result["fields"][field]
                source = previous.get("source")
                if source and source.get("candidate_text"):
                    parts = [_Line(part.get("text", ""), page=part.get("page", 1), confidence=part.get("ocr_confidence", 0.), bbox=part.get("bbox"), index=part.get("line", 0)) for part in source.get("lines", [])]
                    if parts:
                        candidates.append((source["candidate_text"], _combine_lines(parts), source.get("method", "whole_page_ocr")))
                updated = self._field_result(field, candidates, is_handwritten, True)
                conflicts = [issue for issue in previous.get("validation_issues", []) if issue in {"conflicting_candidates", "ocr_segmentation_passes_disagree"}]
                if conflicts:
                    updated["value"] = None
                    updated["status"] = "unreadable"
                    updated["validation_issues"].extend(conflicts)
                # For ordinary printed names/address only, an explicit localized
                # two-pass consensus can outrank a weak whole-page association.
                # Never use this rule for IDs, dates, MRZ or handwritten fields.
                roi_values = [self._validate(field, raw)[0] for raw, _, _ in roi_candidates]
                strong_consensus = len(roi_candidates) == 2 and all(value is not None for value in roi_values) and len({value.casefold() for value in roi_values}) == 1 and min(line.confidence for _, line, _ in roi_candidates) >= .80
                page_value = self._validate(field, source.get("candidate_text", ""))[0] if source else None
                weak_page = source is None or source.get("ocr_confidence", 0.) < .80 or source.get("method") == "strict_format_unlabelled"
                page_agrees = page_value is not None and page_value.casefold() == roi_values[0].casefold() if strong_consensus else False
                if not is_handwritten and field in {"full_name", "father_name", "name", "address"} and updated["value"] is None and strong_consensus and (weak_page or page_agrees):
                    updated = self._field_result(field, roi_candidates, False, True)
                    resolution = "localized_printed_consensus_resolves_segmentation_disagreement" if page_agrees else "localized_printed_consensus_over_weak_page_ocr"
                    updated["validation_issues"] = list(dict.fromkeys(updated["validation_issues"] + [resolution]))
                    updated["source"]["superseded_whole_page_source"] = source
                if field == "mrz_line_2" and updated["value"] is None and len(roi_candidates) == 2:
                    observed = [re.sub(r"\s", "", raw) for raw, _, _ in roi_candidates]
                    if observed[0] == observed[1] and re.fullmatch(r"[A-Z0-9<]{30,80}", observed[0]) and min(line.confidence for _, line, _ in roi_candidates) >= .60:
                        updated["observed_value"] = observed[0]
                        updated["observation_confidence"] = round(min(line.confidence for _, line, _ in roi_candidates), 3)
                        updated["validation_status"] = "invalid_source_format"
                        updated["status"] = "invalid_format"
                        updated["validation_issues"].append("readable_observation_kept_separate_from_validated_mrz")
                updated["confidence"] = min(updated["confidence"], .85)
                updated["needs_review"] = True
                if updated["value"] is not None:
                    updated["status"] = "needs_review"
                updated["validation_issues"].append("explicit_region_alignment_requires_review")
                if self.region_preprocessing != "none":
                    updated["validation_issues"].append("destructive_rule_cleanup_requires_visual_review")
                updated["region_candidates"] = evidence
                result["fields"][field] = updated
        if regions:
            result["needs_review"] = True
            result["review_reasons"] = list(dict.fromkeys(result["review_reasons"] + ["explicit_regions_require_alignment_review"]))
        return _refresh_field_review_reasons(result)

    def _augment_handwriting(self, path: Path, result: dict[str, Any]) -> dict[str, Any]:
        """Optional explicit local crop recognition. Never infer checkbox marks."""
        if self.handwriting_backend is None or result.get("document_type") not in HANDWRITTEN_TYPES:
            return result
        if path.suffix.lower() == ".pdf":
            result["ocr"].setdefault("warnings", []).append("Local handwriting ROI integration currently supports original raster images, not PDF coordinates")
            return result
        from PIL import Image
        model_candidates = []
        with Image.open(path) as original:
            regions = self._regions_for_file(path, result["document_type"], original.size)
            if not regions:
                result["ocr"].setdefault("warnings", []).append("Local handwriting backend enabled but no guarded ROI entry/template matches this file")
                return result
            for region in regions:
                field = region.get("field")
                if field not in result["fields"] or field in {"frequency", "reason_for_multiple_policies"}:
                    continue
                bbox = region.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(n, int) for n in bbox) or not (0 <= bbox[0] < bbox[2] <= original.width and 0 <= bbox[1] < bbox[3] <= original.height):
                    result["ocr"].setdefault("warnings", []).append(f"Invalid handwriting ROI for {field}; skipped")
                    continue
                page = int(region.get("page", 1))
                if page != 1:
                    result["ocr"].setdefault("warnings", []).append(f"Non-first-page raster ROI for {field} is unsupported; skipped")
                    continue
                try:
                    recognized = self.handwriting_backend.recognize(original.crop(bbox).convert("RGB"), field=field, page=page, bbox=bbox)
                    raw = str(recognized.get("text", ""))
                    confidence = max(0., min(1., float(recognized.get("confidence", 0.))))
                except (ImportError, OSError, ValueError, RuntimeError, TimeoutError) as exc:
                    result["ocr"].setdefault("warnings", []).append(f"Local handwriting recognition failed for {field}: {type(exc).__name__}: {exc}")
                    continue
                model_candidates.append({"field": field, **recognized, "roi": region})
                previous = result["fields"][field]
                candidate_line = _Line(raw, page=page, confidence=confidence, bbox=bbox)
                candidates = []
                source = previous.get("source")
                if source and source.get("candidate_text"):
                    original_parts = [_Line(part.get("text", ""), page=part.get("page", 1), confidence=part.get("ocr_confidence", 0.), bbox=part.get("bbox"), index=part.get("line", 0)) for part in source.get("lines", [])]
                    original_line = _combine_lines(original_parts) if original_parts else _Line(source.get("text", ""), page=source.get("page", 1), confidence=source.get("ocr_confidence", 0.), bbox=source.get("bbox"), index=source.get("line", 0))
                    candidates.append((source["candidate_text"], original_line, source.get("method", "tesseract_candidate")))
                if raw and not recognized.get("terminated_early"):
                    candidates.append((raw, candidate_line, "local_handwriting_crop"))
                updated = self._field_result(field, candidates, True, True)
                updated["validation_issues"] = list(dict.fromkeys(updated["validation_issues"] + recognized.get("validation_issues", []) + ["handwritten_model_not_calibrated", "explicit_roi_requires_template_alignment_review"]))
                original_conflicts = [issue for issue in previous.get("validation_issues", []) if issue in {"conflicting_candidates", "ocr_segmentation_passes_disagree"}]
                if original_conflicts:
                    updated["value"] = None
                    updated["confidence"] = min(updated["confidence"], .55)
                    updated["status"] = "unreadable"
                    updated["validation_issues"].extend(original_conflicts)
                updated["needs_review"] = True
                updated["handwriting_candidate"] = {"text": raw, "model": recognized.get("model"), "confidence": confidence, "confidence_kind": recognized.get("confidence_kind", "heuristic_quality_score_not_probability"), "page": page, "bbox": bbox, "region_source": region.get("region_source", "explicit_roi"), "provenance": recognized.get("provenance"), "terminated_early": recognized.get("terminated_early", False), "raw_candidates": recognized.get("raw_candidates"), "verification_candidates": recognized.get("verification_candidates")}
                result["fields"][field] = updated
        result["ocr"]["handwriting"] = {"engine": "optional_local_model", "candidates": model_candidates, "mandatory_human_review": True}
        result["needs_review"] = True
        result["review_reasons"] = list(dict.fromkeys(result["review_reasons"] + ["optional_local_handwriting_model_requires_review"]))
        return result

    def _read_document(self, path: Path) -> tuple[list[_Line], dict[str, Any]]:
        from PIL import Image, ImageOps
        images = []
        native_lines = []
        warnings = []
        total_pixels = 0
        pdf = None
        if path.suffix.lower() == ".pdf":
            import fitz
            pdf = fitz.open(path)
            if pdf.needs_pass:
                pdf.close()
                raise ValueError("Encrypted PDF requires an unlocked local copy")
            if len(pdf) > self.max_pages:
                pdf.close()
                raise ValueError(f"PDF exceeds {self.max_pages} page safety limit")
            for page_no, page in enumerate(pdf, 1):
                native = page.get_text("text")
                # Prefer native text when available; sparse text alone is not enough.
                has_raster = bool(page.get_images(full=True))
                native_ok = len(re.sub(r"\s", "", native)) >= 60 and "\ufffd" not in native
                if native_ok:
                    native_confidence = .85 if has_raster else 1.0
                    native_lines.extend(_Line(line, page=page_no, index=i + 1, confidence=native_confidence) for i, line in enumerate(native.splitlines()) if line.strip())
                if has_raster and native_ok:
                    warnings.append(f"Page {page_no} contains raster imagery; native text also checked against local OCR")
                if not native_ok or has_raster:
                    import math
                    render_pixels = math.ceil(page.rect.width * 300 / 72) * math.ceil(page.rect.height * 300 / 72)
                    if render_pixels > self.max_pixels:
                        pdf.close()
                        raise ValueError("PDF page exceeds rendered pixel safety limit")
                    if total_pixels + render_pixels > self.max_total_pixels:
                        pdf.close()
                        raise ValueError("PDF exceeds cumulative rendered pixel safety limit")
                    pix = page.get_pixmap(dpi=300, alpha=False)
                    if pix.width * pix.height > self.max_pixels or total_pixels + pix.width * pix.height > self.max_total_pixels:
                        pdf.close()
                        raise ValueError("PDF raster exceeds pixel safety limit")
                    total_pixels += pix.width * pix.height
                    images.append((page_no, Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")))
            pages = len(pdf)
            pdf.close()
        else:
            try:
                opened_image = Image.open(path)
            except Image.DecompressionBombError as exc:
                raise ValueError("Image exceeds Pillow decompression safety limit") from exc
            with opened_image as im:
                pages = getattr(im, "n_frames", 1)
                if pages > self.max_pages:
                    raise ValueError(f"Image exceeds {self.max_pages} page safety limit")
                for page_index in range(pages):
                    im.seek(page_index)
                    if im.width * im.height > self.max_pixels:
                        raise ValueError("Image exceeds pixel safety limit")
                    if total_pixels + im.width * im.height > self.max_total_pixels:
                        raise ValueError("Image exceeds cumulative pixel safety limit")
                    total_pixels += im.width * im.height
                    image = ImageOps.exif_transpose(im.copy())
                    if image.mode in ("RGBA", "LA"):
                        background = Image.new("RGB", image.size, "white")
                        background.paste(image, mask=image.getchannel("A"))
                        image = background
                    images.append((page_index + 1, image.convert("RGB")))
        if images and not Path(self.tesseract_command).is_file() and not shutil.which(self.tesseract_command):
            raise RuntimeError("Local Tesseract not installed; no cloud OCR fallback is used")
        lines = native_lines[:]
        passes = []
        alternate = []
        confidence_total = 0.0
        word_count = 0
        for page_no, image in images:
            gray = ImageOps.autocontrast(ImageOps.grayscale(image))
            # Enlarge small scans, with a bounded output size. Never threshold away ink.
            if gray.width < 1600:
                factor = min(2.5, 1600 / gray.width)
                new_size = (round(gray.width * factor), round(gray.height * factor))
                if new_size[0] * new_size[1] <= self.max_pixels:
                    gray = gray.resize(new_size, Image.Resampling.LANCZOS)
            with tempfile.TemporaryDirectory(prefix="desicrew_ocr_") as temp:
                input_path = Path(temp) / "page.png"
                gray.save(input_path)
                page_runs = []
                for psm in (6, 11):
                    command = [self.tesseract_command, str(input_path), "stdout", "-l", self.language, "--oem", "1", "--psm", str(psm), "tsv"]
                    completed = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout, check=True)
                    page_lines, words = self._parse_tsv(completed.stdout, page_no)
                    mean_conf = sum(words) / len(words) if words else 0.0
                    page_runs.append((page_lines, mean_conf, psm, len(words)))
                # The higher word-confidence pass is a baseline, not proof of correctness.
                # Field conflicts across both independent segmentation passes are flagged.
                chosen = max(page_runs, key=lambda run: (run[1], run[3]))
                lines.extend(chosen[0])
                alternate.extend(next(run[0] for run in page_runs if run[2] != chosen[2]))
                confidence_total += chosen[1] * chosen[3]
                word_count += chosen[3]
                passes.extend({"page": page_no, "psm": run[2], "mean_word_confidence": round(run[1], 3), "word_count": run[3], "selected": run[2] == chosen[2]} for run in page_runs)
        if images:
            warnings.append("English Tesseract baseline is not a validated handwriting recognizer")
        ocr = {"engine": "local_tesseract" if images else "native_pdf_text", "language": self.language, "pages": pages, "is_image_ocr": bool(images), "mean_word_confidence": round(confidence_total / word_count, 3) if word_count else (1.0 if native_lines else 0.0), "warnings": warnings, "passes": passes}
        ocr["_alternate_lines"] = alternate
        return lines, ocr

    @staticmethod
    def _parse_tsv(tsv: str, page_no: int) -> tuple[list[_Line], list[float]]:
        groups = {}
        confidences = []
        for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
            if row.get("level") != "5" or not row.get("text", "").strip():
                continue
            try:
                conf = max(0.0, min(1.0, float(row["conf"]) / 100))
                box = [int(row[k]) for k in ("left", "top", "width", "height")]
            except (KeyError, ValueError):
                continue
            key = (row.get("block_num"), row.get("par_num"), row.get("line_num"))
            groups.setdefault(key, []).append((row["text"], conf, box))
            confidences.append(conf)
        lines = []
        for i, words in enumerate(groups.values(), 1):
            left = min(w[2][0] for w in words)
            top = min(w[2][1] for w in words)
            right = max(w[2][0] + w[2][2] for w in words)
            bottom = max(w[2][1] + w[2][3] for w in words)
            # A low-confidence identifier token should not hide behind confident labels.
            confidence = min(w[1] for w in words)
            lines.append(_Line(" ".join(w[0] for w in words), page_no, confidence, [left, top, right, bottom], i))
        return lines, confidences

    def _nearby_value_line(self, field: str, label: _Line, index: int, lines: list[_Line], label_pattern: re.Pattern) -> _Line | None:
        """Match sparse OCR values spatially instead of unrelated reading order."""
        if label.bbox is None:
            return lines[index + 1] if index + 1 < len(lines) and lines[index + 1].page == label.page else None
        left, top, right, bottom = label.bbox
        height = max(10, bottom - top)
        nearby = []
        for candidate in lines:
            if candidate is label or candidate.page != label.page or candidate.bbox is None or label_pattern.search(candidate.text):
                continue
            if self._validate(field, candidate.text)[0] is None:
                continue
            x0, y0, x1, y1 = candidate.bbox
            y_distance = abs((y0 + y1 - top - bottom) / 2)
            if x0 >= right - 10 and y_distance <= height:
                nearby.append((1 + (x0 - right) / (5 * height) + y_distance / height, candidate))
            elif -5 <= y0 - bottom <= 3 * height and abs(x0 - left) <= 2 * height:
                nearby.append((3 + max(0, y0 - bottom) / (2 * height) + abs(x0 - left) / (3 * height), candidate))
        return min(nearby, key=lambda item: item[0])[1] if nearby else None

    @staticmethod
    def _address_continuation_lines(last_line: _Line, lines: list[_Line]) -> list[_Line]:
        """Follow aligned address rows, excluding interleaved adjacent columns."""
        left, top, right, bottom = last_line.bbox
        height = max(10, bottom - top)
        aligned = sorted((line for line in lines if line is not last_line and line.page == last_line.page and line.bbox is not None and line.bbox[1] >= bottom - 2 and abs(line.bbox[0] - left) <= max(30, height)), key=lambda line: line.bbox[1])
        result = []
        for line in aligned:
            if line.bbox[1] - bottom > 2.5 * height or len(result) >= 4:
                break
            result.append(line)
            bottom = line.bbox[3]
        return result

    def _extract(self, lines: list[_Line], doc_type: str) -> dict[str, list[tuple[str, _Line, str]]]:
        fields = DOCUMENT_SCHEMAS[doc_type]
        candidates = {field: [] for field in fields}
        aliases = [(alias, field) for field in fields for alias in ALIASES[field]]
        aliases.sort(key=lambda item: len(item[0]), reverse=True)
        pattern = re.compile(r"(?<![A-Za-z])(?:" + "|".join(f"(?P<f{i}>{_label_re(alias)})" for i, (alias, _) in enumerate(aliases)) + r")(?![A-Za-z])", re.I)
        for i, line in enumerate(lines):
            matches = [
                match for match in pattern.finditer(line.text)
                if not line.text[:match.start()].strip()
                or re.fullmatch(r"[\d\s().:-]+", line.text[:match.start()].strip())
                or re.fullmatch(r"[a-zA-Z]\s*[).]\s*", line.text[:match.start()].strip())
                or (doc_type not in HANDWRITTEN_TYPES and line.text[:match.start()].strip().endswith("/"))
                or re.match(r"\s*[:=|]", line.text[match.end():])
            ]
            for j, match in enumerate(matches):
                alias_index = int(match.lastgroup[1:])
                _, field = aliases[alias_index]
                prefix = line.text[:match.start()].strip()
                # A label in prose is insufficient. Labels start a line or have a
                # separator/another earlier field; this avoids copying declarations.
                if prefix and j == 0 and not re.fullmatch(r"(?:[\d\s().:-]+|[a-zA-Z]\s*[).]\s*)", prefix) and not (doc_type not in HANDWRITTEN_TYPES and prefix.endswith("/")):
                    continue
                end = matches[j + 1].start() if j + 1 < len(matches) else len(line.text)
                tail = line.text[match.end():end]
                value = _clean_value(tail)
                if doc_type not in HANDWRITTEN_TYPES and (value.startswith("/") or (field == "pan_number" and value.lower() == "card")):
                    value = ""  # Bilingual label suffix or title, not a field value.
                candidate_index = None
                continuation_start = i + 1
                if value and not re.fullmatch(r"[.\-_/\s]+", value):
                    candidates[field].append((value, line, "label_same_line"))
                    candidate_index = len(candidates[field]) - 1
                else:
                    nxt = self._nearby_value_line(field, line, i, lines, pattern)
                    if nxt is None:
                        continue
                    if not pattern.search(nxt.text) and nxt.text.strip() and not re.fullmatch(r"[.\-_/\s]+", nxt.text.strip()):
                        candidates[field].append((_clean_value(nxt.text), _combine_lines([line, nxt]), "label_next_line"))
                        candidate_index = len(candidates[field]) - 1
                        continuation_start = lines.index(nxt) + 1
                if field == "address" and candidate_index is not None:
                    additions = []
                    value, source, method = candidates[field][candidate_index]
                    last_part = (source.contributing_lines or [source])[-1]
                    continuation_lines = self._address_continuation_lines(last_part, lines) if last_part.bbox else lines[continuation_start:continuation_start + 4]
                    for nxt in continuation_lines:
                        if nxt.page != line.page or pattern.search(nxt.text) or re.match(r"^[A-Za-z][A-Za-z /'()]{1,30}:", nxt.text) or re.search(r"\b(?:signature|uidai|www\.|male|female)\b", nxt.text, re.I):
                            break
                        if nxt.text.strip():
                            additions.append(nxt)
                    if additions:
                        candidates[field][candidate_index] = (", ".join([value] + [_clean_value(part.text) for part in additions]), _combine_lines([source] + additions), method + "_multiline")
                elif field in {"reason_for_multiple_policies", "frequency"} and candidate_index is not None:
                    additions = []
                    for nxt in lines[continuation_start:i + 9]:
                        if nxt.page != line.page or not re.match(r"^\s*(?:\[[ xX✓✔]*\]|[☐☑☒✓✔])", nxt.text):
                            break
                        additions.append(nxt)
                    if additions:
                        value, source, method = candidates[field][candidate_index]
                        candidates[field][candidate_index] = (" ".join([value] + [_clean_value(part.text) for part in additions]), _combine_lines([source] + additions), method + "_multiline")
        # Strict-format fallbacks do not invent unlabeled names/addresses.
        fallback = {
            "aadhaar_number": r"(?<!\d)(?:\d{4}[ -]?){2}\d{4}(?!\d)",
            "pan_number": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
            "passport_number": r"\b[A-Z][0-9]{7}\b",
            "ifsc_code": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
            "mrz_line_2": r"(?m)^[A-Z0-9< ]{30,80}$",
        }
        for field, regex in fallback.items():
            if field not in candidates or any(self._validate(field, raw)[0] is not None for raw, _, _ in candidates[field]):
                continue
            for line in lines:
                if field == "passport_number" and ("<" in line.text or re.search(r"\bmrz\b", line.text, re.I)):
                    continue  # Use the checked MRZ path instead of a shape-only match.
                for match in re.finditer(regex, line.text.strip()):
                    if field == "mrz_line_2" and ("<" not in match.group() or sum(ch.isdigit() for ch in match.group()) < 10):
                        continue
                    candidates[field].append((match.group(), line, "strict_format_unlabelled"))
        # MRZ line 2: never remove fillers or turn digits into an integer.
        if "mrz_line_2" in candidates and candidates["mrz_line_2"]:
            raw, line, _ = candidates["mrz_line_2"][0]
            checked_mrz, mrz_issues = self._validate("mrz_line_2", raw)
            if checked_mrz is not None and not mrz_issues and not candidates["passport_number"]:
                candidates["passport_number"].append((checked_mrz[:9].rstrip("<"), line, "checked_mrz_passport_number"))
        return candidates

    def _validate(self, field: str, raw: str) -> tuple[str | None, list[str]]:
        value = _clean_value(raw)
        if not value or re.fullmatch(r"[.\-_/\s]+", value) or value.lower() in {"n/a", "na", "illegible", "unreadable", "unknown", "not provided", "[unreadable]"}:
            return None, ["empty_or_unreadable"]
        if field in DATE_FIELDS:
            return _valid_date(value)
        if field == "mrz_line_2":
            value = re.sub(r"\s", "", value)  # Remove OCR spacing, preserve fillers/characters.
            if not re.fullmatch(r"[A-Z0-9<]{44}", value):
                return None, ["mrz_line_2_must_have_44_unmodified_characters"]
            if any(not value[index].isdigit() for index in (9, 19, 27, 43)):
                return None, ["mrz_mandatory_check_digit_missing"]
            checks = ((value[:9], value[9]), (value[13:19], value[19]), (value[21:27], value[27]))
            if any(_checksum_mrz(part) != int(digit) for part, digit in checks):
                return value, ["mrz_check_digit_failed"]
            # ICAO permits '<' at position 43 only when optional data is unused.
            if value[42] == "<" and value[28:42] != "<" * 14:
                return None, ["mrz_optional_data_check_digit_missing"]
            if value[42] != "<" and (not value[42].isdigit() or _checksum_mrz(value[28:42]) != int(value[42])):
                return value, ["mrz_optional_data_check_digit_failed"]
            if _checksum_mrz(value[:10] + value[13:20] + value[21:43]) != int(value[43]):
                return value, ["mrz_composite_check_digit_failed"]
            return value, []
        if field in IDENTIFIER_FIELDS:
            # Only formatting spaces/hyphens for Aadhaar are removed. Never repair
            # O/0, I/1, S/5 or B/8 using a regex and silently call it correct.
            if field == "aadhaar_number":
                value = re.sub(r"[ -]", "", value)
                if not re.fullmatch(r"[0-9]{12}", value):
                    return None, ["invalid_aadhaar_number_shape"]
                return value, (["aadhaar_leading_digit_not_valid_for_issued_number"] if value[0] in "01" else []) + ["aadhaar_checksum_not_verified"]
            if field == "pan_number":
                value = value.replace(" ", "")
                return (value, []) if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value) else (None, ["invalid_pan_shape_no_character_repair"])
            if field == "ifsc_code":
                value = value.replace(" ", "")
                return (value, ["ifsc_directory_not_verified"]) if re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", value) else (None, ["invalid_ifsc_shape_no_character_repair"])
            if field == "bank_account_number":
                value = value.replace(" ", "")
                return (value, ["bank_account_not_verified"]) if re.fullmatch(r"[0-9]{6,20}", value) else (None, ["invalid_bank_account_shape_no_character_repair"])
            if field == "passport_number":
                return (value, []) if re.fullmatch(r"[A-Z][0-9]{7}", value) else (None, ["invalid_indian_passport_number_shape"])
            if field == "dl_number":
                # Different state/year layouts exist; preserve hyphens and zeros.
                return (value, ["dl_format_jurisdiction_not_verified"]) if re.fullmatch(r"[A-Z]{2}[ -]?[0-9]{2}[ -]?[0-9]{4}[ -]?[0-9]{7}", value) else (None, ["unrecognized_dl_number_shape"])
            if field == "tin_pan":
                value = value.replace(" ", "")
                if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", value):
                    return value, []
                if re.fullmatch(r"[A-Z0-9-]{4,25}", value):
                    return value, ["tin_jurisdiction_and_format_not_verified"]
                return None, ["invalid_tin_or_pan_shape"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/ -]{2,35}", value):
                return None, ["invalid_identifier_shape"]
            if not any(character.isdigit() for character in value):
                return None, ["identifier_has_no_digits"]
            return value, ["identifier_format_not_verified"]
        if field == "amount_figures":
            value = re.sub(r"^(?:INR|Rs\.?|₹)\s*", "", value, flags=re.I)
            value = re.sub(r"\s*/-\s*$", "", value)
            if not re.fullmatch(r"(?:[0-9]+|[1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]?(?:,[0-9]{2})*,[0-9]{3})(?:\.[0-9]{1,2})?", value):
                return None, ["invalid_amount_figures"]
            return value.replace(",", ""), ["currency_assumed_inr"]
        if field in {"reason_for_multiple_policies", "frequency"}:
            selected = re.findall(r"(?:\[[xX✓✔]\]|[☑☒✓✔])\s*([^\[☑☒✓✔]+)", value)
            if len(selected) == 1:
                selection = _clean_value(re.split(r"\[\s*\]|☐", selected[0])[0])
                return (selection, ["checkbox_selection_requires_visual_review"]) if selection else (None, ["checkbox_mark_without_option_label"])
            if len(selected) > 1:
                return None, ["multiple_selected_checkboxes"]
            if field == "reason_for_multiple_policies":
                return None, ["checkbox_selection_not_explicit"]
            if re.search(r"\[\s*\]|☐", value):
                return None, ["frequency_checkbox_selection_not_explicit"]
            if re.fullmatch(r"(?:monthly|quarterly|half[- ]yearly|yearly|annually|annual|as and when presented|on demand|one[- ]time|weekly)", value, re.I):
                return value, []
            return None, ["unrecognized_or_ambiguous_frequency"]
        # Periods and commas at the edge of a handwritten name/place are often
        # pen marks or sentence punctuation; preserve internal initials and raw
        # OCR evidence while normalizing only the field value.
        normalized = value.rstrip(" .,") if field != "address" else value
        punctuation_removed = normalized != value
        value = normalized
        if re.search(r"\b(?:signature|declaration|hereby|certify|agree|please|mandatory)\b", value, re.I) or len(value) > 200:
            return None, ["candidate_looks_like_instruction_or_form_prose"]
        if field != "address" and not re.search(r"[A-Za-z\u0900-\u097f]", value):
            return None, ["name_or_place_has_no_letters"]
        if field != "address" and re.search(r"[0-9<>\[\]{}]", value):
            return None, ["unexpected_characters_in_name_or_place"]
        if field != "address" and any(not (ch.isalpha() or ch.isspace() or ch in "'’‘.,/-&") for ch in value):
            return None, ["unexpected_characters_in_name_or_place"]
        return value, (["terminal_name_or_place_punctuation_removed"] if punctuation_removed else [])

    def _field_result(self, field: str, candidates: list[tuple[str, _Line, str]], handwritten: bool, image_ocr: bool, alternate_candidates: list[tuple[str, _Line, str]] | None = None) -> dict[str, Any]:
        if not candidates:
            return {"value": None, "confidence": 0.0, "status": "missing", "source": None, "validation_issues": ["field_not_found"], "needs_review": True}
        parsed = [(self._validate(field, raw), raw, line, method) for raw, line, method in candidates]
        valid = [item for item in parsed if item[0][0] is not None]
        if not valid:
            (_, issues), raw, line, method = parsed[0]
            return {"value": None, "confidence": 0.0, "status": "unreadable", "source": _source_evidence(line, raw, method), "validation_issues": issues, "needs_review": True}
        def comparison_key(value):
            return value if field in IDENTIFIER_FIELDS or field in DATE_FIELDS or field in {"mrz_line_2", "amount_figures"} else value.casefold()
        values = {comparison_key(item[0][0]) for item in valid}
        ((value, issues), raw, line, method) = max(valid, key=lambda item: item[2].confidence)
        issues = list(issues)
        score = .96 if method.startswith("label_same_line") else .86
        if method == "checked_mrz_passport_number":
            score = .92
        if method == "strict_format_unlabelled":
            score = .84 if field != "mrz_line_2" else .91
        score = min(score, line.confidence)
        if len(values) > 1:
            issues.append("conflicting_candidates")
            score = min(score, .50)
            # No arbitrary choice between two plausible bank/identity values.
            value = None
        if alternate_candidates:
            alternate_values = {comparison_key(v) for raw_alt, _, _ in alternate_candidates for v, _ in [self._validate(field, raw_alt)] if v is not None}
            if alternate_values and (value is None or alternate_values != {comparison_key(value)}):
                issues.append("ocr_segmentation_passes_disagree")
                score = min(score, .55)
                value = None
        critical = {"mrz_check_digit_failed", "mrz_composite_check_digit_failed", "mrz_optional_data_check_digit_failed", "aadhaar_leading_digit_not_valid_for_issued_number", "checkbox_selection_requires_visual_review", "two_digit_year_requires_review", "numeric_date_assumed_day_first"}
        if any(issue in critical for issue in issues):
            score = min(score, .82)
        if field in IDENTIFIER_FIELDS and issues:
            score = min(score, .88)
        if handwritten:
            issues.append("handwritten_form_requires_review")
            score = min(score, .78 if field in HIGH_RISK_HANDWRITTEN else .84)
            if image_ocr:
                issues.append("tesseract_handwriting_not_validated")
                score = min(score, .65 if field in HIGH_RISK_HANDWRITTEN else .72)
        score = round(max(0.0, score), 3)
        # A nearly unreadable handwritten OCR token is evidence to inspect, not a
        # usable field value. Keep the raw source in the review report.
        if handwritten and image_ocr and score < .35:
            issues.append("handwritten_ocr_evidence_too_weak_for_candidate_value")
            value = None
        review = value is None or score < self.review_threshold or bool(set(issues) & critical) or handwritten or len(values) > 1
        return {"value": value, "confidence": score, "status": "unreadable" if value is None else ("needs_review" if review else "accepted"), "source": _source_evidence(line, raw, method), "validation_issues": list(dict.fromkeys(issues)), "needs_review": review}

    def _process(self, lines: list[_Line], hint: str | None, source_name: str, handwritten: bool | None, ocr: dict[str, Any]) -> dict[str, Any]:
        text = "\n".join(line.text for line in lines)
        detected, confidence, evidence = self.classify(text)
        doc_type = hint or detected
        reasons = []
        if hint:
            reasons.append("document_type_supplied_by_user")
            if detected and detected != hint:
                reasons.append("document_type_hint_conflicts_with_detected_content")
                confidence = min(confidence, .50)
            elif detected is None:
                confidence = 0.0
        elif doc_type is None:
            reasons.append("document_type_unknown_or_ambiguous")
        if confidence < self.review_threshold:
            reasons.append("classification_below_review_threshold")
        alternate_lines = ocr.pop("_alternate_lines", [])
        if doc_type:
            is_handwritten = handwritten if handwritten is not None else doc_type in HANDWRITTEN_TYPES
            candidates = self._extract(lines, doc_type)
            alternate = self._extract(alternate_lines, doc_type) if alternate_lines else {}
            fields = {field: self._field_result(field, candidates[field], is_handwritten, ocr["is_image_ocr"], alternate.get(field)) for field in DOCUMENT_SCHEMAS[doc_type]}
            reasons.extend(f"{field}:{item['status']}" for field, item in fields.items() if item["needs_review"])
        else:
            is_handwritten = handwritten
            fields = {}
        if not lines:
            reasons.append("no_readable_document_text")
        if any(w.startswith("read_or_ocr_failed") for w in ocr.get("warnings", [])):
            reasons.append("file_read_or_ocr_failed")
        return {"source_file": source_name, "document_type": doc_type or "unknown", "document_type_label": DOCUMENT_LABELS.get(doc_type, "Unknown"), "classification_confidence": confidence, "classification_source": "user_hint" if hint else "content_rules", "classification_evidence": evidence, "handwritten": is_handwritten, "fields": fields, "review_threshold": self.review_threshold, "confidence_kind": "heuristic_quality_score_not_probability", "needs_review": bool(reasons and (len(reasons) > 1 or reasons[0] != "document_type_supplied_by_user")), "review_reasons": reasons, "ocr": ocr}

    def review_report(self, results: Iterable[dict[str, Any]]) -> dict[str, Any]:
        return review_report(results, self.review_threshold)


def review_report(results: Iterable[dict[str, Any]], review_threshold: float = .90) -> dict[str, Any]:
    """Return a review queue with each failed/missing target, not just doc flags."""
    if not 0 < review_threshold <= 1:
        raise ValueError("review_threshold must be greater than 0 and at most 1")
    results = list(results)
    documents = []
    flagged_fields = []
    for index, result in enumerate(results):
        source = result.get("source_file", f"document_{index + 1}")
        threshold = result.get("review_threshold", review_threshold)
        for field, item in result.get("fields", {}).items():
            if item.get("needs_review", item.get("confidence", 0) < threshold) or item.get("value") is None:
                flagged_fields.append({"source_file": source, "document_type": result["document_type"], "field": field, "value": item.get("value"), "confidence": item.get("confidence", 0.), "threshold": threshold, "status": item.get("status"), "below_threshold": item.get("confidence", 0.) < threshold, "reasons": list(dict.fromkeys(item.get("validation_issues", []) + (["confidence_below_threshold"] if item.get("confidence", 0.) < threshold else []))), "source": item.get("source")})
        if result.get("needs_review"):
            documents.append({"source_file": source, "document_type": result["document_type"], "classification_confidence": result.get("classification_confidence", 0.), "reasons": result.get("review_reasons", [])})
    return {"review_threshold": review_threshold, "threshold_rationale": f"{review_threshold:.2f} is the configured heuristic triage threshold for identity and bank data; it is not a calibrated {review_threshold:.0%} accuracy claim. Unreadable, conflicting, invalid, uncertain checkbox and handwritten fields are reviewed even if a configured threshold is lower. Each field entry preserves the threshold used during its extraction; mixed-threshold results are not rescored. Ground-truth samples are required for calibration.", "document_count": len(results), "documents_needing_review": len(documents), "fields_needing_review": len(flagged_fields), "documents": documents, "fields": flagged_fields, "confidence_kind": "heuristic_quality_score_not_probability"}
