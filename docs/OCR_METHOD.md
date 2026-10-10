# Local classification, extraction and human review

> **Historical implementation notes.** This document was imported from an earlier local checkout. Its test count and evaluation observations are snapshots, not results of the current source package. For current ten-image measurements and the recommended local backend, see `docs/Q3_LOCAL_EVALUATION.md` and `artifacts/q3_evaluation_summary.json`.

## Scope and truthfulness

`desicrew/documents.py` implements the ten types and all **44 extraction targets** in `BRIEF.md`. Every recognized or human-hinted document returns every field in its schema, including explicit `null` for missing, unreadable, invalid or conflicting values. The six form schemas are treated as potentially handwritten by default.

This is a **local-only, conservative baseline**, not a demonstrated high-accuracy handwriting model. No identity, bank or insurance document is uploaded. There are no hosted OCR, LLM or paid API calls. Tesseract, Pillow and PyMuPDF are the default OCR/rendering dependencies. An optional CPU-only local handwriting backend uses separately installed model dependencies and already-downloaded official weights; its runtime never uploads a crop or fetches weights. The supplied set has now been visually inspected and processed locally: ten required images and two extra scanned PDFs. No supplied answer key has been scored. Synthetic fixtures are kept separate and must not be presented as supplied samples.

## Processing

1. Recognize PDF, PNG, JPEG, TIFF, BMP, WebP and plain-text transcripts
2. Prefer native PDF text when sufficient readable text exists. Pages containing raster imagery are also rendered and OCRed, even when a native text layer is present. This avoids relying only on a printed form's searchable labels while missing handwritten image content
3. Render scanned PDF pages locally at 300 DPI. Respect a 20-page limit, a 40-million-pixel per-page/frame limit and a cumulative 60-million-pixel budget across PDF renders/TIFF frames. The cumulative budget is checked before additional raster allocation. Correct EXIF orientation, composite transparency over white, convert to grayscale and apply autocontrast. Enlarge small scans within a bounded pixel budget. Original files are not modified
4. Run local Tesseract twice, with page-segmentation modes 6 and 11, requesting TSV text, word confidence and geometry. The higher mean word-confidence run supplies the primary transcript. Differences in valid field candidates between segmentation runs cause an explicit conflict flag and `null`, rather than silently choosing an ID
5. Classify using weighted document-title/content rules. An ambiguous tie returns `unknown`, not a forced type. The optional human type hint is recorded and does not manufacture classification confidence. Filenames are not classification evidence
6. Match labels and strict-format fallbacks, including bilingual label prefixes, typographic apostrophes and spatially adjacent sparse-OCR values; collect page, line, recognized text, extraction method, OCR confidence and bounding box provenance. Multiline addresses and checkbox options retain every contributing line and use their minimum confidence rather than the first line alone. Preserve leading zeros and identifier strings. Normalize valid dates to ISO format while retaining the original in provenance
7. Validate formats and build a field-level human-review queue

Tesseract's official [quality guidance](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html) discusses 300-DPI input, skew, preprocessing and segmentation choices. The [command-line reference](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html) documents TSV geometry/confidence and page-segmentation modes. PDF rendering uses the local APIs in [PyMuPDF's image documentation](https://pymupdf.readthedocs.io/en/latest/recipes-images.html). Automatic perspective correction and deskew are not implemented; EXIF orientation correction is not equivalent to deskew.

## Handwritten text is treated differently

Tesseract is used as a printed-text baseline. It is not assumed to reliably read the supplied handwriting. For the six handwritten-form types, every extracted field remains in the review queue even when OCR emits a plausible, format-valid string. On image OCR the confidence cap is **0.65** for handwritten IFSC, TIN/PAN, account numbers, dates and places, and **0.72** for other handwritten targets. For provided text transcripts the corresponding caps are **0.78** and **0.84**. These caps deliberately prevent a confidently recognized printed label from hiding uncertainty in its handwritten value.

The default implementation does not download a handwriting model. `DocumentProcessor(handwriting_backend=..., roi_manifest_path=...)` optionally invokes the local recognizer on explicit value-line crops only. An operator supplies original-image bounding boxes in a local manifest; this is template/ROI-assisted extraction, not automatic crop detection. No reference transcription is sent to the recognizer. Model candidates carry crop/page/model provenance and mandatory `handwritten_model_not_calibrated` flags. They cannot erase an existing OCR conflict or infer checkbox selection. A supplied answer key and independent calibration remain necessary before claiming handwriting accuracy.

Specific safeguards:

- IFSC/PAN/TIN/account characters are never automatically changed from `O` to `0`, `I` to `1`, `S` to `5` or similar
- Unreadable/malformed identifiers return `null`; raw candidates remain review evidence. A readable twelve-digit dummy Aadhaar value is preserved literally even if its leading digit would be invalid for an issued card, with a mandatory validity-review flag
- Account, application and policy numbers stay strings, preserving leading zeros
- MRZ Line 2 preserves characters, fillers and leading zeroes; OCR-inserted whitespace is removed while the original transcript remains in provenance. Mandatory numeric check digits at TD3 positions 10, 20, 28 and 44, optional-data check digits and the composite checksum are validated. Missing mandatory digits return `null`. A malformed or checksum-failed MRZ cannot supply derived passport data; a valid MRZ does not prove identity authenticity. The optional-data filler exception follows [ICAO Doc 9303 Part 4](https://www.icao.int/publications/documents/9303_p4_cons_en.pdf)
- Indian passport/PAN/IFSC pattern checks are structural checks, not issuer/bank lookups. Aadhaar checksum, issuer authenticity, bank ownership and external IFSC directories are not verified. TIN/DL variants need jurisdiction-aware verification
- Numeric dates use day-first interpretation in the Indian form context; dates where both day and month are at most 12 are explicitly flagged. Two-digit years return `null` because their century is ambiguous
- Checkbox reasons are extracted only from explicit selected markers such as `[x]` or `☑`, with exactly one selected option. Multiple selections, empty boxes and unlabeled marks are uncertain. Every extracted checkbox selection still requires visual review. An optional local pixel heuristic inspects explicitly templated box interiors, removes borders/long rules and ignores tiny disconnected speckles. It returns null for multiple or ambiguous marks; it is not a general checkbox detector

## Field confidence and threshold

The default threshold is **0.90**. It is a conservative heuristic triage cutoff for identity/banking data, not a calibrated 90% correctness probability. A labeled same-line candidate starts at 0.96; a next-line candidate at 0.86; strict unlabelled shape matches at 0.84; and a check-digit-validated MRZ path can start at 0.91–0.92. OCR field evidence is limited by the **lowest token confidence across every contributing source line**, so highly confident labels cannot average away a weak identifier. Validation concerns and handwriting caps reduce the score further.

All missing, unreadable, conflicting, invalid, ambiguous-date, uncertain-checkbox and handwritten targets require review even when the configured threshold is lower. The report includes the configured threshold/rationale, every flagged field, its value or `null`, confidence, whether it is below threshold, reasons and source evidence. Mixed-threshold batches preserve each extraction's threshold and do not silently rescore it.

No accuracy percentage is reported without a ground-truth evaluation. A plausible-looking identifier can be wrong despite passing a regex; confidence is therefore explicitly described as `heuristic_quality_score_not_probability` in JSON.

## Reproducible checks and observed cases

Run from the project root:

```bash
.venv/bin/python -m unittest discover -s tests -p test_documents.py -v
.venv/bin/python scripts/demo_documents.py
.venv/bin/python scripts/demo_documents.py --ocr-images --output-dir outputs/documents_synthetic_ocr
```

The current focused suite completes **78 tests**, including local synthetic printed-PNG OCR, a native PDF and a blank scan. The synthetic transcript demo emits ten JSON files and 44 field objects. The separate synthetic printed-image demo performs real local Tesseract calls. Both are engineering checks on artificial templates, not handwriting or supplied-sample accuracy measurements. All fixture pages visibly state that they are synthetic.

Constructed failure cases that were exercised:

- An empty/blank scan produced a review result and explicit missing/null fields when its type was supplied
- `DEMOO123456` was rejected as an IFSC; its `O` was not silently repaired into the required zero
- An account beginning with letter `O` was rejected rather than rewritten as an account starting with zero
- Two plausible account numbers or disagreement between segmentation passes produced `null` plus a conflict reason
- An impossible date (`31/02/1994`) returned `null`; `03/04/1994` remained reviewable with the day-first assumption disclosed
- Two selected checkbox reasons or a list of unselected boxes returned `null`
- MRZ length/check-digit failures were flagged, without adding or removing fillers
- A parser edge case initially truncated `Example Bank` at a second `Bank` token; the label filtering was corrected and now has a regression test

Known limitations include multilingual scripts beyond installed English, complex table layouts, skew/perspective, unfamiliar labels, pixel-only checkbox marks and weak or absent titles. Actual supplied form scans exposed missed or corrupted handwritten identifiers, dates and place names; a plausible-looking candidate is not evidence of correctness. The initial Tesseract run also missed an OCR-damaged title and falsely matched an extra proposal form from an incidental ECS mention. Title/label/spatial handling was improved with regressions; unsupported title guards keep extra forms outside the ten target schemas. Printed synthetic PNGs cannot establish robustness on any of these. The current parser does not guess unlabeled names/addresses from prose and does not infer a person or place from insurer defaults.

## Authorized supplied-sample run

Process separately authorized local samples with:

```bash
.venv/bin/python scripts/demo_documents.py --input-dir data/private/samples --output-dir outputs/private_samples
```

Review every flagged field against the actual image and retain the JSON provenance. Report actual observed failures separately from the synthetic checks. Keep private source documents/results out of a public repository or demo bundle unless specifically authorized. Publishing, deploying or submitting this work is not part of the local demo command.

### Optional local handwriting command

The backend requires an explicit local model directory and a private ROI manifest. It never downloads weights automatically:

```bash
.venv/bin/python scripts/demo_documents.py --input-dir data/private/samples --output-dir outputs/private_samples/with_handwriting --handwriting-model /path/to/local/official-model --roi-manifest /path/to/private/roi-manifest.json
```

Keep ROI manifests, crops, model candidates and visual reference transcriptions private. Human visual transcription is separate QA material and is never substituted for automatic pipeline output. Non-null candidate counts are coverage counts, not accuracy measurements.

## Supplied-layout ROI and checkbox API

`desicrew.regions.SAMPLE_REGION_TEMPLATES` contains only normalized geometry, aspect ratios, field keys and all possible checkbox-option labels. It contains no names, identifiers, reference answers, private paths or selected answers. Its 42 text regions plus two checkbox target definitions are narrow aids for the supplied layouts.

```python
from desicrew.documents import DocumentProcessor
from desicrew.regions import SAMPLE_REGION_TEMPLATES

processor = DocumentProcessor(
    region_templates=SAMPLE_REGION_TEMPLATES,
    use_printed_region_ocr=True,
    use_handwritten_region_ocr=False,
    handwriting_backend=optional_local_backend,
    use_checkbox_detection=True,
)
```

All template-derived candidates require alignment review. Uploaded temporary paths are supported by normalized coordinates with an aspect-ratio guard. An explicit private manifest instead matches an exact original path, or basename plus detected type plus exact image dimensions; ambiguous matches are skipped. The default processor does not apply templates unless explicitly enabled.

The optional checkbox heuristic computes marks from pixels only. Template data lists every option, never which option is selected. It can be confused by handwriting, rules, shadows, low resolution and misplaced boxes, so its selected fields are capped at 0.55 and always flagged. The full option description is returned when available, with short option labels retained as metadata. Synthetic tests cover empty, ticked and filled boxes, horizontal rules, speckles, ambiguous marks and multiple selections.

## Independent private visual-reference diagnostic

A separate actual-pixel inspection checked all 44 targets without reading automatic extraction outputs first. It found 41 readable references and conservatively excluded three ambiguous IFSC/date regions. Reference values never fill pipeline output. These are local visual references, not the official assessment answer key, and no authenticity or general accuracy claim is made.

The strongest completed conservative configuration combines printed-region Tesseract, local TrOCR-base handwritten crops and optional checkbox pixels. Among the 41 readable references it has **21 normalized final-value exact matches**, **26 fields with a correct raw candidate**, **five correct raw candidates kept null**, **12 null final values** and **eight incorrect non-null review candidates**. All 44 targets still require review. Correct raw extraction can be rejected appropriately, for example when a visibly printed MRZ has nonstandard length or when OCR passes conflict.

Literal exact matches, explicit formatting normalization and configuration comparisons are in `DOCUMENTS_DIAGNOSTIC_SUMMARY.json`. It contains aggregate counts only. Private references, field diagnostics, source pixels and automatic extraction values are kept separately under private output. More non-null candidates do not imply more correct extraction; small-model and destructive-cleanup experiments demonstrate that limitation.

`python scripts/evaluate_documents.py --references /path/to/private/references.json --results configuration=/path/to/automatic/results.json` compares outputs read-only and records final-value matches separately from rejected raw candidates. Its regressions verify that references do not mutate/fill outputs, repair identifier characters or guess a date century.

## Latest completed guarded GOT integration

Printed-localization refinements recover three held values, raising the earlier21/41 diagnostic to 24/41. The same downloaded GOT checkpoint then processes26 canonical handwritten crops under the unchanged4GB/600s guard. The completed combination matches 27/41 normalized readable references, includes31 correct raw-candidate fields, leaves four correct raw fields conservatively null, nine null final values and five incorrect non-null review candidates. All44 targets remain flagged. These are local visual-reference diagnostics, not official-answer-key accuracy.

See `GOT_PIPELINE_RUN.md` for the explicit reproducible command. Runtime never downloads weights or sends private pages to a provider.
