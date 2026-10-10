# Q3 supplied-file local evaluation (9 October 2026)

The ten supplied target images were processed locally with Tesseract, printed
value-region OCR, and pixel-based checkbox inspection. An optional macOS Vision
recognizer was also measured on explicit handwriting crops. The two extra PDFs were
inspected by the same classifier and returned `unknown`; they are outside the
ten requested document types. No identity or banking image was sent to an
external model. Each of the 44 requested targets is present in the JSON schema.

The portable base run writes `artifacts/private/q3/final/results.json`, ten per-document
JSON files, and `review_report.json`. The optional macOS Vision run writes the same
forms under `artifacts/private/q3/vision_verified/`. All ten documents and all 44 fields require
human review at the 0.90 heuristic threshold. These scores are triage signals,
not probabilities of correctness. Missing or weak handwritten readings stay null
with OCR evidence attached; the reviewer must inspect the original image.

An independent visual transcription of the source pixels gives 42 readable
references and two excluded ambiguous references. The handwritten TIN/PAN and
bank account were resolved by enlarging the original pixels. The portable base
automatic values match 18/42 after the evaluator's declared normalization:
18 correct non-null, zero incorrect non-null, and 24 null values. This is a
small, non-blind diagnostic against manually inspected pixels, not the official
answer key or a general accuracy estimate. The private references and field
details are in `data/private/q3_visual_references.json` and
`artifacts/private/q3/current_comparison/visual_reference_comparison_private.json`.
Only aggregate counts should leave the private workspace.

The unmodified printed-region baseline matched 17/42, with two wrong non-null
handwritten tokens and 23 null values. A handwritten-region Tesseract variant
matched 18/42 but raised wrong non-null values to six, so it was rejected.
The final source accepts a printed name when two local OCR segmentations and
the chosen page candidate agree despite a conflicting alternate page pass.
It also retains extremely weak handwritten OCR text as review evidence while
setting the field value to null. These general rules yield the portable base 18/42 and
zero wrong non-null diagnostic result. Checkbox selections are local pixel
heuristics and remain flagged for visual confirmation.

The optional macOS Vision crop run matched **31/42** readable references with
eight null and three incorrect non-null values. The three errors were a father's
name character, an application-number character, and a place-name character.
All three are explicitly flagged. The macOS OS framework ran locally, with
language correction disabled; no private image upload or model download was
used. The bridge dependencies are pinned in `requirements-macos-vision.txt`.
Vision's reported recognition score is uncalibrated here, so the adapter assigns
a conservative 0.35 candidate quality score and refuses to choose when an
explicit crop contains multiple text lines. For bank account digits it also
requires agreement between two crop scales; disagreement leaves a null value
with both readings retained for review. This backend is available through the
Mac CLI and local Streamlit document interface; it does not make the Linux
application depend on macOS.

The optional guarded Linux GOT run completed all 26 explicit handwritten
value-line crops in two 13-crop batches. Its ten-document result matched
**18/42** readable references, with 23 null and one incorrect non-null value.
All 26 generation calls ended at their soft generation budget; this checkpoint
did not improve the current extraction over the portable baseline. Batch A
finished in 474.551 seconds with 3991.7MB peak supervised RSS; batch B in
457.814 seconds with 4009.5MB. Both were below the unchanged 600-second and
4096MB limits. All 44 fields remained review-required. The combined private
output and per-batch resource evidence are under `outputs/private_samples/`.
The pinned 1.1GB official weights and optional runtime were installed privately
in the local Linux VM for this run. The Vision run remains the recommended
local reviewer output on the current inspected samples because it has more
correct readable values, while its three incorrect values show why review is
mandatory.

The historical guarded GOT run reported 27/41 readable matches on a separate
private reference set with three exclusions. Its prior per-document results
are unavailable for rescoring on the current 42-readable-reference set, so
that historical figure is not directly comparable. The guarded current CLI,
batches and limits are documented in `docs/GOT_PIPELINE_RUN.md`.

To reproduce the current artifact in a prepared local environment, place only
the ten target images in `data/private/target_images/`, then run:

```bash
python scripts/demo_documents.py --input-dir data/private/target_images \
  --output-dir artifacts/private/q3/final --use-layout-regions --detect-checkboxes
python scripts/evaluate_documents.py \
  --references data/private/q3_visual_references.json \
  --results final=artifacts/private/q3/final/results.json \
  --output-dir artifacts/private/q3/current_comparison
```

For the measured Mac path, install `requirements-macos-vision.txt` in a local
Python 3.12 environment and run the same document command with
`--handwriting-backend vision --output-dir artifacts/private/q3/vision_verified`.
The same Vision option was also exercised through the actual local Mac UI.
The source PDFs and images never leave the machine. The separate local
evaluation command can compare all three result files without feeding references
into OCR.

The reference file is read only by the separate evaluator. It never fills
the extraction output. The `--use-layout-regions` templates contain geometry,
not expected field values, and are suitable only for this supplied layout.
