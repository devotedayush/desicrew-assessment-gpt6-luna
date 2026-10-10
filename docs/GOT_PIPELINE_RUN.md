# Reproduce the guarded local GOT pipeline

This optional Linux CLI mode stages printed OCR before loading the model, then runs local GOT in a supervised worker with a 600-second hard deadline and 4096MB measured-RSS watchdog. It uses explicit supplied-layout value-line crops. Every candidate requires review. No private visual-reference value enters recognition or extraction.

## Setup

```bash
python -m venv .got-venv
.got-venv/bin/pip install -r requirements.txt -r requirements-got-ocr.txt
.got-venv/bin/python scripts/evaluate_got_ocr.py --download-model --download-only
```

Setup downloads only pinned public official config, tokenizer and safetensors files and verifies hashes. No account token is needed. Weights are excluded from the source package and inference never downloads them.

Place only the ten requested target images under `data/private/target_images/`. One all-document GOT invocation can exceed the 600-second guard on CPU. Create a private printed-OCR stage with the guarded command below; if its model phase times out, its `tesseract_stage_results_private.json` remains reusable. The current first run processed 15 of 26 crops before the deadline. Its incomplete model predictions were not used in the final result.

```bash
.got-venv/bin/python scripts/demo_documents.py \
  --input-dir data/private/target_images \
  --output-dir outputs/private_samples/got_initial \
  --handwriting-backend got --handwriting-model data/private/models/got-ocr2 \
  --use-layout-regions --detect-checkboxes \
  --batch-timeout 600 --memory-budget-mb 4096
```

Copy the six handwritten images into two private three-image directories: `ECS.jpeg`, `Fatca.jpeg`, `Illustration.jpeg` in A; `Moral.jpeg`, `split.jpeg`, `suitability.jpeg` in B. Run the batches sequentially with fresh output directories and the same Tesseract stage:

```bash
for batch in a b; do
  .got-venv/bin/python scripts/demo_documents.py \
    --input-dir "data/private/got_batch_$batch" \
    --output-dir "outputs/private_samples/got_batch_$batch" \
    --handwriting-backend got --handwriting-model data/private/models/got-ocr2 \
    --use-layout-regions --detect-checkboxes \
    --staged-results outputs/private_samples/got_initial/tesseract_stage_results_private.json \
    --batch-timeout 600 --memory-budget-mb 4096
done
.got-venv/bin/python scripts/combine_got_batches.py \
  --stage outputs/private_samples/got_initial/tesseract_stage_results_private.json \
  --batch outputs/private_samples/got_batch_a/results.json \
  --batch outputs/private_samples/got_batch_b/results.json \
  --output-dir outputs/private_samples/got_combined
```

The combiner accepts only completed guarded batches covering all six handwritten types. It takes the four printed documents from the staged pass and does not read the visual reference. The stage-reuse gate checks private path, source filename and local Tesseract engine. The prior stage has no original-image hash, so this is same-copy procedural provenance, not cryptographic validation of its original inputs. The six batch images were separately checked byte-identical to their current target-image copies, with hashes recorded privately. Use spaces in `--batch-timeout 600` and `--memory-budget-mb 4096`; those are the maximum permitted limits. Do not mix handwritten Tesseract crop OCR into this run.

The supplied-layout templates contain geometry and all possible checkbox labels, with no selected answer, private path or transcription. They require alignment inspection and can fail on other layouts. Unsupported PDFs go to review.

## Actual evidence and limits

The current Linux VM runs completed 13/13 crops each: A took 474.551 seconds with 3991.7MB peak supervised RSS; B took 457.814 seconds with 4009.5MB. The combined ten-document JSON emits all 44 fields, all review-required. On the current independent 42-readable-reference diagnostic, it matched 18/42, with 23 null and one incorrect non-null field. All 26 GOT generations ended at the per-crop soft generation budget. On the same references, optional local Mac Vision matched 31/42, so it is the recommended reviewer output for this supplied set. Actual resource reports, model outputs and field diagnostics stay private; `artifacts/q3_evaluation_summary.json` contains aggregate counts only.

The older imported checkpoint diagnostic reported 27/41 readable matches, three excluded references, 250.3 seconds and 4047.4MB for 26 crops. Its raw per-document results are unavailable for rescoring against the current 42 readable references. These historical figures are not the current run and must not be used to claim an improvement or regression across reference sets. Neither set is an official benchmark, calibrated confidence or authenticity verification.

GOT is exposed through this guarded CLI; production UI wiring is not claimed. The ordinary Streamlit TrOCR mode is separate. Docker and macOS GOT execution have not been verified. Source scripts create local JSON and review reports without external submission.
