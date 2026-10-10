# Bounded local GOT-OCR2 alternative diagnostic

> **Historical checkpoint diagnostic.** The 18-crop and 27/41 findings below came from an earlier local workspace and reference revision. They are retained as provenance, not current-run scores. Current ten-image scores and bounded-run status are in `docs/Q3_LOCAL_EVALUATION.md` and `artifacts/q3_evaluation_summary.json`.

This optional backend is experimental and is not selected automatically by the
application. It does not establish robust extraction of handwriting, and every
prediction requires human review. No result below is official-answer-key accuracy.

## Verified source and isolation

- Official StepFun checkpoint: [stepfun-ai/GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
- Pinned revision: `d3017ef2c2c1395888c8d635c5e0508bcb0ac78d`
- License reported by the official model card: Apache-2.0
- Native implementation: [Transformers GOT-OCR2](https://huggingface.co/docs/transformers/model_doc/got_ocr2)
- Seven allowlisted public config/tokenizer/safetensors files only; no custom
  model-repository Python, unsafe pickle weights, credentials, or hosted inference
- Official safetensors size: 1,121,114,488 bytes
- Official safetensors SHA-256:
  `6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4`
- Official metadata hashes verified for every file: Git blob SHA-1 for small JSON
  files and LFS SHA-256 for the tokenizer and weights. A local SHA-256 manifest is
  checked again before model loading
- Separate `.got-venv`; the existing `.hwr-venv` and `.qa-venv` were not modified
- Runtime: torch 2.6.0+cpu, transformers 4.51.3, safetensors 0.4.5,
  accelerate 1.6.0, Pillow 11.3.0
- `local_files_only=True`, `trust_remote_code=False`, `token=False`,
  `use_safetensors=True`, CPU BF16 and two CPU threads

## Scope and safety

One inference configuration on the same 18 independently inspected handwritten
value-line crops used by the TrOCR diagnostic. Seventeen line references are
readable; one crowded date reference is excluded. Some fragments are outside the
brief's required targets. Repeated identifiers/names/places limit diversity.

The model receives only the crop pixels and the native plain OCR prompt. Visual
references are accessed only after inference to score the prediction. They never
fill or repair outputs. Comparison collapses whitespace and ignores case; it does
not substitute identifier characters, normalize dates, or invent missing text.

Generation is greedy, limited to 64 new tokens and a 20-second soft generation
deadline per crop. The batch has a 600-second hard process deadline and a 4,096 MB
measured-RSS watchdog. Soft generation limits do not guarantee a hard encoding or
loading limit. Partial predictions, actual field values, reference values, logs,
and process resource evidence stay within `outputs/private_samples`.

All predictions carry a capped, explicitly uncalibrated heuristic quality score,
review flag, validation issues, crop coordinates and model revision. A completed
generation is not evidence that its text is correct. Zero private image uploads,
zero paid API calls, no deployment, push or submission.

## Completed 18-crop result

The authorized same-checkpoint memory-recovery run **completed all 18 crops**
under the original 4,096 MB/600-second limits. Its aggregate result is separate
from the earlier failed prefix:

- 17 readable references, one ambiguous reference excluded
- **11/17 case/whitespace exact line matches**
- 31 edits across 179 reference characters; CER 0.1732
- Zero generation truncations; all 18 predictions require review
- 169.169 seconds total inference, median 9.364 seconds/crop,
  slowest 12.298 seconds/crop
- 173.438 seconds worker time including load; 174.720 seconds supervised wall
- Sampled peak process RSS: 4,015.5 MB; OS worker high-water RSS: 4,067.3 MB
- Maximum RSS after per-crop cleanup: 1,603.609 MB
- All eight raw predictions from the original prefix were identical in the rerun

The OS high-water leaves only 28.7 MB below the 4,096 MB cap. Completion on this
small fixed crop set is not evidence of comfortable memory headroom on other
inputs. Keep the guard; do not raise it silently or treat this as a production
capacity benchmark.

GOT outperformed the completed TrOCR-base raw line diagnostic on the same crops:
11/17 versus 5/17 exact matches, and CER 0.1732 versus 0.4972. It took 169.169
seconds versus 24.769 seconds of inference. GOT is the strongest **measured line
candidate** here, with substantially greater latency and memory requirements.

Six readable lines still missed strict comparison: account number, IFSC,
father's name, noisy place, nominee relationship, and one date. The TIN target was
not present in this selected crop set and cannot be claimed evaluated. Some
formatting-only mismatches may differ under the full pipeline's field-specific
normalization; this report does not repair characters or fill fields from gold.

Before GOT target-field integration, the completed recommended hybrid pipeline
had 19/41 normalized final matches and 24 correct raw-candidate fields; its
TrOCR-base component had 19/41 and 23 respectively. A line recognizer's 11/17
score is not a new 41-field pipeline score. Keep human review and evaluate any
future routing/integration independently. This comparison did not change the
production extractor.

### Original memory failure

Actual native offline inference succeeded, but **the full diagnostic failed its
memory budget**. The measured-RSS watchdog stopped the worker after eight of the
planned 18 crops, at 4,111.7 MB against the 4,096 MB cap. Exit code was -9; wall
time was 82.674 seconds. This original failed prefix remains separate from a
subsequently authorized same-checkpoint memory-recovery rerun.

Completed-prefix results only:

- Eight readable completed crops, four case/whitespace exact line matches
- 27 character edits across 84 reference characters; completed-prefix CER 0.3214
- Zero generation truncations among the eight completed crops
- 74.764 seconds total completed-crop inference, median 9.361 seconds/crop,
  slowest 10.542 seconds/crop
- All eight predictions require human review
- Ten crops were not completed. Dates and TIN were not established by this run

The completed amount, policy/application identifier and short policyholder-name
lines matched. Account number, IFSC, father's-name and noisy-place lines were not
case/whitespace exact matches. These are diagnostic observations, not claims of
identifier validity. Field-specific whitespace-only normalization may differ from
the stricter line metric; no character repair was attempted.

The runtime emitted the native warning that sliding-window attention is not
implemented for SDPA. The saved slow image processor was used; final source now
selects it explicitly with `use_fast=False`, preserving this run's default
behavior. No alternate attention mode, preprocessing sweep, memory-limit increase
or whole-form trial was attempted.

### Authorized memory recovery

The same 18-crop comparison was explicitly authorized for one bounded rerun after
safe CPU memory cleanup. The checkpoint, crop set, greedy decoding, BF16 math,
thread count, 64-token/20-second limits, 600-second wall deadline and 4,096 MB
watchdog are unchanged. The old partial prefix is not counted as a complete run.

`output_scores=False` was already set, so the fix does not pretend that retained
scores were the proven cause. Unused returned generation metadata can contain KV
cache references; the revised call returns only token IDs, disables scores,
logits, attentions and hidden states, and drops per-crop tensors in a `finally`
block. Garbage collection and optional [standard glibc free-page trimming](https://man7.org/linux/man-pages/man3/malloc_trim.3.html)
then release unreachable objects and already-free allocator pages. The worker's
process-only [allocator-arena limit](https://man7.org/linux/man-pages/man3/mallopt.3.html)
is two; it does not alter model math or increase the memory budget. Per-crop RSS
before/after cleanup is recorded privately.

The memory-recovery run's completed result is reported above. There were no new
checkpoints, preprocessing sweeps, whole-form trials or budget increases. The
failed original prefix cannot be combined with the rerun to enlarge the sample.

Evidence is saved privately in `partial_predictions_private.json`,
`partial_summary_without_field_values.json` and `process_resource_report.json`
under the original `got_ocr2` diagnostic directory. The completed result is
separate, in `got_ocr2_memory_retry/got_line_evaluation_private.json`,
`summary_without_field_values.json` and `process_resource_report.json`. The
original failed directory deliberately has no successful full-evaluation report.
The alternative remains optional and experimental.

## Verification

- Twenty-one focused adapter/privacy/budget/memory-release tests passed
- Full source test snapshot: 205 passed, five existing SWIG deprecation warnings
- Native GOT imports and optional-environment `pip check` passed
- Syntax compilation passed
- Original real attempt failed its memory guard after eight crops; one explicitly
  authorized same-checkpoint recovery completed 18 crops below the unchanged cap
- No deployment, push or submission

## Reproduction

Install the optional environment separately, then explicitly download only the
approved pinned public files. The inference command works from the repository root
after private inspected crops and the manifest are present locally:

```bash
python3 -m venv .got-venv
.got-venv/bin/pip install -r requirements-got-ocr.txt
.got-venv/bin/python scripts/evaluate_got_ocr.py --download-model --download-only
.got-venv/bin/python scripts/evaluate_got_ocr.py --limit 18 --threads 2 \
  --batch-timeout 600 --memory-budget-mb 4096
```

The application does not download weights or invoke this alternative backend.
Use a fresh private `--output-dir` for any explicitly requested repeat; the script
refuses to mix or overwrite evidence from an existing trial.
Do not include model weights, optional environments, crop pixels, manifests with
references, or private output JSON in a public source package.
