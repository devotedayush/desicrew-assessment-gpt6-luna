# Verification record — 9 October 2026

This report separates current local runs from historical evidence imported in
the original source archive. The final counts and clean-package audit are in
`artifacts/verification/`; exact architecture and limitations are in
`SUBMISSION_REPORT.md`.

## Current environment

Q1 runs in a real Ubuntu 24.04 aarch64 Linux VM on this Mac, using Lima's VZ
virtualization, Linux 6.8, Python 3.12.3, Tesseract 5.3.4 and libseccomp 2.5.5.
The unchanged runner requires the `linux-seccomp` response marker and retains
filesystem/network/process denial, AST policy, resource bounds and minimal
child environment. There is no macOS execution fallback. Docker was not used
or tested. Mac Python 3.12 is used for the optional local Vision extraction
and Q2/Q3 browser capture.

## Original and revised source checks

The supplied source ZIP was extracted into a separate untouched baseline.
`python -m pytest -q` reproduced **205 passed**, five PyMuPDF SWIG deprecation
warnings, in **23.77 seconds** on Linux. This is a new reproduction of the
reported original count.

The first revised suite passed **217 tests** in 29.26 seconds. A preliminary
combined submission ZIP, extracted into a new folder, passed **221 tests** in
35.35 seconds after further inventory and Vision changes. Later focused
multi-sheet schema-binding tests also passed. The revised source passed **223 tests** in 22.63 seconds, and the clean extracted
submission passed **223 tests** in 34.11 seconds. The final rebuilt ZIP is checked
again from a fresh directory; exact byte hashes and that final confirmation are
in the adjacent `FINAL_VERIFICATION.json` and `SHA256SUMS.txt`.

`pip check` reported no broken requirements. Compilation with
`python -m compileall -q app.py desicrew scripts tests` passed. AppTest covers
all page callbacks and failure handling; it is not screenshot evidence.

## Q1

The supplied workbook has 46 rows after generic header detection. Controlled
code-execution checks and independent oracles are implementation verification,
not real-model accuracy. The new live-model evaluator independently uses
Python `math.fsum` for totals and sorting/source-ID/metric checks for ranking.
It never supplies expected values to the model or substitutes a deterministic
query for failed model code.

The original imported Qwen diagnostic was 3/4. A fresh initial run in this
checkout was **2/4**, failing stock-column whitespace and ranking direction.
The latest genuine run and its separate query/summary provenance are recorded
in `artifacts/inventory_model_final_qa_public.json` and `LOCAL_MODEL_QA.md`.
The narrow ranking validator is not a general semantic verifier.

An actual public definition lookup for the canonical term `reorder point`
returned three sources on Linux. macOS's Python CA configuration initially
failed; TLS verification was not disabled. No private rows are sent in public
searches.

## Q2

One assistant instance completes 12 sequential exchanges. The independent
expected-section/source-quote checks pass **12/12**, covering clear follow-ups,
repetition, topic changes, unsupported SAML, and an ambiguous-link clarification.
All responses cite document sections. The default method is deterministic
extractive retrieval over an original fictional KB. Role-permission and
password-reset wording regressions are tested.

## Q3

Automatic classification covers the ten supplied target images. The two extra
PDFs are `unknown` and outside the requested ten types. All 44 target slots,
per-field heuristic scores, null/error evidence and review reasons are emitted.
All supplied fields require review at 0.90; scores are not calibrated.

The local Mac Vision CLI and actual private AppTest interface run both match
**31/42 readable references after declared formatting normalization** (21
literal string matches), with **eight null and three wrong non-null values**.
The private reference set was refined from enlarged original pixels and is
non-blind; it is not an official answer key. Two ambiguous references are
excluded. The portable base Tesseract configuration matches 18/42 and emits
no wrong non-null values among the readable references. See
`Q3_LOCAL_EVALUATION.md` for handwritten/printed breakdowns and raw candidates.

The first fresh GOT arm64 run hit the unchanged 600-second hard deadline after
15/26 crops, with supervised peak RSS 4082.5MB below the 4096MB cap. Partial
recognition is not a completed accuracy measurement. Two subsequent 13-crop batches completed in 474.551 and 457.814 seconds,
with peak RSS 3991.7MB and 4009.5MB, within the unchanged guards. All 26
recognitions hit the per-crop soft generation budget. The combined configuration
matched 18/42 (23 null, one wrong non-null), and Vision remains selected. Historical 27/41 GOT
results remain separate from current runs and reference judgments.

## Genuine interface evidence

Browser screenshots were captured from the local running apps with installed
Chrome and Playwright. Synthetic data/fictional KB keep private values out of
public evidence. Q2 memory/citations and Q3 schema/real synthetic-image OCR,
unreadable-image review, and Q1 invalid-workbook handling are captured.

Visual review caught an initial capture assertion that treated `llm` mode as
success even with a null result, and stale document results during a rerun.
Both capture checks were strengthened to inspect downloaded JSON. The failed
model query is retained as failure evidence. A successful LLM screenshot must
match an independent row count; a failure is never reported as success merely
because the model toggle is on. See `SCREENSHOT_CAPTURE.md` and the capture
manifests for final evidence status and hashes.

## Packaging boundaries

`DesiCrew-Source.zip` contains source, pinned dependencies, documentation,
synthetic outputs, aggregate reports and genuine synthetic screenshots.
`DesiCrew-Submission-LOCAL.zip` adds only the ten real per-document extraction
JSONs, their batch JSON and review report. The combined ZIP contains sensitive
identity/banking values and must stay private. Supplied scans/workbook, visual
answer references, crops, credentials, environments and weights are excluded.
An interview guide is not included. No email, publication or submission occurred.
