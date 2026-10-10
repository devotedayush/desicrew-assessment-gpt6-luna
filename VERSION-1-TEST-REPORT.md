# Version 1 verification against Questions.docx

Tested on 10 October 2026. This is the copy of the submitted original with the
existing Q1/Q2 LLM calls connected to GPT-6 Luna. Q3 remains unchanged.

## Overall result

All 230 automated tests passed on the supported Linux environment. Dependencies
passed pip check. Live GPT-6 Luna requests, live public definition search, and
Streamlit interaction tests succeeded. No missing essential Q1/Q2/Q3 feature was
found in these checks. These finite tests do not establish correctness for every
possible question or document.

## Q1 inventory

Used the bundled synthetic workbook and actual model-generated code executed in
the original Linux seccomp sandbox. Independent Python calculations verified:

| Check | Expected | Result |
| --- | --- | --- |
| Inventory row count | 8 | PASS |
| Total Stock | 294 | PASS |
| Total Stock × Unit_Price | 62490 | PASS |
| Products with Stock below Reorder_Level | 4 | PASS |
| Three lowest-stock products | DEMO-107: 0; DEMO-106: 6; DEMO-103: 8 | PASS |
| Reorder-point definition with live sources | Sources and explanation returned | PASS |

The live interface also returned 8 rows with GPT-6 Luna enabled. Code generation,
execution, definition lookup, plain-English output, and chat were exercised.

Anomaly: the low-stock count was correct, but the model-written summary failed
the existing grounding/number checks. The app disclosed the problem and returned
a direct description of the executed result. This was a handled model-output
problem, not a wrong final count. Structured rankings intentionally retain the
original deterministic wording from executed fields.

Platform limitation: the complete macOS run had 190 passes, 29 failures, and 11
skips, because Q1 deliberately refuses execution outside its Linux sandbox.
The clean Linux run passed all 230 tests. The Mac launcher does not provide a
Linux execution environment automatically.

The initial tests used the bundled synthetic workbook. After the actual source
folder was supplied, browser checks also used the original workbook; see below.

## Q2 support

Ran all 12 demo questions in one live-model-configured session. All 12 passed
independent checks of expected answer status, section/quote citations, repeats,
and citation existence in the actual KB. Follow-ups, topic switches, repeated
questions, unsupported SAML, and ambiguous links were covered. A separate live
Streamlit test passed an invitation question and its expiry follow-up.

No anomaly appeared in the tested sequence. GPT-6 Luna selects allowed source
sentences; session management and final cited wording remain grounded in the
existing implementation. Repeats and unsupported questions can be resolved
locally without an API call, by design. This is not an unrestricted generative
support assistant and has limited paraphrase coverage.

## Q3 documents

Fresh Tesseract processing of all ten bundled synthetic PNGs identified all ten
types and represented all 44 required fields, with scores and review flags.
The generated review report included threshold 0.90 and rationale. Existing
blank-image, malformed-file, conflict, date/identifier, and checkbox cases passed
in the full suite and UI tests.

Audited the saved supplied-document outputs: ten per-document JSONs match the
combined JSON, all exact schema fields are represented, all scores are in range,
and the review report lists all 44 review-required fields with threshold/rationale.
Q3 core files remain byte-identical to the submitted original.

At the initial audit, original private scans were excluded from this folder.
After the user supplied their source folder, all ten actual assessment images
were freshly reprocessed in the browser; see the additional results below. The saved submitted result
remains 31/42 normalized readable-reference matches: 15/15 printed, 16/27
handwritten, eight nulls, three wrong values, and two excluded ambiguous references.
This is a local-reference diagnostic, not official-answer-key accuracy. Synthetic
image tests do not establish handwritten accuracy. All required deliverables are
present, but handwritten recognition remains incomplete and needs review.

## Test evidence and scope

Evidence is in outputs/v1-audit: full Linux and macOS suite logs/XML; live-q1.json;
live-q2.json; live-ui.json; fresh synthetic image results; saved-output audit;
API/search checks; and summary.json. Existing submitted artifacts are preserved.
No application code was changed during this audit.

The initial Linux test transfer introduced Apple metadata sidecar files. Those
transfer-generated files caused decoding errors; removing only those sidecars
from the temporary VM test copy restored all 230 passes. They were not failures
of the source folder's normal workflow. The final clean run is the cited result.

The API key was used only in temporary credential files on host/VM and removed
from both after testing. It was not added to source or reports.

## Actual browser verification and newly supplied files

Used the actual browser UI at localhost:8517, backed by the unchanged Version 1
source on Linux. This supplements the initial automated and AppTest checks.

Q1: uploaded Inventory-Records-Sample-Data.xlsx from the user-supplied folder.
The loader correctly found the header after five title/blank rows and loaded
46 products. Live GPT-6 Luna generated and executed queries returning 46 products,
2,004 stock units, and $359,760 inventory value. Separate pandas calculations
against the original workbook agreed with all three values. Downloaded transcript
JSON was also checked. Earlier browser checks covered live public definition
sources and the synthetic workbook's 62,490 value.

Q2: replayed all 12 turns through the browser, checked downloaded answers against
the actual KB, verified section quotes and repeat handling, then confirmed that
reset cleared the session.

Q3: uploaded the actual PAN image (ID.png). In the default Tesseract configuration,
the PAN number and birth date were correct, but full name and father name were
null because segmentation passes disagreed. Enabling the EXISTING "Use inspected
supplied-layout region templates" option returned all four values correctly by
direct comparison with the card. All remained review-required because region
alignment is conservatively flagged. No code was changed to achieve this.

Then uploaded all ten target assessment images (four identity images and six
insurance forms). Automatic classification identified all ten types and emitted
all 44 exact schema field slots. With existing region templates and option-box
detection enabled, 20 fields had values and 24 were null. Non-null does not mean
correct: for example, Moral.jpeg returned place "OS", while the visible handwritten
place is "West Bihar". All 44 fields were flagged for review. The downloaded
extraction JSON and review report had the correct document/field counts, threshold
0.90, and rationale. Optional TrOCR weights and macOS Vision were unavailable in
this Linux run; these results must not be represented as the saved 31/42 run.
A full accuracy score for this fresh batch was not measured.

Also verified a blank image is classified unknown and sent for review, and the
synthetic ten-document browser demonstration emits 44 field slots. Downloads
worked. Changing inputs or returning to a page can leave prior extraction results
visible until another run; the interface labels them as the displayed extraction.

The workflow and exports operate, but automatic handwriting extraction is
incomplete. The PAN default-options failure is another concrete setup shortfall.
Enable the existing supplied-layout option for these layouts and retain human
review for uncertain/incorrect fields. Q1 still requires Linux for code execution.

Browser evidence and downloaded JSONs are under outputs/v1-browser. Actual identity
outputs are private evidence. The temporary API key was cleared from the browser
session after live tests. Original submission and application code remain unchanged.
