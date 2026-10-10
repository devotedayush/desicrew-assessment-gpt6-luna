# Genuine browser evidence

All evidence images in the source ZIP are screenshots of the running local
Streamlit app captured with Playwright and installed Google Chrome. They were
not generated or reconstructed. Chrome's browser sandbox was enabled. Q1 code
execution used the unchanged Linux seccomp subprocess inside Ubuntu 24.04;
browser sandboxing and generated-code sandboxing are separate controls.

The public evidence uses synthetic workbooks/documents and the fictional
MeadowDesk KB. No supplied identity/bank scan or value appears in these images.
Actual supplied-file accuracy is measured separately; these screenshots prove
interface behavior, not official benchmark accuracy.

- Q1: offline query/code/result, local LLM query, genuine model-query failure,
  and invalid workbook rejection. The LLM success capture must have a non-null
  scalar result matching an independently counted workbook, not merely an
  `llm` mode label. The initially captured synthetic-product question failed
  because the model called `tolist()` on an integer; it is retained as failure
  evidence and is not counted as a successful query.
- Q2: first/middle/final exchanges of the real 12-turn session, source section
  citations, reset, and an out-of-scope response.
- Q3: ten synthetic text fixtures and 44 field targets, per-field confidence
  and review statuses, actual local OCR on a synthetic PNG, and an unreadable
  PNG routed to human review. No handwritten-accuracy claim is made from the
  synthetic text fixtures.

The capture manifests in `artifacts/ui_evidence_*.json` record page URLs,
checks, timestamps and SHA-256 hashes. Synthetic inventory transcript JSON
records the actual responses. The corrected capture script downloads the
response and checks its executed row count against the workbook.

```bash
python scripts/capture_evidence.py --base-url http://127.0.0.1:8501 \
  --chromium '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
# Model already running and configured in the Linux UI:
python scripts/capture_evidence.py --base-url http://127.0.0.1:8501 \
  --chromium '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
  --pages inventory --llm
```

Capture Q1 from the Linux app. Q2/Q3 may be captured from a local Mac app;
its optional Vision backend requires the Mac bridge. If using Playwright's
bundled browser instead, install it with `python -m playwright install chromium`
and omit `--chromium`. The final ZIP excludes interim debug screenshots.

The offline Q1 replay also performs a real allowlisted public definition search. `q1_definition_search.png` shows retrieved links; `ui_inventory_definition_transcript.json` records the response. Its wording is explicitly `built-in-context`, distinct from retrieved public sources and actual LLM code generation.
