# DesiCrew Assessment — Version 1 (GPT-6 Luna)

A Streamlit app with three assessment workflows: ask questions about an Excel inventory, get cited support answers from a small knowledge base, and extract fields from identity and insurance documents for human review.

This version is based on the submitted original. **GPT-6 Luna replaces the existing LLM calls in Q1 and Q2; Q3 processing is unchanged and stays local.** The submitted-original folder is preserved separately and must not be edited.

## What the app does

| Workflow | Behaviour |
| --- | --- |
| Q1: Inventory code agent | Loads XLSX/CSV, generates pandas code with GPT-6 Luna, runs it in a restricted Linux sandbox, and shows the answer, code and result. Can look up public definitions. |
| Q2: Document support | Retrieves information from fictional MeadowDesk Markdown documents, cites specific sections, tracks the conversation, handles follow-ups and avoids repeating the same information. GPT-6 Luna selects from allowed source sentences. |
| Q3: Document review | Classifies ten document types, represents 44 required fields, attaches quality scores and flags uncertain values for human review. Exports extraction JSON and a review report. |

## Run on Linux

Use Python 3.12, Tesseract 5 and libseccomp. **Q1 code execution requires Linux.** On macOS or Windows, run the app inside a Linux VM or use the Docker option below.

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv tesseract-ocr libseccomp2
git clone https://github.com/devotedayush/desicrew-assessment-gpt6-luna.git
cd desicrew-assessment-gpt6-luna
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip check
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501**. The repository is private, so cloning requires access to it.

### Enable GPT-6 Luna

1. Enter your OpenAI API key in the sidebar password field.
2. Turn on **Enable configured LLM**.
3. For Q1, also enable **Allow inventory schema, preview and query results to this provider**.
4. Upload your workbook and ask a question. Q2 is available under **Document support**.

You can alternatively set `OPENAI_API_KEY` in the process environment before starting Streamlit. The app does not automatically load `.env`. Keys are not included in this repository. The adapter calls the Responses API with model `gpt-6-luna`; your API account must have access and billing enabled. Without configuration, Q1 uses a labelled offline planner demonstration.

Q1 sends workbook schema, bounded previews and query results to OpenAI after its permission toggle is enabled. Q2 sends fictional KB snippets and allowlisted concepts. **Q3 does not send document images to OpenAI.** Responses use `store: false`; this does not imply zero provider retention.

### Docker option

```bash
docker compose up --build
```

Open the same localhost URL and enter your API key in the sidebar. Docker Compose binds the app only to loopback. Linux execution was verified in a local Lima VM; this project's Docker build was not separately verified.

### macOS use

Q2 and base Q3 can run directly on macOS with Python 3.12 and Tesseract installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Q1 refuses code execution on native macOS. Use the Linux VM or container for the complete app. Optional local macOS Vision recognition is documented in [docs/OCR_METHOD.md](docs/OCR_METHOD.md).

## Using Q3

Upload the document images under **Document review**. Leave automatic classification selected and click **Classify and extract**. Inspect the fields and download the extraction JSON and flagging report.

For the supplied assessment layouts, expand **Optional local value-line recognition** and enable **Use inspected supplied-layout region templates**. You may also enable the existing option-box detection. These templates tell OCR where to look; they do not contain expected answers. They are specific to the inspected layouts and may not suit differently arranged documents.

The default whole-image OCR missed both names on the supplied PAN image. Enabling the existing layout option recovered all four PAN fields correctly in the browser check. All remained flagged for review because crop alignment requires checking.

The review threshold defaults to **0.90**. Scores are uncalibrated quality heuristics, not probabilities. Handwriting remains incomplete; a populated field is not proof of correctness. Optional TrOCR needs separately installed dependencies and local model weights. Opening the app does not download model weights.

## Verification and known limitations

- **230 automated tests passed** in the supported Linux environment.
- Live browser checks on the supplied workbook returned **46 products, 2,004 stock units and $359,760 inventory value**, matching independent calculations.
- The Q2 12-turn browser demonstration passed answer-status, citation and repeat checks; reset worked.
- All ten supplied assessment images were identified as their expected document types and produced all **44 required field slots**. With existing layout regions and option-box detection on Linux/Tesseract, **20 slots had values and 24 were blank**. All 44 were flagged for review. Optional TrOCR and macOS Vision were unavailable in that run. This is not a full accuracy measurement.
- Earlier synthetic checks covered all ten types, downloads and blank-image review. Synthetic results do not measure handwriting accuracy.
- Q1 has validation and sandbox controls, but these finite tests do not guarantee every analytical question is answered correctly. Q2 has limited paraphrase coverage. Q3 requires human review.

See [VERSION-1-TEST-REPORT.md](VERSION-1-TEST-REPORT.md) for the test scope and [VERSION-1.md](VERSION-1.md) for the model integration details. Other reports under `docs/` and `artifacts/` describe the historical submitted baseline; they are not new GPT-6 Luna measurements.

## Tests and sample data

```bash
python -m pytest -q
python scripts/demo_support.py
python scripts/evaluate_support.py
python scripts/demo_inventory.py
```

Run the full suite on Linux. Native macOS failures in sandbox execution are expected. Fictional KB documents and synthetic workbooks/images are bundled for demonstration. The actual assessment workbook, identity scans, private extraction outputs and visual reviews are excluded from Git.

## Repository layout

```text
app.py                 Streamlit entry point
desicrew/              Inventory, support, OCR, sandbox and OpenAI adapter
scripts/               Demonstrations, evaluations and packaging
tests/                 Automated checks
docs/kb/               Fictional support knowledge base
data/synthetic/        Demonstration workbook and documents
artifacts/             Selected historical synthetic/aggregate evidence
```

Keep API keys, uploaded workbooks, identity images and generated private outputs out of source control. Local `outputs/`, `data/private/`, `artifacts/private/`, `reviewer_outputs/` and `review_appendix/` are excluded.
