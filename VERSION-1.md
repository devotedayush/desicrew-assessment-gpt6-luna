# Version 1 with GPT-6 Luna

Copied from `desicrew-assessment (submitted original)`. The submitted source
folder is unchanged. No later one-point-improvement code was used.

## What changes

- Q1: GPT-6 Luna replaces the configured LLM for code planning and existing
  model summary calls. Schema validation, ranking checks, sandbox execution,
  definition search, and chat behaviour remain in place.
- Q2: GPT-6 Luna replaces the configured LLM for grounded sentence selection.
  Answers still use validated knowledge-base sentences. Session memory,
  repetition handling, topic switching, and citations remain in place.
- Q3: unchanged. No documents are sent to OpenAI by Q3.

The adapter uses `gpt-6-luna`, Responses API, medium reasoning, bounded output,
JSON mode where existing callers request JSON, and `store: false`. This flag
controls response storage; it is not a claim of zero provider retention.
API errors do not include keys or provider response bodies.

## Run

Install the original requirements as described in README.md. Q1 still needs
Linux with its execution sandbox; use the original Linux/VM/Docker setup.
Q2 and Q3 can run locally on macOS, with the existing Q3 optional dependencies.

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Enter a replacement OpenAI API key in the sidebar password field, or set
`OPENAI_API_KEY` in the environment before launching. `.env` is not auto-loaded.
The key is not written to source or output files. The LLM toggle enables calls.
For Q1, enable the existing inventory-data permission checkbox: it permits
workbook schema, bounded previews, and query results to be sent to OpenAI.
Q2 sends fictional KB passages and allowlisted query concepts, as before.

The pasted chat credential is not bundled. Revoke it and enter a replacement.

## Results

Copied reports and reviewer_outputs are historical submitted-baseline evidence.
They are not GPT-6 Luna results. No new Q3 accuracy claim applies to this copy.
Live API verification needs a valid key with model access and billing.
See VERSION-1-VERIFICATION.json for checks actually performed on this copy.

## Official API references

- https://developers.openai.com/api/docs/models/gpt-6-luna
- https://developers.openai.com/api/reference/responses/overview
