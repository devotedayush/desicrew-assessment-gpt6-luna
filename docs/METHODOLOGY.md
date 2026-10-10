# Methodology and candidate notes

## Q1: executable, auditable inventory answers

The workbook is parsed into named DataFrames. Schema and a small preview ground query planning. With an explicitly enabled model, a JSON response carries generated pandas code, which is validated before execution. The core accepts a model/search interface so transport can be tested without credentials. Offline demonstration planning is explicitly labelled and does not count as an LLM run.

Generated code does not execute in the Streamlit server. A separate Linux runner uses a restrictive AST/call policy, a minimal environment, bounded time and resources, and OS syscall restrictions. The result is serialized into bounded JSON for explanation. This is an assessment-scale defensive runner, not a claim of universal arbitrary-Python isolation. Unsupported queries fail safely rather than widening the allowlist silently. A production multi-tenant service should add container/VM isolation and security review.

Public concept lookup is separate from data queries. Only predefined conceptual phrases such as reorder point, inventory turnover and stock keeping unit reach the public search endpoint; names, rows and arbitrary user strings are not transmitted as search queries. Search failure is disclosed, never replaced with fabricated citations.

## Q2: cited support with session context

The knowledge base is original and fictional, with explicit document and section identities. The default assistant uses deterministic extractive retrieval and quotes actual source sentences. Session memory tracks topics and already-covered sections. Follow-up questions use earlier context, a topic switch changes active topic, and repeats can acknowledge prior coverage rather than reprint the same answer. Unsupported questions use a cited scope boundary.

The reproducible full-exchange demo tests this behavior. Source quotations are retained so each answer can be audited. Optional model assistance remains constrained by source validation; unconstrained generation is not treated as authoritative knowledge.

## Q3: conservative local extraction

The pipeline accepts local PDF/image/text files. PDF text is used when available; local OCR is the default for scanned pages. Document classification combines document titles and identifying signals. The schema always represents every required field for a recognized type, including nulls for unavailable values.

Extracted identifiers remain strings to preserve leading zeros and MRZ punctuation. Basic syntax validation catches some implausible values but is not proof of correct transcription. Source evidence, field status and validation issues accompany each value.

A default **0.90 review threshold** is a conservative operating choice: weak/missing/conflicting values and handwritten critical identifiers are routed to review. It is not selected from ground-truth accuracy measurements and not calibrated as a probability. Every handwritten form is handled more cautiously than printed text; extra attention goes to IFSC, TIN/PAN, account numbers, dates and place names. Tesseract-only handwriting remains an explicit limitation. See the detailed OCR note for actual implementation and observed synthetic failures.

## What the included evidence establishes

Synthetic tests demonstrate software behavior, schemas, safety controls and review routing. They do not establish field accuracy on unseen scans. Actual supplied-workbook model queries and supplied-document extraction were evaluated separately; see the current aggregate reports and their reference limitations. Mocked model transport demonstrates adapter integration, not live model reasoning quality. The verification report distinguishes tests actually run, historical diagnostics and remaining limitations.

No supplied identity/bank data is transmitted to an external model/OCR provider. No deployment, paid API request, third-party submission, or credential creation is performed by these demonstrations.
