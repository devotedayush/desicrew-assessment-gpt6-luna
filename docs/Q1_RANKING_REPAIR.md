# Inventory ranking repair and evaluation

The imported implementation documents a genuine Qwen2.5-Coder-0.5B baseline of 3/4 independently checked questions. The lowest-stock query used descending order, and the original generated summary also introduced product labels absent from its executed table. This is a model failure; a successful sandbox execution alone does not establish that the requested question was answered.

The existing application already withholds ranked results whose source identifier/stock pairs, ordering or prefix membership fail its narrow independent check. It permits one genuine model correction attempt, then reports failure. Structured summaries use executed fields directly and are labelled `deterministic-structured`; numeric model summaries undergo grounding checks. These controls prevent displaying unsupported results without repairing a weak model's reasoning by silently substituting answers.

The final-work revision retains raw generated code and records every unambiguous whitespace-only schema binding in `schema_bindings_normalized`. This adapts selectors such as a space omitted beside a hyphen to the one matching actual column; it changes neither query operations nor sorting direction. Ambiguous names and predicate values receive no such rewrite. A literal `df = sheets['known sheet']` rebind selects that sheet's schema after the assignment; complex or repeated rebindings disable schema normalization. The revised code still passes the same sandbox.

The final-work revision adds explicit generation instructions for ascending lowest-stock and descending highest-stock requests, retaining source identifiers and requested row counts. The example uses illustrative columns; all generated code must still use the actual workbook schema and pass the unchanged AST and Linux seccomp restrictions. This is prompt guidance, not a deterministic query replacement.

The live-model evaluator now calculates scalar reference totals with Python `math.fsum` and ranking references with Python `sorted` over independently read source records. It no longer casts reference totals to integers, accepts valid cutoff ties and verifies identifier/metric pairs together. Reference arithmetic is separate from the application's planner and pandas query execution. It also distinguishes model-written numeric summaries from deterministic structured summaries. `query_result_verified` measures the executed model query; `summary_method` records the wording's actual provenance. A deterministic summary is never counted as model-written.

Twelve additional regression cases cover fractional totals, changed metric values, fabricated/repeated identifiers, genuine ties, wrong direction/prefix membership, actual restricted arithmetic execution and a controlled generated-code transport fixture. The fixture tests are implementation checks, not live-model accuracy results. The final Q1 and sandbox subset passes 67 tests on supported Linux (20.52 seconds). The full revised suite separately passes 223 tests (22.63 seconds), and optional model dependencies pass `pip check`.

Fresh genuine inference runs are preserved separately: the first revision passed 2/4; an unscoped server ranking reminder passed ranking but regressed the scalar questions, passing 1/4 overall. The final server reminder repeats the exact question and schema near generation and adds an ascending/descending hint only for that request's application-generated ranking constraints. It supplies neither a replacement query nor independent reference answers.

The final revision passes all four independently checked queries: record count, recorded stock total, recorded USD value total and the five lowest-stock products. Five genuine code-generation requests and three genuine summary requests were made; record count needed its one permitted correction for malformed JSON. All four accepted computations were mechanically bound from bare expressions to `result`. The ranking query additionally received one disclosed whitespace-only column binding; its model-generated ascending direction was preserved and independently verified. Three scalar summaries were model-written and numerically grounded. The ranking description was explicitly `deterministic-structured`, with no model summary call and no deterministic summary fallback. All raw private traces remain local; the public aggregate is `artifacts/inventory_model_final_qa_public.json`.

The parent also verified a genuine browser model query on the synthetic workbook: `df.shape[0]` returned scalar 8, matching its independent row count, with a checked model-written summary. A separate earlier model failure screenshot is retained as failure-handling evidence. These small diagnostic successes do not establish general analytical accuracy; this tiny model remains sensitive to wording and can emit malformed JSON.

On supported Linux, after installing baseline and development dependencies:

```bash
python -m pytest -q tests/test_inventory.py tests/test_inventory_oracle.py tests/test_sandbox.py
```

For genuine local inference, prepare a separate supported Linux Python environment and install `requirements.txt` plus `requirements-inventory-model.txt`. On amd64, install the pinned Torch version from PyTorch's CPU wheel index first; on aarch64, the measured run uses the PyPI `torch==2.6.0` wheel and the server explicitly runs the model on CPU. Download only the official pinned local weights as described in `LOCAL_MODEL_QA.md`; no weights are shipped. Start the offline safetensors server:

```bash
python scripts/evaluate_inventory_llm.py serve \
  --model-dir data/private/models/qwen2.5-coder-0.5b-instruct \
  --threads 2 --serve-seconds 1800 \
  --trace-dir outputs/private_samples/final_local_llm_generations
```

Then run:

```bash
python scripts/evaluate_inventory_llm.py evaluate \
  --workbook 'data/private/Inventory-Records-Sample-Data.xlsx' \
  --private-output outputs/private_samples/final_inventory_llm_qa.json
```

Pass the actual header and column parameters when a different workbook layout is used. The evaluator independently reads the first worksheet; the defaults match the supplied assessment layout. Raw prompts, rows and responses stay in the private output and must not enter public evidence. Public summary counts and current execution status belong in the final verification report. The historical 3/4 result and the failed intermediate 2/4 and 1/4 runs remain disclosed separately from the final 4/4 diagnostic. Re-running a different provider, prompt or dataset requires a new measurement.

The intent validator remains deliberately narrow: unfiltered individual stock rankings with unambiguous source metrics and unique recognized identifiers. Filtered/grouped questions and ambiguous schemas are inconclusive. It is not a general semantic checker or proof of complete analytical accuracy.
