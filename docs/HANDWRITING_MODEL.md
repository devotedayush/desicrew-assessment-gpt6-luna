# Offline cropped-line handwriting recognizer

> Historical optional TrOCR extension. The setup below targets Linux amd64 because its pinned Torch CPU wheel is amd64-only. It was not rerun in the final arm64 environment. Current measured extraction uses portable Tesseract, local macOS Vision, and the separately guarded arm64 GOT comparison; see `Q3_LOCAL_EVALUATION.md`.

## What is implemented

`desicrew/handwriting.py` adds an optional local CPU recognizer using Microsoft's
[TrOCR small handwritten checkpoint](https://huggingface.co/microsoft/trocr-small-handwritten).
The publisher describes this IAM-fine-tuned checkpoint as a recognizer for single
text-line images. It is therefore used only on explicit, visually checked
handwritten-value ROIs, never on a whole identity card, banking form, or form page.
Printed page classification and label discovery still use local Tesseract.

The small checkpoint's exact official revision is
`b4648cfa171985a6745f37ddd637e98c0da958ac`. Its main branch has PyTorch weights,
not a safetensors file. Explicit setup downloads the public files from that pinned
revision, checks the publisher's LFS SHA-256 hashes, loads the state dict with
PyTorch 2.6 `weights_only=True`, and converts locally to safetensors. Runtime uses
only the local safe weights. No private document is part of the download request.

A bounded comparison also actually ran Microsoft's [TrOCR base handwritten
checkpoint](https://huggingface.co/microsoft/trocr-base-handwritten), at revision
`eaacaf452b06415df8f10bb6fad3a4c11e609406`. This official revision provides
native safetensors (1,333,384,464 bytes), so no PyTorch pickle was downloaded or
converted for base. Its official LFS SHA-256 is
`25a40cddc7e6120140a3d5b9e3dd3878a92ada7b4f312953ab22edc19c2a5acc`.

Inference passes `local_files_only=True`, `token=False`,
`trust_remote_code=False`, and `use_safetensors=True`. It uses CPU only and does
not call hosted inference, an external OCR service, or a paid API. Model weights,
private crops, references, and real document outputs are excluded from packaging.
The base application environment does not need torch or transformers.

## Setup and evaluation

From the repository root, use an isolated environment:

```bash
python -m venv .hwr-venv
.hwr-venv/bin/python -m pip install -r requirements-handwriting.txt
.hwr-venv/bin/python scripts/evaluate_handwriting.py --download-model --download-only
.hwr-venv/bin/python scripts/evaluate_handwriting.py --manifest outputs/private_samples/handwriting_crops.json
```

Downloads are explicit, public-weight-only setup. Runtime never downloads missing
weights or searches for credentials. `requirements-handwriting.txt` pins the
versions used for this extension; PyTorch's CPU wheels come from its official
registry. If a model is missing, inference fails with a clear local setup message.
The separate application requirements remain unchanged.

To reproduce the optional base comparison:

```bash
.hwr-venv/bin/python scripts/evaluate_handwriting.py --model-size base --download-model --download-only
.hwr-venv/bin/python scripts/evaluate_handwriting.py --model-size base --output-dir outputs/private_samples/handwriting_base
```

For either model, `--preprocessing contrast_rule_cleanup` enables a bounded
diagnostic variant: grayscale/autocontrast, then removal of dark horizontal runs
longer than 58% of crop width. This can erase touched strokes and is explicitly
flagged as destructive/unvalidated. It is off by default. It does not change
characters in recognized strings, use reference answers, or alter original images.

`DocumentProcessor(handwriting_backend=LocalHandwritingRecognizer(model_path),
roi_manifest_path=...)` accepts explicit original-image field/page/bbox regions.
The document CLI accepts `--handwriting-model` and `--roi-manifest`. A supplied
ROI manifest uses this structure:

```json
{
  "coordinate_space": "original_image_pixels",
  "documents": [{
    "file": "path/to/local/form.png",
    "document_type": "nach_ecs_mandate",
    "regions": [{"field": "ifsc_code", "page": 1,
      "bbox": [250, 150, 600, 200],
      "region_source": "human_verified_single_line"}]
  }]
}
```

The example is illustrative, not a supplied sample. Private reference text is used
only by the evaluation comparison, never sent as a model prompt or used to repair
recognition output. Reference crops are independent of the pipeline's extraction
ROI manifest, which contains coordinates and field labels rather than answers.

## Bounds and uncertainty

The default recognizer allows 64 new tokens, a 15-second soft generation budget,
two CPU threads, and at most one million input pixels. Geometric guards reject
obvious full pages. These guards cannot prove that a region contains a single
line; crop selection remains explicit. Transformers checks its soft time limit
after a decoding step; model loading and encoding can exceed it. The evaluation
script therefore runs the entire offline batch in a subprocess with a 600-second
hard timeout. A truncated sequence is flagged and its score capped at 0.25.

Each result carries raw recognized text, original-page crop geometry, model ID and
revision, token count, decoding time, preprocessing details, and mandatory review flags.
Model ID/revision come from its local download manifest rather than a hardcoded
small-model label. Runtime rechecks the safetensors SHA-256 before loading.
An arbitrary local checkpoint without a manifest is labeled an unverified local
source and gains an explicit review issue. The decoder's
geometric-mean token softmax score is reported as a decoding statistic. The field
quality score is capped at 0.72, before the document processor's validation and
high-risk caps. Neither statistic is a calibrated probability or accuracy score.
A valid-looking account number or IFSC can still be wrong.

No character confusion is silently repaired. Pattern checks do not change O/0,
I/1 or other characters to make a bank/identity field pass. Model and Tesseract
candidates that disagree remain null and require review. Multiple model candidates
for the same field that disagree also remain null. Checkbox selection is not
inferred from line-model text. All handwritten fields require human review,
including plausible strings that pass format checks.

## Evaluation evidence

The local evaluation manifest contains 18 manually selected line crops from six
supplied handwritten forms. A visual inspection produced 17 readable local
references; the crowded Benefit Illustration date is ambiguous and has a null
reference. Actual recognizer results and any private values belong only in
`outputs/private_samples/handwriting_line_evaluation.json`.

Case and whitespace are normalized for exact-line and aggregate character-edit
comparison; characters are never repaired. Null/unverified references are excluded.
This is a small diagnostic sample, not the official answer key, a held-out benchmark,
or complete-field accuracy. Repeated names/place names reduce its diversity.
IAM English handwriting differs from the supplied insurance-form layout, boxed
identifiers, mixed printed/handwritten regions, short fragments, and noisy scan
backgrounds. Counts and observed failures may be summarized publicly without
names, bank/identity numbers, cropped pixels, or typed private references.

### Actual local runs

Both official checkpoints emit a loader notice about newly initialized encoder
pooler weights. Local Transformers source confirms generation consumes the
encoder's last hidden state, not its pooled output; the unused pooler does not
turn this into a newly trained model. The notice remains in private run logs.

All four runs processed the same 18 inspected crops across six forms. The 17
readable references contain 179 case/whitespace-normalized characters; one
ambiguous date was excluded. No private reference answer was fed to either model.

| Model / preprocessing | Exact lines / 17 | Character edits | Aggregate CER | CPU inference time |
| --- | ---: | ---: | ---: | ---: |
| Small / original crop | 1 | 152 | 0.8492 | 6.9 s |
| Small / diagnostic cleanup | 4 | 87 | 0.4860 | 6.6 s |
| Base / original crop | 5 | 89 | 0.4972 | 24.8 s |
| Base / diagnostic cleanup | 5 | 87 | 0.4860 | 23.6 s |

No generation budget was exhausted, and all 72 recognizer results remained
mandatory-review. These are line diagnostics, including a few fragments beyond
brief targets, rather than verified 44-field accuracy. Exact matches are strict
apart from case/whitespace; plausible punctuation differences can still miss.

Pixel inspection showed genuine errors beyond punctuation: table boxes confused
bank account/IFSC recognition, numeric identifiers dropped/substituted characters,
short relationships became unrelated words, noisy place text changed spelling,
and dates lost separators or became unrelated text. Base correctly read some
names, an amount, and an application identifier, but still failed high-value
bank/date targets. Cleanup improved small-model results while adding a risk of
removing handwriting touching horizontal rules; it did not increase base's exact
match count. Neither checkpoint should be treated as robust or calibrated on
these forms. Use optional model text as review evidence, retain conflicts as
null, and validate every field against pixels/authorized ground truth.

`docs/handwriting_model_qa.json` contains sanitized run counts and limitations.
Private per-line values, references, and comparison contact sheets remain only
under the excluded private output tree. The evaluation script rejects output
destinations outside `outputs/private_samples`.

The sixteen deterministic adapter/evaluation tests validate offline path requirements, safe-weight
requirements, lazy loading, input/budget checks, explicit offline/safe/CPU loader
options, model identity/hash checks, diagnostic cleanup, and conservative scoring. They do
not run the model or measure recognition accuracy. Actual inference evidence must
be reported separately; installing weights alone does not establish that a model ran.

Official implementation details:
[Transformers 4.46.3 TrOCR documentation](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/trocr)
