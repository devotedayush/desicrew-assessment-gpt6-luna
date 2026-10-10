#!/usr/bin/env python3
"""Combine guarded private GOT batches with their shared Tesseract stage.

No answer key or visual reference is read. Printed identity results come from
the same staged OCR pass; every handwritten type must be supplied by exactly
one completed guarded batch. Outputs stay under outputs/private_samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desicrew.documents import DOCUMENT_SCHEMAS, HANDWRITTEN_TYPES, review_report


def private_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((ROOT / "outputs/private_samples").resolve()):
        raise ValueError("Use only private GOT stage, batch and output paths")
    return resolved


def combine(stage_path: Path, batch_paths: list[Path]) -> tuple[list[dict], dict]:
    if stage_path.name != "tesseract_stage_results_private.json":
        raise ValueError("The base must be the private Tesseract-stage result")
    staged = json.loads(private_path(stage_path).read_text())
    if len(staged) != len(DOCUMENT_SCHEMAS) or {r["document_type"] for r in staged} != set(DOCUMENT_SCHEMAS):
        raise ValueError("The staged source must contain exactly the ten target types")
    by_file = {result["source_file"]: result for result in staged}
    if len(by_file) != len(staged):
        raise ValueError("Duplicate staged source filenames")
    replacements = {}
    provenance = {}
    for batch_path in batch_paths:
        batch_path = private_path(batch_path)
        if batch_path.name != "results.json":
            raise ValueError("A completed guarded batch needs results.json")
        resource_path = batch_path.parent / "process_resource_report.json"
        if not resource_path.is_file() or json.loads(resource_path.read_text()).get("status") != "completed":
            raise ValueError("A batch lacks a completed guarded resource report")
        for result in json.loads(batch_path.read_text()):
            source = result["source_file"]
            if source not in by_file or source in replacements or result["document_type"] != by_file[source]["document_type"]:
                raise ValueError("Batch source does not uniquely match the staged classification")
            if result["document_type"] not in HANDWRITTEN_TYPES or result.get("ocr", {}).get("handwriting", {}).get("engine") != "optional_local_model":
                raise ValueError("Each batch replacement must contain guarded handwritten OCR")
            replacements[source] = result
            provenance[source] = str(batch_path)
    if {result["document_type"] for result in replacements.values()} != HANDWRITTEN_TYPES:
        raise ValueError("Completed batches must cover all six handwritten types")
    combined = [replacements.get(result["source_file"], result) for result in staged]
    return combined, {"base_stage": str(stage_path.resolve()), "handwriting_batches_by_file": provenance,
                      "model_results_replaced": len(replacements), "visual_references_used": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = private_path(args.output_dir)
    results, provenance = combine(args.stage, args.batch)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    for index, result in enumerate(results, 1):
        (output / f"{index:02d}_{result['document_type']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    report = review_report(results)
    report["evaluation_mode"] = "supplied_local_files_guarded_got_batches"
    report["sample_accuracy"] = None
    report["sample_accuracy_note"] = "No answer key or visual reference was read by this combiner"
    (output / "review_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (output / "batch_provenance_private.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"documents_processed": len(results), "fields_emitted": sum(len(r["fields"]) for r in results),
                      "documents_needing_review": report["documents_needing_review"],
                      "fields_needing_review": report["fields_needing_review"],
                      "handwriting_batches": len(args.batch)}, indent=2))


if __name__ == "__main__":
    main()
