#!/usr/bin/env python3
"""Read-only comparison to PRIVATE independent visual references.

Never fills extraction output from a reference. Aggregate counts are diagnostics,
not official-answer-key accuracy or evidence of identity authenticity.
"""
from __future__ import annotations
import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desicrew.documents import DATE_FIELDS, IDENTIFIER_FIELDS, HANDWRITTEN_TYPES, _valid_date, _clean_value


def normalize(field: str, value: object) -> str | None:
    if value is None:
        return None
    text = _clean_value(str(value))
    if field in DATE_FIELDS:
        date, _ = _valid_date(text)
        return date if date is not None else re.sub(r"\s+", " ", text).casefold()
    if field == "amount_figures":
        text = re.sub(r"^(?:₹|INR|Rs\.?)\s*", "", text, flags=re.I)
        text = re.sub(r"\s*/-\s*$", "", text)
        try:
            return str(Decimal(text.replace(",", "").replace(" ", "")).normalize())
        except InvalidOperation:
            return text
    if field in IDENTIFIER_FIELDS or field == "mrz_line_2":
        return re.sub(r"\s+", "", text).casefold()
    if field == "address":
        return re.sub(r"[\s,]+", "", text).casefold()
    return re.sub(r"\s+", " ", text).casefold()


def candidate_texts(item: dict) -> list[str]:
    values = []
    source = item.get("source") or {}
    if source.get("candidate_text"):
        values.append(source["candidate_text"])
    for candidate in item.get("region_candidates", []):
        if candidate.get("text"):
            values.append(candidate["text"])
    model = item.get("handwriting_candidate") or {}
    if model.get("text"):
        values.append(model["text"])
    for option in item.get("checkbox_candidates", []):
        if option.get("status") == "selected":
            values.extend([option["label"], option.get("description", option["label"])])
    return list(dict.fromkeys(values))


def compare(references: dict, results: list[dict]) -> dict:
    by_file = {result["source_file"]: result for result in results}
    counts = Counter()
    groups = {"printed": Counter(), "handwritten": Counter()}
    fields = []
    for document in references["documents"]:
        result = by_file.get(Path(document["file"]).name, {})
        group = "handwritten" if document["document_type"] in HANDWRITTEN_TYPES else "printed"
        for field, reference in document["fields"].items():
            counts["targets_checked"] += 1
            groups[group]["targets_checked"] += 1
            if not reference.get("readable") or reference.get("value") is None:
                counts["excluded_references"] += 1
                groups[group]["excluded_references"] += 1
                continue
            expected = normalize(field, reference["value"])
            item = result.get("fields", {}).get(field, {})
            actual = item.get("value")
            normalized_match = actual is not None and normalize(field, actual) == expected
            literal_match = actual is not None and str(actual) == str(reference["value"])
            raw_match = any(normalize(field, value) == expected for value in candidate_texts(item))
            conflict = any("conflict" in issue or "disagree" in issue for issue in item.get("validation_issues", []))
            additions = {"readable_references": 1, "literal_final_exact_matches": int(literal_match), "normalized_final_exact_matches": int(normalized_match), "correct_raw_candidate_fields": int(raw_match), "correct_but_rejected_raw_candidate_fields": int(actual is None and raw_match), "missing_or_null_final_values": int(actual is None), "conflicting_final_fields": int(conflict), "incorrect_non_null_final_values": int(actual is not None and not normalized_match)}
            counts.update(additions)
            groups[group].update(additions)
            fields.append({"file": Path(document["file"]).name, "document_type": document["document_type"], "field": field, "normalized_final_exact_match": normalized_match, "literal_final_exact_match": literal_match, "raw_candidate_exact_match": raw_match, "final_value_null": actual is None, "conflict_flag": conflict, "status": item.get("status"), "validation_issues": item.get("validation_issues", [])})
    keys = ("targets_checked", "readable_references", "excluded_references", "literal_final_exact_matches", "normalized_final_exact_matches", "correct_raw_candidate_fields", "correct_but_rejected_raw_candidate_fields", "missing_or_null_final_values", "conflicting_final_fields", "incorrect_non_null_final_values")
    return {"counts": {key: counts[key] for key in keys}, "groups": {group: {key: values[key] for key in keys} for group, values in groups.items()}, "field_diagnostics": fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--results", action="append", required=True, help="Repeat CONFIG_NAME=local_results_json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "private_samples" / "diagnostics")
    parser.add_argument("--public-summary", type=Path, help="Optional aggregate-only JSON: no values, filenames, private paths or field details")
    args = parser.parse_args()
    references = json.loads(args.references.read_text())
    comparisons = {}
    for specification in args.results:
        name, filename = specification.split("=", 1)
        comparisons[name] = compare(references, json.loads(Path(filename).read_text()))
    common = {"purpose": "independent_visual_reference_diagnostic_not_official_answer_key", "accuracy_claim": False, "reference_method": "Manual inspection of original pixels, including enlarged crops for ambiguous handwriting; some visual references were refined after the first baseline run, but never from model predictions or used to fill extraction output. Unresolved references are excluded.", "normalization": "Field separator/underline characters at outer edges are removed as in the extractor. Dates: unambiguous four-digit dates to ISO; no guessed centuries. Amounts: currency/grouping formatting only. Identifiers/MRZ: whitespace and case only, no character repair. Names: whitespace and case only. Addresses: whitespace/case/commas only. Literal exact matches also reported separately.", "caveat": "Correct readable extraction does not establish an identifier's validity/authenticity. A correct raw candidate may remain rejected because source format, checksum, ambiguity or conflicting OCR evidence requires review."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {**common, "configurations": comparisons}
    (args.output_dir / "visual_reference_comparison_private.json").write_text(json.dumps(report, indent=2))
    summary = {**common, "configurations": {name: {"counts": report["counts"], "groups": report["groups"]} for name, report in comparisons.items()}}
    (args.output_dir / "summary_without_field_values.json").write_text(json.dumps(summary, indent=2))
    if args.public_summary:
        args.public_summary.parent.mkdir(parents=True, exist_ok=True)
        args.public_summary.write_text(json.dumps(summary, indent=2))
    print(json.dumps({name: report["counts"] for name, report in comparisons.items()}, indent=2))


if __name__ == "__main__":
    main()
