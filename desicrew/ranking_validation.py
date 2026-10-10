"""Narrow independent validation of unfiltered individual-item stock rankings.

This module checks a generated answer. It never replaces or authors its query.
"""

from __future__ import annotations

import ast
import math
import numbers
import re

import pandas as pd

QUANTITY_ALIASES = {"quantity", "qty", "stock", "stockquantity", "stockqty", "stockonhand", "onhand", "units", "currentstock", "handinstock", "closingstock", "endingstock", "availablequantity"}
IDENTIFIER_ALIASES = {"sku", "productid", "itemid", "productcode", "itemcode", "stockcode"}
NAME_ALIASES = {"productname", "itemname"}
COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20}


class AnalysisValidationError(ValueError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def _normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _identifier(value):
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _equal_number(left, right):
    if isinstance(left, numbers.Integral) and isinstance(right, numbers.Integral):
        return int(left) == int(right)
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)


def validate_item_ranking(workbook, question: str, code: str, result: dict) -> dict:
    report = {"status": "not_applicable", "check": "individual-stock-ranking", "rejected": False}
    lowered = question.lower()
    direction = re.search(r"\b(lowest|least|highest|most)\b", lowered)
    if not direction or not re.search(r"\b(?:items?|products?|skus?)\b", lowered):
        report["reason"] = "No explicit individual-item lowest/highest ranking was requested."
        return report
    report.update(status="inconclusive", direction="lowest" if direction.group() in {"lowest", "least"} else "highest")
    if re.search(r"\b(?:lowest|least)\b", lowered) and re.search(r"\b(?:highest|most)\b", lowered):
        report["reason"] = "The question contains both ranking directions."
        return report
    if re.search(r"\b(?:categor(?:y|ies)|suppliers?|vendors?|warehouses?|locations?|brands?|departments?|groups?|grouped|grouping|filters?|filtered|where|excluding|except|only|between|under|over|below|above)\b", lowered):
        report["reason"] = "Filtered or grouped intent is outside the unfiltered individual-item check."
        return report
    if not re.search(r"\b(?:stock|quantity|qty|units|onhand)\b|\bon hand\b", lowered) or re.search(r"\b(?:value|valuation|worth|price|cost|sold|sales|revenue|turnover)\b", lowered):
        report["reason"] = "The question does not unambiguously request a stock-quantity ranking."
        return report
    try:
        tree = ast.parse(code)
    except SyntaxError:
        report["reason"] = "The generated source cannot be inspected."
        return report
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"groupby", "crosstab", "agg", "aggregate"} for node in ast.walk(tree)):
        report["reason"] = "Generated grouping/aggregation is outside this individual-row check."
        return report
    named = [name for name in workbook.sheets if name.lower() in lowered]
    referenced = {node.slice.value for node in ast.walk(tree) if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "sheets" and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) and node.slice.value in workbook.sheets}
    if len(named) > 1 or (not named and len(referenced) > 1):
        report["reason"] = "The source worksheet is ambiguous."
        return report
    sheet = named[0] if named else next(iter(referenced)) if referenced else next(iter(workbook.sheets))
    frame = workbook.sheets[sheet].reset_index(drop=True)
    metrics = [column for column in frame.columns if _normalized(column) in QUANTITY_ALIASES]
    if len(metrics) > 1:
        # Generic words like stock/quantity are not enough to resolve two fields.
        explicit = [column for column in metrics if re.search(r"['\"]" + re.escape(str(column)) + r"['\"]", question, flags=re.I) or (len(re.findall(r"[a-z0-9]+", str(column).lower())) > 1 and " ".join(re.findall(r"[a-z0-9]+", str(column).lower())) in " ".join(re.findall(r"[a-z0-9]+", lowered)))]
        metrics = explicit if len(explicit) == 1 else metrics
    if len(metrics) != 1:
        report["reason"] = "The schema has no single unambiguous stock-quantity column."
        return report
    metric = metrics[0]
    identifiers = []
    for aliases in [IDENTIFIER_ALIASES, NAME_ALIASES]:
        for column in frame.columns:
            if _normalized(column) in aliases and frame[column].notna().all():
                keys = [_identifier(value) for value in frame[column]]
                if len(set(keys)) == len(frame):
                    identifiers.append(column)
    report.update(sheet=sheet, metric_column=metric)
    if not identifiers:
        report["reason"] = "No recognized, complete, unique item identifier is available."
        return report
    report["source_identifier_columns"] = identifiers
    if result.get("truncated"):
        report["reason"] = "Bounded output is incomplete; full ranking membership cannot be verified."
        return report

    count_pattern = r"(?:\d{1,4}|" + "|".join(COUNTS) + r")"
    count_match = re.search(r"\b(" + count_pattern + r")\s+(?:items?|products?|skus?)\b", lowered) or re.search(r"\b(?:top|bottom|lowest|highest|least|most)\s+(" + count_pattern + r")\b", lowered) or re.search(r"\b(" + count_pattern + r")\s+(?:lowest|highest|least|most)\b", lowered)
    requested = (int(count_match.group(1)) if count_match.group(1).isdigit() else COUNTS[count_match.group(1)]) if count_match else 1 if re.search(r"\b(?:item|product|sku)\b", lowered) and not re.search(r"\b(?:items|products|skus)\b", lowered) else None
    report["requested_limit"] = requested
    if requested is not None and requested < 1:
        report["reason"] = "The requested ranking size is not positive."
        return report
    ascending = report["direction"] == "lowest"
    numeric = pd.to_numeric(frame[metric], errors="coerce")
    finite = numeric.notna() & (numeric != float("inf")) & (numeric != float("-inf"))
    ranked = numeric[finite].sort_values(ascending=ascending, kind="stable")
    report["excluded_missing_or_nonfinite_metrics"] = int((~finite).sum())

    def reject(detail):
        report.update(status="inconclusive", rejected=True, reason=detail)
        size = f" Select the first {requested} ranked rows, or all eligible rows if fewer exist." if requested else " Return a correctly ordered prefix of eligible rows."
        instruction = f"Ranking validation rejected the generated answer: {detail} Return a table containing the exact metric column {metric!r} and a unique identifier column from {identifiers!r}. Sort {metric!r} {'ascending' if ascending else 'descending'}, excluding missing/non-finite metrics.{size} Do not invent identifiers or replace the query with fixed answers."
        raise AnalysisValidationError(instruction, dict(report))

    if result.get("type") != "table":
        reject("Individual-item ranking requires an identifier/metric table, not a scalar or unlabeled collection.")
    rows = result.get("rows", [])
    returned_ids = [column for column in identifiers if column in result.get("columns", [])]
    if metric not in result.get("columns", []):
        reject("The returned table omits the source metric needed for independent verification.")
    if not returned_ids:
        reject("The returned table omits a source item identifier needed for independent verification.")
    target_count = min(requested, len(ranked)) if requested is not None else len(rows)
    report.update(checked_rows=len(rows), identifier_columns=returned_ids, limit_source="question" if requested is not None else "returned_prefix")
    if result.get("row_count") != len(rows) or len(rows) != target_count or (len(ranked) and not rows):
        reject("The number of returned rows does not match the requested ranking prefix.")
    lookups = {column: {_identifier(value): position for position, value in enumerate(frame[column])} for column in returned_ids}
    positions, values = [], []
    for row in rows:
        located = [lookups[column].get(_identifier(row.get(column))) for column in returned_ids]
        if any(position is None for position in located) or len(set(located)) != 1:
            reject("A returned identifier is unknown or identifiers do not refer to the same source row.")
        position = located[0]
        number = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
        if not finite.iloc[position] or pd.isna(number) or not math.isfinite(float(number)) or not _equal_number(number, numeric.iloc[position]):
            reject("A returned identifier/metric pair does not match the source data.")
        for column in result.get("columns", []):
            if column in frame.columns and _normalized(column) in IDENTIFIER_ALIASES | NAME_ALIASES:
                original, returned = frame.iloc[position][column], row.get(column)
                if not (pd.isna(original) and returned is None) and _identifier(original) != _identifier(returned):
                    reject("A returned item label or identifier does not match its source row.")
        positions.append(position)
        values.append(number)
    if len(set(positions)) != len(positions):
        reject("The result repeats an individual source item.")
    if any((left > right if ascending else left < right) for left, right in zip(values, values[1:])):
        reject("The returned rows are ordered in the wrong ranking direction.")
    if target_count:
        cutoff = ranked.iloc[target_count - 1]
        if any((value > cutoff if ascending else value < cutoff) for value in values):
            reject("Returned items are outside the correct numeric ranking prefix.")
        required = set(ranked[ranked < cutoff if ascending else ranked > cutoff].index.tolist())
        if not required.issubset(set(positions)):
            reject("The result omits items that rank strictly ahead of the cutoff.")
    report.update(status="passed", rejected=False, reason="Identifier/metric pairs, ordering and numeric prefix membership match the source; any genuine cutoff ties are allowed.")
    return report
