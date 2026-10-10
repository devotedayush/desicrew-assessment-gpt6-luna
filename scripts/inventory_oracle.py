"""Evaluation-only reference calculations, independent of pandas query planning.

This module never generates agent answers. It is not imported by the application.
Ranks use Python sorting and scalar totals use math.fsum over original records.
"""
from __future__ import annotations

import math


def finite_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def total(records, column):
    return math.fsum(number for row in records if (number := finite_number(row.get(column))) is not None)


def verify_scalar(result, expected):
    actual = finite_number((result or {}).get("value"))
    return bool((result or {}).get("type") == "scalar" and actual is not None and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9))


def verify_ranking(result, records, id_column, metric_column, count, ascending=True):
    """Check identifier/metric pairs and cutoff membership, permitting true ties."""
    eligible = [(row[id_column], number) for row in records if (number := finite_number(row.get(metric_column))) is not None]
    ranked = sorted(eligible, key=lambda pair: pair[1], reverse=not ascending)
    expected_count = min(count, len(ranked))
    result = result or {}
    if result.get("type") != "table" or result.get("truncated") or result.get("complete") is False:
        return False
    rows = result.get("rows", [])
    if len(rows) != expected_count or result.get("row_count") != expected_count:
        return False
    if id_column not in result.get("columns", []) or metric_column not in result.get("columns", []):
        return False
    # Duplicate identifiers in the reference are ambiguous, not arbitrarily collapsed.
    source = {str(identifier): number for identifier, number in eligible}
    if len(source) != len(eligible):
        return False
    returned = []
    for row in rows:
        identifier = str(row.get(id_column))
        number = finite_number(row.get(metric_column))
        if identifier not in source or number is None or not math.isclose(number, source[identifier], rel_tol=1e-12, abs_tol=1e-9):
            return False
        returned.append((identifier, number))
    if len({identifier for identifier, _ in returned}) != len(returned):
        return False
    values = [number for _, number in returned]
    if values != sorted(values, reverse=not ascending):
        return False
    if expected_count:
        cutoff = ranked[expected_count - 1][1]
        if any(number > cutoff if ascending else number < cutoff for number in values):
            return False
        required = {str(identifier) for identifier, number in ranked if (number < cutoff if ascending else number > cutoff)}
        if not required.issubset({identifier for identifier, _ in returned}):
            return False
    return True
