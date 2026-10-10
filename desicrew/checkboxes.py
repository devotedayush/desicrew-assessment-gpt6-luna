"""Local, uncalibrated checkbox-interior heuristics for explicit layout regions.

No selected answer is encoded. A field is selected only from image ink; every result
requires template/selection review. Text, shadows and misplaced crops can fool it.
"""
from __future__ import annotations
from typing import Any


def inspect_box(image: Any) -> dict[str, Any]:
    from PIL import Image, ImageOps
    import numpy as np
    if not isinstance(image, Image.Image):
        raise TypeError("Checkbox crop must be a PIL image")
    width, height = image.size
    if min(width, height) < 12 or width * height > 10_000:
        return {"status": "ambiguous", "reason": "checkbox_crop_size_unsupported"}
    gray = np.asarray(ImageOps.grayscale(image))
    inset = max(2, round(min(width, height) * .30))
    interior = gray[inset:height-inset, inset:width-inset]
    threshold = min(160, max(100, int(np.percentile(interior, 85)) - 75))
    ink = interior < threshold
    original_fraction = float(ink.mean())
    removed = 0
    # Ignore continuous rules crossing the window; no digit/text OCR is involved.
    cleaned = ink.copy()
    for y, row in enumerate(ink):
        if float(row.mean()) >= .70:
            removed += int(cleaned[y].sum())
            cleaned[y] = False
    for x in range(cleaned.shape[1]):
        if float(cleaned[:, x].mean()) >= .85:
            removed += int(cleaned[:, x].sum())
            cleaned[:, x] = False
    # Tiny disconnected speckles are not check marks. Judge spatial spread of
    # one connected mark instead of combining unrelated rules/text fragments.
    visited = np.zeros_like(cleaned, dtype=bool)
    components = []
    for start_y, start_x in zip(*np.where(cleaned)):
        if visited[start_y, start_x]:
            continue
        queue = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component = []
        while queue:
            y, x = queue.pop()
            component.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < cleaned.shape[0] and 0 <= xx < cleaned.shape[1] and cleaned[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        queue.append((yy, xx))
        components.append(component)
    small_fragments = sum(len(component) for component in components if len(component) < 5)
    retained = [component for component in components if len(component) >= 5]
    cleaned[:] = False
    for component in retained:
        for y, x in component:
            cleaned[y, x] = True
    largest = max(retained, key=len, default=[])
    ys = np.asarray([point[0] for point in largest])
    xs = np.asarray([point[1] for point in largest])
    count = int(cleaned.sum())
    fraction = float(cleaned.mean())
    span_x = (int(xs.max() - xs.min() + 1) / cleaned.shape[1]) if largest else 0.
    span_y = (int(ys.max() - ys.min() + 1) / cleaned.shape[0]) if largest else 0.
    if original_fraction >= .50:
        status, reason = "selected", "filled_interior"
    elif len(largest) >= max(5, round(cleaned.size * .035)) and fraction >= .035 and min(span_x, span_y) >= .40:
        status, reason = "selected", "non_horizontal_spread_ink_mark"
    elif count <= 2 and fraction <= .015:
        status, reason = "unselected", "interior_clear_after_rule_removal"
    else:
        status, reason = "ambiguous", "interior_ink_not_a_clear_selection"
    return {"status": status, "reason": reason, "threshold": threshold, "interior_inset": inset, "original_ink_fraction": round(original_fraction, 4), "cleaned_ink_fraction": round(fraction, 4), "ink_pixels": count, "largest_component_pixels": len(largest), "ignored_small_fragment_pixels": small_fragments, "span_x": round(span_x, 3), "span_y": round(span_y, 3), "removed_rule_pixels": removed, "confidence_kind": "heuristic_quality_score_not_probability"}


def selected_option(evidence: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    selected = [option for option in evidence if option.get("status") == "selected"]
    if len(selected) > 1:
        return None, ["multiple_selected_checkboxes"]
    if any(option.get("status") == "ambiguous" for option in evidence):
        return None, ["checkbox_marks_ambiguous"]
    if len(selected) != 1:
        return None, ["no_selected_checkbox_detected"]
    return selected[0].get("description", selected[0]["label"]), ["checkbox_selection_requires_visual_review", "checkbox_template_alignment_requires_review"]
