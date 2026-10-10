#!/usr/bin/env python3
"""One bounded native GOT-OCR2 diagnostic; public files in, private pixels stay local."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desicrew.got_ocr import LocalGotOcrRecognizer, MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256

# Official Hugging Face repository API blob metadata, retrieved 2026-10-07.
# No Python, pickle, or arbitrary repository files are permitted.
PUBLIC_FILES = {
    "config.json": (608, "git_sha1", "7ef8853e38fd8f079d1c3fdb876811aafc889db4"),
    "generation_config.json": (74, "git_sha1", "ad351c881ba0f8894f365a3585560e9e27c237a3"),
    "preprocessor_config.json": (439, "git_sha1", "e8a43aa873e54e03b1d24ab2dd9fb7a7ae5963a8"),
    "special_tokens_map.json": (213, "git_sha1", "83d5f7dc41b50e3755784ddf2a748d5d07eac7da"),
    "tokenizer_config.json": (39228, "git_sha1", "646ebbd36ef2eb39a0b3ffbe279b6e3d8b88b9a7"),
    "tokenizer.json": (18702549, "sha256", "36b382a3c48c9a143c30139dac6c8230ddfb0b46a3dc43082af6052abe99d9de"),
    "model.safetensors": (1121114488, "sha256", WEIGHTS_SHA256),
}


def ensure_private_dir(path: Path, *, model: bool = False) -> Path:
    root = (ROOT / ("data/private/models" if model else "outputs/private_samples")).resolve()
    path = path.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Use a dedicated directory within the permitted private artifact root")
    return path


def hash_file(path: Path, method: str = "sha256") -> str:
    if method == "git_sha1":
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_public_model(destination: Path) -> dict:
    destination = ensure_private_dir(destination, model=True)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, (expected_size, method, expected_digest) in PUBLIC_FILES.items():
        target = destination / name
        if not target.exists():
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            temporary = target.with_suffix(target.suffix + ".part")
            count = 0
            try:
                # Public URL only. No Hugging Face token or saved credentials.
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > expected_size:
                            raise RuntimeError(f"Official file exceeds pinned byte budget: {name}")
                        stream.write(chunk)
                if count != expected_size:
                    raise RuntimeError(f"Official file does not match pinned byte count: {name}")
                if hash_file(temporary, method) != expected_digest:
                    raise RuntimeError(f"Official pinned file hash mismatch: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        if target.stat().st_size != expected_size or hash_file(target, method) != expected_digest:
            raise RuntimeError(f"Local file differs from official pinned hash: {name}")
        hashes[name] = hash_file(target)
        print(f"Verified official pinned public file: {name}", flush=True)
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "public_source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
                "official_metadata_source": f"https://huggingface.co/api/models/{MODEL_ID}?blobs=true",
                "source_format": "official_safetensors", "runtime_format": "safetensors",
                "official_hashes": {name: {"method": method, "hash": digest, "bytes": size}
                                    for name, (size, method, digest) in PUBLIC_FILES.items()},
                "sha256": hashes, "downloaded_custom_python": False, "authentication_used": False}
    (destination / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def normalize(text: str) -> str:
    return " ".join(text.upper().split())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def evaluate(manifest_path: Path, model_path: Path, output_dir: Path, *, limit: int,
             max_seconds: float, threads: int) -> dict:
    from PIL import Image
    output_dir = ensure_private_dir(output_dir)
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_relative_to((ROOT / "outputs/private_samples").resolve()):
        raise ValueError("Only a private crop manifest is permitted")
    manifest = json.loads(manifest_path.read_text())
    crops = manifest.get("crops", [])[:limit]
    if not crops or any(not crop.get("visually_verified") for crop in crops):
        raise ValueError("Use only a nonempty list of explicitly inspected value-line crops")
    output_dir.mkdir(parents=True, exist_ok=True)
    recognizer = LocalGotOcrRecognizer(model_path, max_seconds=max_seconds, threads=threads)
    results = []
    started = time.monotonic()
    for index, crop in enumerate(crops, 1):
        path = Path(crop["crop_path"])
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if not path.is_relative_to((ROOT / "outputs/private_samples").resolve()):
            raise ValueError("Crop path must stay within the private sample directory")
        with Image.open(path) as image:
            prediction = recognizer.recognize(image.copy(), field=crop.get("field"),
                                             page=crop.get("page", 1), bbox=crop.get("bbox_xyxy"))
        # References are accessed only AFTER inference, exclusively for scoring.
        reference = crop.get("reference_text")
        comparable = isinstance(reference, str) and bool(reference.strip())
        expected = normalize(reference) if comparable else None
        predicted = normalize(prediction["text"])
        results.append({"crop_id": crop["id"], "source_file": Path(crop["source"]).name,
                        "field": crop.get("field"), "recognition": prediction,
                        "visual_reference": reference, "reference_verified": comparable,
                        "exact_match": predicted == expected if comparable else None,
                        "character_edits": edit_distance(expected, predicted) if comparable else None,
                        "reference_characters": len(expected) if comparable else None,
                        "crop_concerns": crop.get("concerns", [])})
        (output_dir / "partial_predictions_private.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Offline GOT diagnostic: {index}/{len(crops)} inspected crops processed", flush=True)
    references = [row for row in results if row["reference_verified"]]
    edits = sum(row["character_edits"] for row in references)
    characters = sum(row["reference_characters"] for row in references)
    times = [row["recognition"]["elapsed_seconds"] for row in results]
    summary = {"purpose": "bounded_private_visual_reference_diagnostic_not_official_accuracy",
               "model_actually_ran": True, "status": "completed", "completed_requested_subset": True,
               "model": MODEL_ID,
               "model_revision": MODEL_REVISION, "weights_hash_verified": recognizer.weights_hash_verified,
               "device": "cpu", "dtype": "bfloat16", "cpu_threads": threads,
               "runtime_versions": {name: importlib.metadata.version(name) for name in
                                    ("torch", "transformers", "safetensors", "accelerate", "Pillow")},
               "external_image_uploads": 0, "paid_calls": 0, "custom_repository_code_executed": False,
               "crops_evaluated": len(results), "source_documents": len({r["source_file"] for r in results}),
               "visually_readable_references": len(references),
               "excluded_ambiguous_references": len(results) - len(references),
               "exact_line_matches_case_whitespace_normalized": sum(r["exact_match"] for r in references),
               "aggregate_character_edits": edits, "aggregate_reference_characters": characters,
               "character_error_rate": round(edits / characters, 4) if characters else None,
               "generation_budget_failures": sum(r["recognition"]["terminated_early"] for r in results),
               "all_model_results_require_review": all(r["recognition"]["needs_review"] for r in results),
               "elapsed_inference_seconds": round(sum(times), 3),
               "median_crop_seconds": round(statistics.median(times), 3),
               "slowest_crop_seconds": max(times),
               "elapsed_worker_seconds_including_load": round(time.monotonic() - started, 3),
               "peak_worker_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               "cpu_memory_cleanup": "per_crop_tensor_release_gc_collect_and_optional_glibc_malloc_trim",
               "generation_metadata_returned": False, "decoder_scores_or_logits_retained": False,
               "peak_post_cleanup_rss_mb": max((row["recognition"].get("memory_management", {}).get("rss_after_cleanup_mb") or 0)
                                                for row in results),
               "comparison_kind": "case_and_whitespace_only_no_character_repair",
               "limitations": [f"{len(results)} manually selected crops, {len(references)} readable references; not official answer key",
                               "Crop fragments include some fields beyond the brief; not a 41-field pipeline score",
                               "Repeated names/places and narrow single-line crops reduce diversity",
                               "No reference value fills or repairs predictions; all results need review",
                               "General OCR model may have domain/aspect-ratio shift on boxed handwriting"]}
    (output_dir / "got_line_evaluation_private.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
    (output_dir / "summary_without_field_values.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def summarize_partial_results(output_dir: Path, status: dict, planned_crops: int = 18) -> dict:
    """Aggregate completed predictions after a failed batch; never imply completion."""
    path = output_dir / "partial_predictions_private.json"
    rows = json.loads(path.read_text()) if path.is_file() else []
    references = [row for row in rows if row["reference_verified"]]
    characters = sum(row["reference_characters"] for row in references)
    edits = sum(row["character_edits"] for row in references)
    times = [row["recognition"]["elapsed_seconds"] for row in rows]
    summary = {"purpose": "failed_bounded_diagnostic_partial_prefix_not_official_accuracy",
               "accuracy_claim": False, "completed_full_diagnostic": False,
               "model_actually_ran": bool(rows), "status": status["status"],
               "model": MODEL_ID, "model_revision": MODEL_REVISION,
               "device": "cpu", "dtype": "bfloat16", "cpu_threads": 2,
               "planned_crops": planned_crops, "completed_crops": len(rows),
               "unprocessed_crops": max(0, planned_crops - len(rows)),
               "readable_completed_references": len(references),
               "exact_line_matches_case_whitespace_normalized": sum(row["exact_match"] for row in references),
               "aggregate_character_edits": edits, "aggregate_reference_characters": characters,
               "character_error_rate_on_completed_prefix": round(edits / characters, 4) if characters else None,
               "elapsed_completed_crop_inference_seconds": round(sum(times), 3),
               "median_completed_crop_seconds": round(statistics.median(times), 3) if times else None,
               "slowest_completed_crop_seconds": max(times) if times else None,
               "completed_generation_budget_failures": sum(row["recognition"]["terminated_early"] for row in rows),
               "all_completed_results_require_review": all(row["recognition"]["needs_review"] for row in rows),
               "weights_hash_verified": bool(rows) and all(row["recognition"]["weights_hash_verified"] for row in rows),
               "external_image_uploads": 0, "paid_calls": 0,
               "custom_repository_code_executed": False, "resource_report": status,
               "limitations": ["Failed batch: only completed prefix scored, no full-set ranking or 41-field pipeline claim",
                               "No reference value fills or repairs outputs; every prediction requires human review",
                               "Unprocessed date/TIN targets cannot be claimed evaluated"]}
    (output_dir / "partial_summary_without_field_values.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def attach_process_resource_evidence(output_dir: Path, status: dict) -> None:
    """Keep successful summary evidence consistent without changing predictions."""
    if status.get("status") != "completed":
        raise ValueError("Failed trials must use a partial report, never a completed summary")
    summary_path = output_dir / "summary_without_field_values.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        summary["process_resource_report"] = status
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        evaluation_path = output_dir / "got_line_evaluation_private.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text())
            evaluation["summary"] = summary
            evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n")


def supervise(command: list[str], output_dir: Path, seconds: int, memory_mb: int,
              planned_crops: int = 18) -> None:
    """Hard wall deadline plus measured RSS watchdog; no virtual-memory guess."""
    if any((output_dir / name).exists() for name in
           ("partial_predictions_private.json", "got_line_evaluation_private.json",
            "process_resource_report.json", "partial_summary_without_field_values.json")):
        raise ValueError("Choose a fresh private --output-dir; existing trial evidence must not be mixed or overwritten")
    import psutil
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
                   "MALLOC_ARENA_MAX": "2"}
    start = time.monotonic()
    child = subprocess.Popen(command, env=environment)
    peak = 0
    blocker = None
    try:
        while child.poll() is None:
            try:
                process = psutil.Process(child.pid)
                rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
                peak = max(peak, rss)
                if rss > memory_mb * 1024 * 1024:
                    blocker = "memory_budget_exceeded"
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - start > seconds:
                blocker = "hard_process_deadline_exceeded"
            if blocker:
                child.kill()
                child.wait()
                break
            time.sleep(.5)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    status = {"status": blocker or ("completed" if child.returncode == 0 else "worker_failed"),
              "return_code": child.returncode, "wall_seconds": round(time.monotonic() - start, 3),
              "peak_supervised_rss_mb": round(peak / (1024 * 1024), 1),
              "wall_budget_seconds": seconds, "memory_budget_mb": memory_mb}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "process_resource_report.json").write_text(json.dumps(status, indent=2) + "\n")
    if status["status"] != "completed":
        summarize_partial_results(output_dir, status, planned_crops)
        raise SystemExit(json.dumps(status))
    attach_process_resource_evidence(output_dir, status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/private/models/got-ocr2")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/private_samples/handwriting_crops.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/private_samples/got_ocr2")
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--max-seconds", type=float, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-timeout", type=int, default=600)
    parser.add_argument("--memory-budget-mb", type=int, default=4096)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.limit <= 18 or not 1 <= args.batch_timeout <= 600 or not 512 <= args.memory_budget_mb <= 4096:
        parser.error("Bounded diagnostic: limit1..18, timeout1..600seconds, memory512..4096MB")
    if args.download_only and not args.download_model:
        parser.error("--download-only requires --download-model")
    output_dir = ensure_private_dir(args.output_dir)
    if args.download_model:
        download_public_model(args.model_path)
    if args.download_only:
        return
    if args.worker:
        evaluate(args.manifest, args.model_path, output_dir, limit=args.limit,
                 max_seconds=args.max_seconds, threads=args.threads)
        return
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--model-path", str(args.model_path),
               "--manifest", str(args.manifest), "--output-dir", str(output_dir), "--limit", str(args.limit),
               "--max-seconds", str(args.max_seconds), "--threads", str(args.threads)]
    supervise(command, output_dir, args.batch_timeout, args.memory_budget_mb, args.limit)


if __name__ == "__main__":
    main()
