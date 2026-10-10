#!/usr/bin/env python3
"""Generate reproducible LOCAL synthetic extraction and review artifacts.

These are artificial templates/OCR transcripts, never supplied assessment samples.
Use --input-dir for separately authorized private samples. No uploads or API calls.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desicrew.documents import DocumentProcessor, DOCUMENT_SCHEMAS, _checksum_mrz


def synthetic_mrz() -> str:
    number = "U1234567<"
    birth = "940319"
    expiry = "340319"
    optional = "<" * 14
    body = number + str(_checksum_mrz(number)) + "IND" + birth + str(_checksum_mrz(birth)) + "F" + expiry + str(_checksum_mrz(expiry)) + optional + str(_checksum_mrz(optional))
    return body + str(_checksum_mrz(body[:10] + body[13:20] + body[21:43]))


def fixture_texts() -> dict[str, str]:
    return {
        "aadhaar_card": "Aadhaar Card\nAadhaar Number: 2345 6789 0123\nFull Name: Ananya Demo\nDate of Birth: 19/03/1994\nAddress: 42 Example Lane\nDemonstration Nagar\nPune 411001",
        "pan_card": "PAN Card\nPAN Number: DEMPP4321Q\nFull Name: Ananya Demo\nFather's Name: Arun Demo\nDate of Birth: 19/03/1994",
        "driving_licence": "Driving Licence\nDL Number: MH12 2012 0000123\nName: Ananya Demo\nDate of Issue: 19/03/2012\nValid Till: 19/03/2032",
        "passport": "Passport\nPassport Number: U1234567\nDate of Birth: 19/03/1994\nDate of Expiry: 19/03/2034\nMRZ Line 2: " + synthetic_mrz(),
        "nach_ecs_mandate": "NACH ECS Mandate\nBank Account Number: 000123456789\nIFSC Code: DEMO0123456\nBank Name: Example Bank\nAmount (figures): 25,000\nFrequency: [x] Monthly [ ] Quarterly",
        "fatca_annexure_form": "FATCA Annexure Form\nPolicy Number: 0000012345\nTIN / PAN: DEMPP4321Q\nFather's Name: Arun Demo\nPlace of Birth: Pune\nNationality: Indian",
        "benefit_illustration_declaration": "Benefit Illustration Declaration\nApplication Number: 0000054321\nPolicyholder Name: Ananya Demo\nDate: 19/03/2026\nPlace: Pune",
        "moral_hazard_questionnaire": "Moral Hazard Questionnaire\nApplication Number: 0000054321\nName of Life Assured: Ananya Demo\nNominee Relationship: Spouse\nDate: 19/03/2026\nPlace: Pune",
        "multiple_policies_consent_form": "Multiple Policies Consent Form\nProposer Name: Ananya Demo\nReason for Multiple Policies: [ ] Investment [x] Family protection [ ] Tax benefit\nDate: 19/03/2026\nPlace: Pune",
        "suitability_profiler_declaration": "Suitability Profiler Declaration\nApplication Number: 0000054321\nName of Life Assured: Ananya Demo\nName of Agent/SP: Rohan Example\nDate: 19/03/2026\nPlace: Pune",
    }


def create_fixtures(directory: Path, render_images: bool = False) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    header = "SYNTHETIC TEST FIXTURE - ARTIFICIAL PERSON AND VALUES\nNOT A GOVERNMENT DOCUMENT / NOT AN ASSESSMENT SAMPLE\n\n"
    for name, text in fixture_texts().items():
        path = directory / f"{name}.txt"
        path.write_text(header + text + "\n", encoding="utf-8")
        paths.append(path)
        if render_images:
            from PIL import Image, ImageDraw, ImageFont
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, 24) if Path(font_path).exists() else ImageFont.load_default(size=24)
            image = Image.new("RGB", (1800, 950), "white")
            ImageDraw.Draw(image).multiline_text((45, 45), header + text, fill="black", font=font, spacing=16)
            image.save(directory / f"{name}.png")
    return paths


def make_handwriting_backend(kind: str, model_path: Path | None) -> Any:
    """Construct lazily from explicit existing weights; never download or load here."""
    if kind == "vision":
        from desicrew.mac_vision import LocalMacVisionRecognizer
        return LocalMacVisionRecognizer()
    if model_path is None:
        return None
    if kind == "got":
        from desicrew.got_ocr import LocalGotOcrRecognizer
        return LocalGotOcrRecognizer(model_path, max_new_tokens=64, max_seconds=20, threads=2)
    if kind == "trocr":
        from desicrew.handwriting import LocalHandwritingRecognizer
        return LocalHandwritingRecognizer(model_path, max_new_tokens=64, max_seconds=15, threads=2)
    raise ValueError("handwriting backend must be trocr, got or vision")


def process_staged_files(paths: list[Path], processor: DocumentProcessor, backend: Any,
                         output_dir: Path, *, use_checkbox_detection: bool,
                         staged_results_path: Path | None = None) -> list[dict]:
    """Finish Tesseract/printed crops before a large GOT model becomes resident.

    This avoids concurrent Tesseract child memory during GOT inference. No reference
    file is read. All private outputs stay inside the supervised worker directory.
    """
    if staged_results_path is None:
        results = [processor.process_file(path) for path in paths]
    else:
        private_root = (ROOT / "outputs/private_samples").resolve()
        if not staged_results_path.resolve().is_relative_to(private_root) or staged_results_path.name != "tesseract_stage_results_private.json":
            raise ValueError("GOT staged results must be a previous private Tesseract-stage file")
        staged = json.loads(staged_results_path.read_text())
        by_name = {result["source_file"]: result for result in staged}
        if len(by_name) != len(staged) or any(path.name not in by_name for path in paths):
            raise ValueError("Staged results must uniquely cover every selected source filename")
        results = [by_name[path.name] for path in paths]
        if any(result.get("ocr", {}).get("engine") != "local_tesseract" for result in results):
            raise ValueError("Selected staged results must come from local Tesseract")
    (output_dir / "tesseract_stage_results_private.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    from desicrew.got_ocr import release_cpu_allocator
    release_cpu_allocator()
    processor.handwriting_backend = backend
    predictions = []
    underlying_recognize = backend.recognize
    current_source = None

    def record_recognition(image, *, field=None, page=1, bbox=None):
        prediction = underlying_recognize(image, field=field, page=page, bbox=bbox)
        predictions.append({"crop_id": f"canonical_roi_{len(predictions)+1:02d}", "source_file": current_source,
                            "field": field, "recognition": prediction, "reference_verified": False,
                            "exact_match": None, "character_edits": None, "reference_characters": None})
        (output_dir / "partial_predictions_private.json").write_text(json.dumps(predictions, indent=2, ensure_ascii=False))
        print(f"Guarded GOT: {len(predictions)} explicit value-line crops processed", flush=True)
        return prediction

    backend.recognize = record_recognition
    completed = []
    try:
        for path, result in zip(paths, results):
            current_source = result["source_file"]
            result = processor._augment_handwriting(path, result)
            if use_checkbox_detection:
                result = processor._augment_checkboxes(path, result)
            completed.append(result)
            (output_dir / "partial_results_private.json").write_text(json.dumps(completed, indent=2, ensure_ascii=False))
    finally:
        backend.recognize = underlying_recognize
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Use an authorized local sample directory instead of synthetic transcripts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "documents_synthetic")
    parser.add_argument("--render-images", action="store_true", help="Also create synthetic PRINTED PNGs; this does not simulate handwriting")
    parser.add_argument("--ocr-images", action="store_true", help="Run local Tesseract on synthetic printed PNGs")
    parser.add_argument("--threshold", type=float, default=.90)
    parser.add_argument("--handwriting-model", type=Path, help="Explicit already-downloaded local handwriting model directory; never downloads weights")
    parser.add_argument("--handwriting-backend", choices=["trocr", "got", "vision"], default="trocr", help="Explicit local backend; GOT uses a guarded worker, Vision requires macOS")
    parser.add_argument("--roi-manifest", type=Path, help="Explicit local value-line crop manifest, required with --handwriting-model")
    parser.add_argument("--use-layout-regions", action="store_true", help="Opt in to geometry-only supplied-layout templates; every candidate requires alignment review")
    parser.add_argument("--handwritten-region-ocr", action="store_true", help="Also use local Tesseract on handwritten value crops; not recommended together with a handwriting model")
    parser.add_argument("--region-preprocessing", choices=["none", "contrast_rule_cleanup"], default="none", help="Optional destructive rule cleanup, always review-only")
    parser.add_argument("--detect-checkboxes", action="store_true", help="Opt in to explicit layout-box pixel heuristics; no selected answer is encoded")
    parser.add_argument("--batch-timeout", type=int, default=600, help="GOT hard wall limit, at most 600 seconds")
    parser.add_argument("--memory-budget-mb", type=int, default=4096, help="GOT RSS watchdog limit, at most 4096MB")
    parser.add_argument("--staged-results", type=Path, help="GOT only: reuse a previous private Tesseract stage for selected source filenames")
    parser.add_argument("--got-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.handwriting_backend == "got":
        if not args.handwriting_model or not args.input_dir:
            parser.error("GOT requires --handwriting-model and an authorized local --input-dir")
        if args.handwritten_region_ocr:
            parser.error("GOT mode does not mix handwritten Tesseract crop candidates; use printed regions and explicit model crops")
        if not 1 <= args.batch_timeout <= 600 or not 512 <= args.memory_budget_mb <= 4096:
            parser.error("GOT is bounded to 1..600 seconds and 512..4096MB")
        if not args.got_worker:
            from scripts.evaluate_got_ocr import ensure_private_dir, supervise
            destination = ensure_private_dir(args.output_dir)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--got-worker"]
            supervise(command, destination, args.batch_timeout, args.memory_budget_mb, 26)
            return
    elif args.staged_results:
        parser.error("--staged-results is only for guarded GOT runs")
    if args.handwriting_backend == "vision" and (args.handwriting_model or not (args.roi_manifest or args.use_layout_regions)):
        parser.error("Vision uses the local macOS framework without model weights and requires --roi-manifest or --use-layout-regions")
    if args.input_dir:
        if not args.input_dir.is_dir():
            parser.error("input directory does not exist")
        paths = sorted(path for path in args.input_dir.rglob("*") if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".txt"})
        mode = "supplied_local_files"
    else:
        paths = create_fixtures(ROOT / "data" / "synthetic" / "documents", args.render_images or args.ocr_images)
        if args.ocr_images:
            paths = [path.with_suffix(".png") for path in paths]
        mode = "synthetic_printed_image_ocr" if args.ocr_images else "synthetic_text_transcripts"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handwriting_backend = None
    if args.handwriting_model or args.handwriting_backend == "vision":
        if not args.roi_manifest and not args.use_layout_regions:
            parser.error("--handwriting-model requires --roi-manifest or --use-layout-regions with explicit value-line regions")
        handwriting_backend = make_handwriting_backend(args.handwriting_backend, args.handwriting_model)
    from desicrew.regions import SAMPLE_REGION_TEMPLATES
    is_got = args.handwriting_backend == "got"
    processor = DocumentProcessor(args.threshold, handwriting_backend=None if is_got else handwriting_backend, roi_manifest_path=args.roi_manifest, region_templates=SAMPLE_REGION_TEMPLATES if args.use_layout_regions or args.detect_checkboxes else None, use_printed_region_ocr=args.use_layout_regions, use_handwritten_region_ocr=args.handwritten_region_ocr, region_preprocessing=args.region_preprocessing, use_checkbox_detection=False if is_got else args.detect_checkboxes)
    results = process_staged_files(paths, processor, handwriting_backend, args.output_dir,
                                   use_checkbox_detection=args.detect_checkboxes,
                                   staged_results_path=args.staged_results) if is_got else [processor.process_file(path) for path in paths]
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    for index, result in enumerate(results, 1):
        output = args.output_dir / f"{index:02d}_{result['document_type']}.json"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = processor.review_report(results)
    report["evaluation_mode"] = mode
    report["sample_accuracy"] = None
    report["sample_accuracy_note"] = "This run does not read an answer key or visual reference. Score automatic outputs separately; synthetic fixtures are engineering checks only"
    (args.output_dir / "review_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"mode": mode, "documents_processed": len(results), "required_fields_emitted": sum(len(result["fields"]) for result in results), "documents_needing_review": report["documents_needing_review"], "fields_needing_review": report["fields_needing_review"], "output_directory": str(args.output_dir)}
    if is_got:
        summary.update({"status": "completed", "handwriting_backend": "got", "staged_before_model_load": True,
                        "reused_prior_tesseract_stage": bool(args.staged_results),
                        "no_automatic_weights_download": True, "no_references_used_to_fill_outputs": True,
                        "wall_budget_seconds": args.batch_timeout, "memory_budget_mb": args.memory_budget_mb})
        (args.output_dir / "summary_without_field_values.json").write_text(json.dumps({key: value for key, value in summary.items() if key != "output_directory"}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
