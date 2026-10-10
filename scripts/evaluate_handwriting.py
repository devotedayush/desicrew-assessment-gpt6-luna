"""Explicit public-weight setup and private, bounded cropped-line evaluation.

No supplied image is uploaded, and no authentication token is read. Model setup
uses a pinned official Microsoft revision; inference is offline and CPU-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desicrew.handwriting import LocalHandwritingRecognizer, MODEL_ID, MODEL_REVISION, BASE_MODEL_ID, BASE_MODEL_REVISION

PUBLIC_FILES = (
    'config.json', 'generation_config.json', 'preprocessor_config.json',
    'sentencepiece.bpe.model', 'special_tokens_map.json', 'tokenizer_config.json',
    'pytorch_model.bin',
)
EXPECTED_SHA256 = {
    'pytorch_model.bin': '1b83102cbc1520dee1c3937ac334da83da18cf4683d46ebd1bd4e93ebe584dd7',
    'sentencepiece.bpe.model': '6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4',
}


BASE_FILES = ("config.json", "generation_config.json", "preprocessor_config.json",
              "special_tokens_map.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")
BASE_EXPECTED_SHA256 = {"model.safetensors": "25a40cddc7e6120140a3d5b9e3dd3878a92ada7b4f312953ab22edc19c2a5acc"}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_public_model(destination: Path, model_choice: str = "small") -> dict:
    """Download only fixed public model files, verify LFS hashes, convert safely."""
    import torch
    from safetensors.torch import save_file
    # Older torch weights_only implementations have a known security flaw.
    version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
    if version < (2, 6):
        raise RuntimeError('Setup requires torch >=2.6 for safe weights-only conversion')
    if model_choice not in {"small", "base"}:
        raise ValueError("Only the explicitly supported official small/base checkpoints are allowed")
    model_id, revision = (MODEL_ID, MODEL_REVISION) if model_choice == "small" else (BASE_MODEL_ID, BASE_MODEL_REVISION)
    files = PUBLIC_FILES if model_choice == "small" else BASE_FILES
    expected_hashes = EXPECTED_SHA256 if model_choice == "small" else BASE_EXPECTED_SHA256
    size_budget = 270_000_000 if model_choice == "small" else 1_340_000_000
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in files:
        target = destination / name
        if not target.exists():
            url = f'https://huggingface.co/{model_id}/resolve/{revision}/{name}'
            temporary = target.with_suffix(target.suffix + '.part')
            size = 0
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as stream:
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > size_budget:
                            raise RuntimeError('Public model file exceeded fixed download-size budget')
                        stream.write(chunk)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        digest = file_sha256(target)
        if name in expected_hashes and digest != expected_hashes[name]:
            raise RuntimeError(f'Official pinned-weight hash mismatch: {name}')
        hashes[name] = digest
        print(f'Public model setup: verified {name}', flush=True)
    if model_choice == "small":
        state = torch.load(destination / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
            raise RuntimeError('Unexpected public state-dict structure')
        safe_path = destination / 'model.safetensors'
        # Clone tied entries to avoid aliasing without changing their values.
        save_file({key: value.detach().cpu().contiguous().clone() for key, value in state.items()}, str(safe_path), metadata={'format': 'pt'})
        hashes['model.safetensors'] = file_sha256(safe_path)
    manifest = {'model_id': model_id, 'revision': revision, 'public_source': f'https://huggingface.co/{model_id}',
                'source_format': 'official_pytorch_weights_only' if model_choice == 'small' else 'official_safetensors',
                'runtime_format': 'safetensors', 'sha256': hashes}
    (destination / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Public model setup: safe weights verified; runtime is offline', flush=True)
    return manifest


def normalize(text: str) -> str:
    """Case/whitespace-only comparison, no character repair or invented reference."""
    return ' '.join(text.upper().split())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def ensure_private_output_dir(path: Path) -> Path:
    private_root = (ROOT / 'outputs/private_samples').resolve()
    candidate = path.resolve()
    if not candidate.is_relative_to(private_root):
        raise ValueError('Evaluation artifacts may only be written under outputs/private_samples')
    return candidate


def evaluate(manifest_path: Path, model_path: Path, output_dir: Path, limit: int, max_seconds: float, threads: int, preprocessing: str = "none") -> dict:
    from PIL import Image
    output_dir = ensure_private_output_dir(output_dir)
    manifest = json.loads(manifest_path.read_text())
    crops = manifest.get('crops', [])[:limit]
    if not crops:
        raise ValueError('Evaluation manifest must contain a nonempty crops array')
    recognizer = LocalHandwritingRecognizer(model_path, max_seconds=max_seconds, threads=threads, preprocessing=preprocessing)
    results = []
    for index, crop in enumerate(crops, 1):
        path = Path(crop['crop_path'])
        if not path.is_absolute():
            path = ROOT / path
        with Image.open(path) as original:
            recognized = recognizer.recognize(original.copy(), field=crop.get('field'), page=crop.get('page', 1), bbox=crop.get('bbox_xyxy'))
        reference = crop.get('reference_text')
        comparable = bool(crop.get('visually_verified') and isinstance(reference, str) and reference.strip())
        expected = normalize(reference) if comparable else None
        predicted = normalize(recognized['text'])
        results.append({'crop_id': crop['id'], 'source_file': Path(crop['source']).name,
                        'field': crop.get('field'), 'recognition': recognized,
                        'visual_reference': reference, 'reference_verified': comparable,
                        'comparison_kind': 'case_and_whitespace_only_no_character_repair',
                        'exact_match': predicted == expected if comparable else None,
                        'character_edits': edit_distance(expected, predicted) if comparable else None,
                        'reference_characters': len(expected) if comparable else None,
                        'crop_concerns': crop.get('concerns', [])})
        print(f'Local handwriting evaluation: {index}/{len(crops)} crops processed', flush=True)
    references = [row for row in results if row['reference_verified']]
    errors = sum(row['character_edits'] for row in references)
    characters = sum(row['reference_characters'] for row in references)
    summary = {'model_actually_ran': True, 'model': recognizer.model_id, 'model_revision': recognizer.model_revision,
               'device': 'cpu', 'preprocessing': preprocessing,
               'runtime_versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'safetensors', 'sentencepiece', 'Pillow')}, 'external_image_uploads': 0, 'paid_calls': 0,
               'crops_evaluated': len(results), 'source_documents': len({row['source_file'] for row in results}),
               'visually_readable_references': len(references), 'excluded_ambiguous_references': len(results) - len(references),
               'exact_line_matches_case_whitespace_normalized': sum(row['exact_match'] for row in references),
               'aggregate_character_edits': errors, 'aggregate_reference_characters': characters,
               'character_error_rate': round(errors / characters, 4) if characters else None,
               'generation_budget_failures': sum(row['recognition']['terminated_early'] for row in results),
               'all_model_results_require_review': all(row['recognition']['needs_review'] for row in results),
               'elapsed_inference_seconds': round(sum(row['recognition']['elapsed_seconds'] for row in results), 3),
               'evaluation_limitations': ['Small manually selected crop set, not the official answer key',
                                          'Visual references can themselves be uncertain; null references excluded',
                                          'Repeated names and places reduce diversity',
                                          'Line-level metrics do not establish complete field/pipeline accuracy',
                                          'IAM-trained English model has domain shift on Indian insurance forms']}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'handwriting_line_evaluation.json').write_text(json.dumps({'summary': summary, 'results': results}, indent=2) + '\n')
    (output_dir / 'handwriting_line_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--model-size', choices=['small', 'base'], default='small', help='Official checkpoint selection for explicit setup only')
    parser.add_argument('--download-model', action='store_true', help='Explicitly download pinned public weights only; no private data transmission')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'outputs/private_samples/handwriting_crops.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/private_samples')
    parser.add_argument('--limit', type=int, default=18)
    parser.add_argument('--max-seconds', type=float, default=15)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--preprocessing', choices=['none', 'contrast_rule_cleanup'], default='none', help='Optional diagnostic cleanup can damage touching strokes; review-only')
    parser.add_argument('--batch-timeout', type=int, default=600, help='Hard timeout for offline evaluation subprocess, includes model loading')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.model_path is None:
        args.model_path = ROOT / f"data/private/models/trocr-{args.model_size}-handwritten"
    if not 1 <= args.limit <= 100 or not 1 <= args.batch_timeout <= 3600:
        parser.error('limit must be 1..100 and batch-timeout 1..3600')
    if args.download_only and not args.download_model:
        parser.error('--download-only requires --download-model')
    if args.download_model:
        download_public_model(args.model_path, args.model_size)
    if args.download_only:
        return
    if args.worker:
        evaluate(args.manifest, args.model_path, args.output_dir, args.limit, args.max_seconds, args.threads, args.preprocessing)
        return
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--model-path', str(args.model_path),
               '--manifest', str(args.manifest), '--output-dir', str(args.output_dir), '--limit', str(args.limit),
               '--max-seconds', str(args.max_seconds), '--threads', str(args.threads), '--preprocessing', args.preprocessing, '--batch-timeout', str(args.batch_timeout)]
    environment = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
    try:
        subprocess.run(command, env=environment, check=True, timeout=args.batch_timeout)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit('Offline handwriting batch exceeded its hard process timeout; no successful evaluation claim') from exc


if __name__ == '__main__':
    main()
