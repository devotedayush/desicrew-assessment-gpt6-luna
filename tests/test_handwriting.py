"""Offline adapter validation. These tests do not claim model accuracy."""
import math
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

from PIL import Image
from desicrew.handwriting import LocalHandwritingRecognizer
from scripts.evaluate_handwriting import ROOT, ensure_private_output_dir, normalize, edit_distance


class HandwritingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / 'model.safetensors').write_bytes(b'placeholder-not-loaded')

    def tearDown(self):
        self.tmp.cleanup()

    def recognizer(self, **kwargs):
        return LocalHandwritingRecognizer(self.model, **kwargs)

    def test_evaluation_refuses_public_output_destination(self):
        with self.assertRaises(ValueError):
            ensure_private_output_dir(ROOT / 'docs')
        self.assertEqual(ensure_private_output_dir(ROOT / 'outputs/private_samples/model_qa'),
                         (ROOT / 'outputs/private_samples/model_qa').resolve())

    def test_visual_comparison_does_not_repair_characters(self):
        self.assertEqual(normalize('  Example\nTown  '), 'EXAMPLE TOWN')
        self.assertEqual(normalize('O 0 I 1'), 'O 0 I 1')
        self.assertEqual(edit_distance('O0I1', '00I1'), 1)

    def test_reject_remote_model_identifier(self):
        with self.assertRaises(ValueError):
            LocalHandwritingRecognizer('microsoft/trocr-small-handwritten')

    def test_requires_safe_weights(self):
        (self.model / 'model.safetensors').unlink()
        with self.assertRaises(ValueError):
            self.recognizer()

    def test_model_identity_comes_from_local_download_manifest(self):
        metadata = {'model_id': 'microsoft/trocr-base-handwritten', 'revision': 'pinned-base-revision',
                    'sha256': {'model.safetensors': 'a' * 64}}
        (self.model / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(metadata))
        obj = self.recognizer()
        self.assertEqual(obj.model_id, metadata['model_id'])
        self.assertEqual(obj.model_revision, metadata['revision'])
        self.assertFalse(obj.weights_hash_verified)

    def test_hash_mismatch_rejected_before_optional_model_loading(self):
        metadata = {'sha256': {'model.safetensors': '0' * 64}}
        (self.model / 'DOWNLOAD_MANIFEST.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(RuntimeError, 'hash no longer matches'):
            self.recognizer()._load()

    def test_unknown_local_source_is_not_mislabeled_as_small(self):
        obj = self.recognizer()
        self.assertEqual(obj.model_id, 'local_checkpoint_unverified_source')
        self.assertIsNone(obj.model_revision)

    def test_constructor_is_lazy(self):
        obj = self.recognizer()
        self.assertIsNone(obj._model)
        self.assertIsNone(obj._processor)

    def test_load_is_offline_cpu_safe_and_uses_original_tokenizer(self):
        calls = []
        model = SimpleNamespace(to=lambda device: calls.append(('device', device)) or model,
                                eval=lambda: model)
        processor_class = SimpleNamespace(from_pretrained=lambda path, **opts: calls.append(('processor', opts)) or object())
        model_class = SimpleNamespace(from_pretrained=lambda path, **opts: calls.append(('model', opts)) or model)
        torch = SimpleNamespace(set_num_threads=lambda count: calls.append(('threads', count)))
        transformers = SimpleNamespace(TrOCRProcessor=processor_class, VisionEncoderDecoderModel=model_class)
        with patch.dict('sys.modules', {'torch': torch, 'transformers': transformers}):
            obj = self.recognizer()
            obj._load()
        options = dict(calls)
        self.assertEqual(options['device'], 'cpu')
        self.assertEqual(options['threads'], 2)
        for name in ('processor', 'model'):
            self.assertTrue(options[name]['local_files_only'])
            self.assertFalse(options[name]['trust_remote_code'])
            self.assertFalse(options[name]['token'])
        self.assertFalse(options['processor']['use_fast'])
        self.assertTrue(options['model']['use_safetensors'])

    def test_rejects_bad_budgets(self):
        for kwargs in ({'max_new_tokens': 0}, {'max_new_tokens': 129}, {'max_seconds': 0}, {'max_seconds': math.inf}, {'threads': 0}, {'threads': 9}, {'max_crop_pixels': 0}, {'preprocessing': 'unknown'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.recognizer(**kwargs)

    def test_diagnostic_cleanup_is_explicit_and_original_unchanged(self):
        original = Image.new('RGB', (100, 20), 'white')
        for x in range(100):
            original.putpixel((x, 10), (0, 0, 0))
        unchanged, raw = LocalHandwritingRecognizer._prepare_crop(original, 'none')
        cleaned, info = LocalHandwritingRecognizer._prepare_crop(original, 'contrast_rule_cleanup')
        self.assertEqual(raw['removed_long_rule_pixels'], 0)
        self.assertEqual(info['removed_long_rule_pixels'], 100)
        self.assertTrue(info['destructive_cleanup_not_validated'])
        self.assertEqual(cleaned.getpixel((50, 10)), (255, 255, 255))
        self.assertEqual(original.getpixel((50, 10)), (0, 0, 0))
        self.assertEqual(unchanged.getpixel((50, 10)), (0, 0, 0))

    def test_rejects_whole_page(self):
        with self.assertRaises(ValueError):
            self.recognizer().recognize(Image.new('RGB', (600, 800)))

    def test_pixel_budget_before_model_load(self):
        with self.assertRaises(ValueError):
            self.recognizer(max_crop_pixels=100).recognize(Image.new('RGB', (200, 20)))

    def test_rejects_nonimage(self):
        with self.assertRaises(TypeError):
            self.recognizer().recognize('image.png')

    def test_rejects_invalid_bbox(self):
        with self.assertRaises(ValueError):
            self.recognizer().recognize(Image.new('RGB', (200, 20)), bbox=[2, 1, 1, 2])

    def test_token_score_is_conservatively_capped(self):
        self.assertEqual(LocalHandwritingRecognizer._quality_score(1.0, 'Example', False), .72)
        self.assertEqual(LocalHandwritingRecognizer._quality_score(1.0, 'Example', True), .25)
        self.assertEqual(LocalHandwritingRecognizer._quality_score(1.0, '', False), 0.)
        self.assertEqual(LocalHandwritingRecognizer._quality_score(math.nan, 'Example', False), 0.)


if __name__ == '__main__':
    unittest.main()
