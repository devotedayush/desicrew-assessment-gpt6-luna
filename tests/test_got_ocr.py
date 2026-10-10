"""Privacy/budget/loader checks. These tests alone do not establish model accuracy."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import weakref
from contextlib import nullcontext
from unittest.mock import patch

from PIL import Image
from desicrew.got_ocr import LocalGotOcrRecognizer, MODEL_ID, MODEL_REVISION, WEIGHTS_SHA256, release_cpu_allocator
from scripts.evaluate_got_ocr import ROOT, ensure_private_dir, hash_file, normalize, edit_distance, PUBLIC_FILES, evaluate, summarize_partial_results, supervise, attach_process_resource_evidence


class GotOcrTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)
        (self.path / "model.safetensors").write_bytes(b"not-a-loaded-model")

    def tearDown(self):
        self.temporary.cleanup()

    def recognizer(self, **kwargs):
        return LocalGotOcrRecognizer(self.path, **kwargs)

    def test_local_safe_weights_required(self):
        with self.assertRaises(ValueError):
            LocalGotOcrRecognizer(MODEL_ID)
        (self.path / "model.safetensors").unlink()
        with self.assertRaises(ValueError):
            self.recognizer()

    def test_constructor_is_lazy(self):
        recognizer = self.recognizer()
        self.assertIsNone(recognizer._model)
        self.assertIsNone(recognizer._processor)

    def test_manifest_required(self):
        with self.assertRaisesRegex(RuntimeError, "manifest is required"):
            self.recognizer()._verify_files()

    def test_other_revision_rejected(self):
        (self.path / "DOWNLOAD_MANIFEST.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": "different"}))
        with self.assertRaisesRegex(RuntimeError, "pinned checkpoint"):
            self.recognizer()._verify_files()

    def test_fake_weights_hash_rejected(self):
        (self.path / "DOWNLOAD_MANIFEST.json").write_text(json.dumps({
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "sha256": {"model.safetensors": "0" * 64}}))
        with self.assertRaisesRegex(RuntimeError, "official pinned weights hash"):
            self.recognizer()._verify_files()

    def test_missing_verified_config_rejected(self):
        (self.path / "DOWNLOAD_MANIFEST.json").write_text(json.dumps({
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "sha256": {"model.safetensors": WEIGHTS_SHA256}}))
        with self.assertRaisesRegex(RuntimeError, "Missing verified local model file"):
            self.recognizer()._verify_files()

    def test_loader_is_native_cpu_offline_no_token_and_no_remote_code(self):
        calls = []
        model = SimpleNamespace(to=lambda device: calls.append(("device", device)) or model,
                                eval=lambda: model)
        processor_class = SimpleNamespace(from_pretrained=lambda path, **opts: calls.append(("processor", opts)) or object())
        model_class = SimpleNamespace(from_pretrained=lambda path, **opts: calls.append(("model", opts)) or model)
        torch = SimpleNamespace(set_num_threads=lambda count: calls.append(("threads", count)), bfloat16="bf16")
        transformers = SimpleNamespace(AutoProcessor=processor_class, GotOcr2ForConditionalGeneration=model_class)
        recognizer = self.recognizer()
        with patch.object(recognizer, "_verify_files"), patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            recognizer._load()
        options = dict(calls)
        self.assertEqual(options["device"], "cpu")
        self.assertEqual(options["threads"], 2)
        for name in ("processor", "model"):
            self.assertTrue(options[name]["local_files_only"])
            self.assertFalse(options[name]["trust_remote_code"])
            self.assertFalse(options[name]["token"])
        self.assertFalse(options["processor"]["use_fast"])
        self.assertTrue(options["model"]["use_safetensors"])
        self.assertEqual(options["model"]["torch_dtype"], "bf16")
        self.assertTrue(options["model"]["low_cpu_mem_usage"])

    def test_budget_validation(self):
        for kwargs in ({"max_new_tokens": 0}, {"max_new_tokens": 129}, {"max_seconds": 0},
                       {"max_seconds": math.inf}, {"threads": 0}, {"threads": 3}, {"max_crop_pixels": 1000001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.recognizer(**kwargs)

    def test_whole_page_and_nonimage_rejected_before_load(self):
        for image in (Image.new("RGB", (600, 800)), Image.new("RGB", (1500, 800))):
            with self.assertRaises(ValueError):
                self.recognizer().recognize(image)
        with self.assertRaises(TypeError):
            self.recognizer().recognize("https://example.com/image.png")

    def test_provenance_validation_before_load(self):
        image = Image.new("RGB", (200, 40))
        with self.assertRaises(ValueError):
            self.recognizer().recognize(image, page=0)
        with self.assertRaises(ValueError):
            self.recognizer().recognize(image, bbox=[10, 10, 5, 20])

    def test_public_artifact_paths_rejected(self):
        for path in (ROOT / "docs", ROOT / "outputs/private_samples", ROOT / "data/private/models"):
            with self.assertRaises(ValueError):
                ensure_private_dir(path)
        with self.assertRaises(ValueError):
            ensure_private_dir(ROOT / "docs", model=True)
        self.assertEqual(ensure_private_dir(ROOT / "data/private/models/got-ocr2", model=True),
                         (ROOT / "data/private/models/got-ocr2").resolve())

    def test_comparison_never_repairs_identifiers(self):
        self.assertEqual(normalize("  Example\nTown  "), "EXAMPLE TOWN")
        self.assertNotEqual(normalize("ABO01"), normalize("AB001"))
        self.assertEqual(edit_distance("ABO01", "AB001"), 1)

    def test_git_blob_and_sha256_hashes(self):
        path = self.path / "config.json"
        path.write_bytes(b"{}\n")
        self.assertEqual(hash_file(path), hashlib.sha256(b"{}\n").hexdigest())
        self.assertEqual(hash_file(path, "git_sha1"), hashlib.sha1(b"blob 3\0{}\n").hexdigest())

    def test_download_allowlist_has_no_executable_or_pickle_files(self):
        self.assertEqual(len(PUBLIC_FILES), 7)
        for filename in PUBLIC_FILES:
            self.assertTrue(filename.endswith((".json", ".safetensors")))
        self.assertEqual(PUBLIC_FILES["model.safetensors"][2], WEIGHTS_SHA256)

    def test_reference_is_only_scored_after_inference_and_never_repairs_output(self):
        private_root = ROOT / "outputs/private_samples"
        private_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=private_root) as temporary:
            folder = Path(temporary)
            crop = folder / "crop.png"
            Image.new("RGB", (100, 30), "white").save(crop)
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"crops": [{"id": "synthetic", "source": "synthetic.png",
                "crop_path": str(crop), "field": "identifier", "visually_verified": True,
                "reference_text": "ABO01"}]}))
            received = []
            class FakeRecognizer:
                weights_hash_verified = True
                def __init__(self, *args, **kwargs):
                    pass
                def recognize(self, image, **kwargs):
                    received.append(kwargs)
                    return {"text": "AB001", "elapsed_seconds": 0.01,
                            "terminated_early": False, "needs_review": True}
            with patch("scripts.evaluate_got_ocr.LocalGotOcrRecognizer", FakeRecognizer), \
                 patch("scripts.evaluate_got_ocr.importlib.metadata.version", return_value="test"):
                summary = evaluate(manifest, self.path, folder / "results", limit=1, max_seconds=20, threads=2)
            self.assertEqual(received, [{"field": "identifier", "page": 1, "bbox": None}])
            self.assertEqual(summary["exact_line_matches_case_whitespace_normalized"], 0)
            report = json.loads((folder / "results/got_line_evaluation_private.json").read_text())
            self.assertEqual(report["results"][0]["recognition"]["text"], "AB001")

    def test_failure_report_never_claims_full_completion_or_accuracy(self):
        status = {"status": "memory_budget_exceeded", "return_code": -9}
        summary = summarize_partial_results(self.path, status, 18)
        self.assertFalse(summary["completed_full_diagnostic"])
        self.assertFalse(summary["accuracy_claim"])
        self.assertFalse(summary["model_actually_ran"])
        self.assertEqual(summary["completed_crops"], 0)
        self.assertEqual(summary["unprocessed_crops"], 18)
        self.assertEqual(summary["status"], "memory_budget_exceeded")

    def test_reused_trial_outputs_rejected_before_subprocess_launch(self):
        (self.path / "partial_predictions_private.json").write_text("[]")
        with patch("scripts.evaluate_got_ocr.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "fresh private --output-dir"):
                supervise(["never-executed"], self.path, 600, 4096)
        popen.assert_not_called()

    def test_generation_returns_no_metadata_and_releases_tensors_before_cleanup(self):
        tensors = []
        options = {}
        cleanup_observations = []
        class Pixel:
            def __init__(self):
                tensors.append(weakref.ref(self))
            def to(self, **kwargs):
                return self
        class Output:
            def __init__(self):
                tensors.append(weakref.ref(self))
            def numel(self):
                return 2
            def __getitem__(self, key):
                return SimpleNamespace(item=lambda: 151645)
        class Generated:
            def __init__(self):
                tensors.append(weakref.ref(self))
            def __getitem__(self, key):
                return Output()
        class Processor:
            tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 151645)
            def __call__(self, **kwargs):
                return {"pixel_values": Pixel(), "input_ids": SimpleNamespace(shape=(1, 10))}
            def decode(self, output_ids, **kwargs):
                return "AB001"
        def generate(**kwargs):
            for key in ("return_dict_in_generate", "output_scores", "output_logits", "output_attentions", "output_hidden_states", "use_cache"):
                options[key] = kwargs[key]
            return Generated()
        def cleanup():
            cleanup_observations.append(all(reference() is None for reference in tensors))
            return {"rss_after_cleanup_mb": 1024}
        recognizer = self.recognizer()
        recognizer._processor = Processor()
        recognizer._model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=151643))
        recognizer._torch = SimpleNamespace(inference_mode=nullcontext, bfloat16="bf16")
        with patch("desicrew.got_ocr.release_cpu_allocator", cleanup), \
             patch("desicrew.got_ocr._current_rss_mb", return_value=1024):
            result = recognizer.recognize(Image.new("RGB", (200, 30)))
        for key in ("return_dict_in_generate", "output_scores", "output_logits", "output_attentions", "output_hidden_states"):
            self.assertFalse(options[key])
        self.assertTrue(options["use_cache"])
        self.assertEqual(cleanup_observations, [True])
        self.assertEqual(result["text"], "AB001")
        self.assertFalse(result["terminated_early"])
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["confidence_kind"], "heuristic_quality_score_not_probability")

    def test_cleanup_also_runs_after_generation_failure(self):
        recognizer = self.recognizer()
        class Processor:
            tokenizer = object()
            def __call__(self, **kwargs):
                return {"pixel_values": SimpleNamespace(to=lambda **kwargs: object()),
                        "input_ids": SimpleNamespace(shape=(1, 10))}
        recognizer._processor = Processor()
        recognizer._model = SimpleNamespace(generate=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
        recognizer._torch = SimpleNamespace(inference_mode=nullcontext, bfloat16="bf16")
        with patch("desicrew.got_ocr.release_cpu_allocator", return_value={}) as cleanup, \
             patch("desicrew.got_ocr._current_rss_mb", return_value=1024):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                recognizer.recognize(Image.new("RGB", (200, 30)))
        cleanup.assert_called_once()

    def test_allocator_cleanup_uses_only_free_page_trim_and_handles_unavailable_libc(self):
        with patch("desicrew.got_ocr.gc.collect", return_value=3), \
             patch("desicrew.got_ocr._current_rss_mb", side_effect=[1024, 900]), \
             patch("desicrew.got_ocr.ctypes.CDLL", side_effect=OSError("unsupported")), \
             patch("desicrew.got_ocr.sys.platform", "linux"):
            result = release_cpu_allocator()
        self.assertEqual(result["garbage_objects_collected"], 3)
        self.assertFalse(result["malloc_trim_available"])
        self.assertEqual(result["rss_after_cleanup_mb"], 900)

    def test_terminal_resource_evidence_does_not_change_predictions(self):
        summary = {"status": "completed", "crops_evaluated": 18}
        rows = [{"recognition": {"text": "AB001"}}]
        (self.path / "summary_without_field_values.json").write_text(json.dumps(summary))
        (self.path / "got_line_evaluation_private.json").write_text(json.dumps({"summary": summary, "results": rows}))
        status = {"status": "completed", "memory_budget_mb": 4096, "wall_budget_seconds": 600}
        attach_process_resource_evidence(self.path, status)
        aggregate = json.loads((self.path / "summary_without_field_values.json").read_text())
        evaluation = json.loads((self.path / "got_line_evaluation_private.json").read_text())
        self.assertEqual(aggregate["process_resource_report"], status)
        self.assertEqual(evaluation["summary"], aggregate)
        self.assertEqual(evaluation["results"], rows)
        self.assertNotIn("results", aggregate)
        with self.assertRaises(ValueError):
            attach_process_resource_evidence(self.path, {"status": "memory_budget_exceeded"})


if __name__ == "__main__":
    unittest.main()
