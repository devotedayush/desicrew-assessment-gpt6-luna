"""Deterministic text/parser tests plus local synthetic image/PDF smoke tests.

Passing these tests is not a claim about supplied-sample or handwriting accuracy.
"""
import shutil
import json
import tempfile
import unittest
from pathlib import Path

from desicrew.documents import DocumentProcessor, DOCUMENT_SCHEMAS, _Line, review_report
from scripts.demo_documents import create_fixtures, fixture_texts, synthetic_mrz


class DocumentParserTests(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor()

    def test_exact_schema_has_ten_types_and_44_targets(self):
        self.assertEqual(len(DOCUMENT_SCHEMAS), 10)
        self.assertEqual(sum(map(len, DOCUMENT_SCHEMAS.values())), 44)

    def test_all_synthetic_text_types_classify_and_emit_every_target(self):
        for doc_type, text in fixture_texts().items():
            with self.subTest(doc_type=doc_type):
                result = self.processor.process_text(text)
                self.assertEqual(result["document_type"], doc_type)
                self.assertEqual(set(result["fields"]), set(DOCUMENT_SCHEMAS[doc_type]))
                self.assertTrue(all(item["value"] is not None for item in result["fields"].values()))
                self.assertEqual(result["confidence_kind"], "heuristic_quality_score_not_probability")

    def test_bank_leading_zeroes_remain_string(self):
        result = self.processor.process_text(fixture_texts()["nach_ecs_mandate"])
        field = result["fields"]["bank_account_number"]
        self.assertEqual(field["value"], "000123456789")
        self.assertIsInstance(field["value"], str)
        self.assertTrue(field["needs_review"])

    def test_bank_name_does_not_end_at_second_bank_token(self):
        result = self.processor.process_text("NACH ECS Mandate\nBank Name: Example Bank")
        self.assertEqual(result["fields"]["bank_name"]["value"], "Example Bank")

    def test_application_leading_zeroes(self):
        result = self.processor.process_text(fixture_texts()["suitability_profiler_declaration"])
        self.assertEqual(result["fields"]["application_number"]["value"], "0000054321")

    def test_mrz_is_preserved_exactly_including_fillers(self):
        result = self.processor.process_text(fixture_texts()["passport"])
        mrz = result["fields"]["mrz_line_2"]
        self.assertEqual(mrz["value"], synthetic_mrz())
        self.assertEqual(len(mrz["value"]), 44)
        self.assertEqual(mrz["validation_issues"], [])

    def test_mrz_failed_check_digit_flags(self):
        mrz = synthetic_mrz()
        wrong = mrz[:9] + ("0" if mrz[9] != "0" else "1") + mrz[10:]
        result = self.processor.process_text("Passport\nMRZ Line 2: " + wrong)
        field = result["fields"]["mrz_line_2"]
        self.assertIn("mrz_check_digit_failed", field["validation_issues"])
        self.assertTrue(field["needs_review"])

    def test_mrz_invalid_length_not_repaired(self):
        result = self.processor.process_text("Passport\nMRZ Line 2: " + synthetic_mrz()[:-1])
        self.assertIsNone(result["fields"]["mrz_line_2"]["value"])

    def test_checked_mrz_extracts_passport_number(self):
        result = self.processor.process_text("Passport\n" + synthetic_mrz())
        self.assertEqual(result["fields"]["passport_number"]["value"], "U1234567")
        self.assertEqual(result["fields"]["passport_number"]["source"]["method"], "checked_mrz_passport_number")

    def test_illegible_ifsc_returns_null_without_character_repair(self):
        result = self.processor.process_text("NACH ECS Mandate\nIFSC Code: DEMOO123456")
        self.assertIsNone(result["fields"]["ifsc_code"]["value"])
        self.assertIn("invalid_ifsc_shape_no_character_repair", result["fields"]["ifsc_code"]["validation_issues"])

    def test_account_o_is_not_silently_changed_to_zero(self):
        result = self.processor.process_text("NACH ECS Mandate\nBank Account Number: O00123456789")
        self.assertIsNone(result["fields"]["bank_account_number"]["value"])

    def test_invalid_date_null(self):
        result = self.processor.process_text("PAN Card\nDate of Birth: 31/02/1994")
        self.assertIsNone(result["fields"]["date_of_birth"]["value"])
        self.assertTrue(result["needs_review"])

    def test_ambiguous_numeric_date_is_flagged(self):
        result = self.processor.process_text("PAN Card\nDate of Birth: 03/04/1994")
        field = result["fields"]["date_of_birth"]
        self.assertEqual(field["value"], "1994-04-03")
        self.assertIn("numeric_date_assumed_day_first", field["validation_issues"])
        self.assertLess(field["confidence"], .90)

    def test_two_digit_year_does_not_invent_century(self):
        result = self.processor.process_text("PAN Card\nDate of Birth: 19/03/64")
        field = result["fields"]["date_of_birth"]
        self.assertIsNone(field["value"])
        self.assertIn("two_digit_year_century_ambiguous", field["validation_issues"])

    def test_malformed_amount_grouping_is_not_silently_normalized(self):
        result = self.processor.process_text("NACH ECS Mandate\nAmount (figures): 25,00")
        self.assertIsNone(result["fields"]["amount_figures"]["value"])

    def test_empty_fields_are_explicit(self):
        result = self.processor.process_text("Moral Hazard Questionnaire")
        self.assertEqual(len(result["fields"]), 5)
        self.assertTrue(all(item["value"] is None and item["status"] == "missing" for item in result["fields"].values()))

    def test_unreadable_marker_is_null(self):
        result = self.processor.process_text("FATCA Annexure Form\nPlace of Birth: [unreadable]")
        self.assertIsNone(result["fields"]["place_of_birth"]["value"])

    def test_selected_checkbox_extracts_but_requires_review(self):
        result = self.processor.process_text(fixture_texts()["multiple_policies_consent_form"])
        field = result["fields"]["reason_for_multiple_policies"]
        self.assertEqual(field["value"], "Family protection")
        self.assertIn("checkbox_selection_requires_visual_review", field["validation_issues"])

    def test_unselected_checkbox_options_are_not_a_selection(self):
        result = self.processor.process_text("Multiple Policies Consent Form\nReason for Multiple Policies: [ ] Investment [ ] Family protection")
        field = result["fields"]["reason_for_multiple_policies"]
        self.assertIsNone(field["value"])
        self.assertIn("checkbox_selection_not_explicit", field["validation_issues"])

    def test_two_checked_options_are_ambiguous(self):
        result = self.processor.process_text("Multiple Policies Consent Form\nReason for Multiple Policies: [x] Investment [x] Family protection")
        field = result["fields"]["reason_for_multiple_policies"]
        self.assertIsNone(field["value"])
        self.assertIn("multiple_selected_checkboxes", field["validation_issues"])

    def test_handwritten_fields_conservative_even_with_high_text_confidence(self):
        result = self.processor.process_text(fixture_texts()["fatca_annexure_form"])
        self.assertTrue(all(item["needs_review"] for item in result["fields"].values()))
        self.assertLessEqual(result["fields"]["place_of_birth"]["confidence"], .78)

    def test_handwritten_review_still_forced_with_lower_threshold(self):
        result = DocumentProcessor(.50).process_text(fixture_texts()["fatca_annexure_form"])
        self.assertTrue(result["needs_review"])
        self.assertTrue(all(item["needs_review"] for item in result["fields"].values()))

    def test_conflicting_account_numbers_are_not_arbitrarily_chosen(self):
        result = self.processor.process_text("NACH ECS Mandate\nAccount Number: 000123456789\nAccount Number: 000123456780")
        field = result["fields"]["bank_account_number"]
        self.assertIsNone(field["value"])
        self.assertIn("conflicting_candidates", field["validation_issues"])

    def test_ocr_pass_disagreement_is_explicit_null(self):
        primary = [("000123456789", _Line("Account Number: 000123456789", confidence=.98), "label_same_line")]
        alternate = [("000123456780", _Line("Account Number: 000123456780", confidence=.99), "label_same_line")]
        field = self.processor._field_result("bank_account_number", primary, True, True, alternate)
        self.assertIsNone(field["value"])
        self.assertIn("ocr_segmentation_passes_disagree", field["validation_issues"])

    def test_name_is_not_father_name(self):
        result = self.processor.process_text("PAN Card\nName: Ananya Demo\nFather's Name: Arun Demo")
        self.assertEqual(result["fields"]["full_name"]["value"], "Ananya Demo")
        self.assertEqual(result["fields"]["father_name"]["value"], "Arun Demo")

    def test_line_after_label_extraction(self):
        result = self.processor.process_text("PAN Card\nFull Name:\nAnanya Demo\nPAN Number:\nDEMPP4321Q")
        self.assertEqual(result["fields"]["full_name"]["value"], "Ananya Demo")
        self.assertEqual(result["fields"]["pan_number"]["value"], "DEMPP4321Q")
        self.assertTrue(result["fields"]["pan_number"]["needs_review"])

    def test_multiline_address(self):
        result = self.processor.process_text(fixture_texts()["aadhaar_card"])
        self.assertEqual(result["fields"]["address"]["value"], "42 Example Lane, Demonstration Nagar, Pune 411001")

    def test_multiline_address_uses_lowest_confidence_and_all_line_evidence(self):
        lines = [_Line("Aadhaar Card", index=1), _Line("Address: 42 Demo Street", confidence=.99, bbox=[10, 20, 300, 50], index=2), _Line("WR0NG LOCAL1TY 411001", confidence=.10, bbox=[10, 60, 330, 90], index=3)]
        result = self.processor._process(lines, None, "synthetic", False, {"is_image_ocr": True, "warnings": []})
        field = result["fields"]["address"]
        self.assertTrue(field["needs_review"])
        self.assertEqual(field["confidence"], .10)
        self.assertEqual(field["source"]["ocr_confidence"], .10)
        self.assertEqual([part["line"] for part in field["source"]["lines"]], [2, 3])
        self.assertEqual([part["bbox"] for part in field["source"]["lines"]], [[10, 20, 300, 50], [10, 60, 330, 90]])
        self.assertEqual(field["source"]["bbox"], [10, 20, 330, 90])
        self.assertIn("WR0NG LOCAL1TY", field["source"]["text"])

    def test_label_next_line_name_keeps_label_and_value_evidence(self):
        lines = [_Line("PAN Card", index=1), _Line("Full Name:", confidence=.10, index=2), _Line("Ananya Demo", confidence=.99, index=3)]
        result = self.processor._process(lines, None, "synthetic", False, {"is_image_ocr": True, "warnings": []})
        field = result["fields"]["full_name"]
        self.assertEqual(field["value"], "Ananya Demo")
        self.assertEqual(field["confidence"], .10)
        self.assertEqual(len(field["source"]["lines"]), 2)
        self.assertTrue(field["needs_review"])

    def test_next_line_address_preserves_every_continuation_without_duplicates(self):
        lines = [_Line("Aadhaar Card", index=1), _Line("Address:", confidence=.99, index=2), _Line("42 Demo Street", confidence=.98, index=3), _Line("Wrong Locality", confidence=.20, index=4)]
        result = self.processor._process(lines, None, "synthetic", False, {"is_image_ocr": True, "warnings": []})
        field = result["fields"]["address"]
        self.assertEqual(field["value"], "42 Demo Street, Wrong Locality")
        self.assertEqual(field["confidence"], .20)
        self.assertEqual([part["line"] for part in field["source"]["lines"]], [2, 3, 4])

    def test_multiline_checkbox_keeps_low_confidence_option_evidence(self):
        lines = [_Line("Multiple Policies Consent Form", index=1), _Line("Reason for Multiple Policies: [ ] Investment", confidence=.99, index=2), _Line("[x] Family protection", confidence=.10, index=3)]
        result = self.processor._process(lines, None, "synthetic", False, {"is_image_ocr": True, "warnings": []})
        field = result["fields"]["reason_for_multiple_policies"]
        self.assertEqual(field["value"], "Family protection")
        self.assertEqual(field["confidence"], .10)
        self.assertEqual(len(field["source"]["lines"]), 2)
        self.assertTrue(field["needs_review"])

    def test_multiline_checkbox_cannot_ignore_second_selection(self):
        result = self.processor.process_text("Multiple Policies Consent Form\nReason for Multiple Policies: [x] Investment\n[x] Family protection")
        field = result["fields"]["reason_for_multiple_policies"]
        self.assertIsNone(field["value"])
        self.assertIn("multiple_selected_checkboxes", field["validation_issues"])
        self.assertEqual(len(field["source"]["lines"]), 2)

    def test_mrz_all_mandatory_check_digits_must_be_numeric(self):
        for position in (9, 19, 27, 43):
            with self.subTest(position=position):
                mrz = synthetic_mrz()
                wrong = mrz[:position] + "<" + mrz[position + 1:]
                result = self.processor.process_text("Passport\nMRZ Line 2: " + wrong)
                field = result["fields"]["mrz_line_2"]
                self.assertIsNone(field["value"])
                self.assertIn("mrz_mandatory_check_digit_missing", field["validation_issues"])
                self.assertIsNone(result["fields"]["passport_number"]["value"])
                self.assertIsNone(result["fields"]["date_of_birth"]["value"])
                self.assertIsNone(result["fields"]["date_of_expiry"]["value"])

    def test_mrz_failed_composite_cannot_supply_passport_number(self):
        mrz = synthetic_mrz()
        wrong = mrz[:-1] + str((int(mrz[-1]) + 1) % 10)
        result = self.processor.process_text("Passport\nMRZ Line 2: " + wrong)
        self.assertIn("mrz_composite_check_digit_failed", result["fields"]["mrz_line_2"]["validation_issues"])
        self.assertTrue(result["fields"]["mrz_line_2"]["needs_review"])
        self.assertIsNone(result["fields"]["passport_number"]["value"])

    def test_mrz_unused_optional_check_digit_can_be_filler(self):
        mrz = synthetic_mrz()
        optional_filler = mrz[:42] + "<" + mrz[43]
        result = self.processor.process_text("Passport\nMRZ Line 2: " + optional_filler)
        self.assertEqual(result["fields"]["mrz_line_2"]["value"], optional_filler)
        self.assertEqual(result["fields"]["mrz_line_2"]["validation_issues"], [])

    def test_labelled_valid_mrz_uses_checked_not_shape_only_passport_path(self):
        result = self.processor.process_text("Passport\nMRZ Line 2: " + synthetic_mrz())
        self.assertEqual(result["fields"]["passport_number"]["source"]["method"], "checked_mrz_passport_number")

    def test_bilingual_card_label_prefix_and_unicode_apostrophe(self):
        result = self.processor.process_text("PAN Card\nनाम / Name\nAnanya Demo\nपिता / Father’s Name\nArun Demo\nजन्म / Date of Birth\n19/03/1994")
        self.assertEqual(result["fields"]["full_name"]["value"], "Ananya Demo")
        self.assertEqual(result["fields"]["father_name"]["value"], "Arun Demo")
        self.assertEqual(result["fields"]["date_of_birth"]["value"], "1994-03-19")

    def test_bilingual_card_label_suffix_uses_value_line(self):
        result = self.processor.process_text("Aadhaar Card\nName / नाम\nAnanya Demo\nDate of Birth / जन्म\n19/03/1994")
        self.assertEqual(result["fields"]["full_name"]["value"], "Ananya Demo")
        self.assertEqual(result["fields"]["date_of_birth"]["value"], "1994-03-19")

    def test_sparse_geometry_uses_aligned_date_not_unrelated_column(self):
        lines = [_Line("Driving Licence", index=1), _Line("Date of Issue", bbox=[100, 100, 250, 125], index=2), _Line("19/03/2032", bbox=[750, 100, 900, 125], index=3), _Line("19/03/2012", bbox=[100, 140, 260, 165], index=4)]
        result = self.processor._process(lines, None, "synthetic", False, {"is_image_ocr": True, "warnings": []})
        self.assertEqual(result["fields"]["date_of_issue"]["value"], "2012-03-19")
        self.assertEqual([part["line"] for part in result["fields"]["date_of_issue"]["source"]["lines"]], [2, 4])

    def test_application_proposal_alias_and_enumerated_field(self):
        result = self.processor.process_text("FATCA Annexure Form\na) Policy No.: 0000012345")
        self.assertEqual(result["fields"]["policy_number"]["value"], "0000012345")
        result = self.processor.process_text("Benefit Illustration Declaration\nApplication/Proposal Form Number: 0000012345")
        self.assertEqual(result["fields"]["application_number"]["value"], "0000012345")

    def test_out_of_scope_header_overrides_incidental_ecs_mention(self):
        result = self.processor.process_text("Customer Declaration - Application/Proposal Form\nThis product supports NACH ECS mandate collection")
        self.assertEqual(result["document_type"], "unknown")
        self.assertEqual(result["classification_evidence"][0]["document_type"], "out_of_scope")
        self.assertTrue(result["needs_review"])

    def test_literal_dummy_aadhaar_digits_preserved_and_validity_flagged(self):
        result = self.processor.process_text("Aadhaar Card\nAadhaar Number: 0123 4567 8901")
        field = result["fields"]["aadhaar_number"]
        self.assertEqual(field["value"], "012345678901")
        self.assertIn("aadhaar_leading_digit_not_valid_for_issued_number", field["validation_issues"])
        self.assertTrue(field["needs_review"])

    def test_case_only_name_variants_do_not_create_conflict(self):
        result = self.processor.process_text("PAN Card\nName: Ananya Demo\nName: ANANYA DEMO")
        self.assertIsNotNone(result["fields"]["full_name"]["value"])
        self.assertNotIn("conflicting_candidates", result["fields"]["full_name"]["validation_issues"])

    def test_optional_fake_local_backend_requires_review_and_preserves_crop_source(self):
        from PIL import Image
        class FakeRecognizer:
            def recognize(self, image, **kwargs):
                return {"text": "Ananya Demo", "confidence": .99, "model": "synthetic-fake", "validation_issues": [], "provenance": kwargs}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "form.png"
            Image.new("RGB", (100, 50), "white").save(path)
            manifest = Path(temp) / "roi.json"
            manifest.write_text(json.dumps({"coordinate_space": "original_image_pixels", "documents": [{"file": str(path), "document_type": "moral_hazard_questionnaire", "regions": [{"field": "name_of_life_assured", "page": 1, "bbox": [0, 0, 80, 30]}]}]}))
            processor = DocumentProcessor(.50, handwriting_backend=FakeRecognizer(), roi_manifest_path=manifest)
            result = processor._augment_handwriting(path, processor.process_text("Moral Hazard Questionnaire"))
            field = result["fields"]["name_of_life_assured"]
            self.assertEqual(field["value"], "Ananya Demo")
            self.assertTrue(field["needs_review"])
            self.assertLessEqual(field["confidence"], .72)
            self.assertEqual(field["handwriting_candidate"]["bbox"], [0, 0, 80, 30])
            self.assertIn("handwritten_model_not_calibrated", field["validation_issues"])

    def test_optional_backend_cannot_erase_existing_ocr_conflict(self):
        from PIL import Image
        class FakeRecognizer:
            def recognize(self, image, **kwargs):
                return {"text": "000123456789", "confidence": .99, "model": "synthetic-fake", "validation_issues": []}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "form.png"
            Image.new("RGB", (100, 50), "white").save(path)
            manifest = Path(temp) / "roi.json"
            manifest.write_text(json.dumps({"coordinate_space": "original_image_pixels", "documents": [{"file": str(path), "document_type": "nach_ecs_mandate", "regions": [{"field": "bank_account_number", "page": 1, "bbox": [0, 0, 80, 30]}]}]}))
            processor = DocumentProcessor(handwriting_backend=FakeRecognizer(), roi_manifest_path=manifest)
            result = processor._augment_handwriting(path, processor.process_text("NACH ECS Mandate\nAccount Number: 000123456789\nAccount Number: 000123456780"))
            field = result["fields"]["bank_account_number"]
            self.assertIsNone(field["value"])
            self.assertIn("conflicting_candidates", field["validation_issues"])

    def test_geometry_only_templates_cover_all_ten_types_and_42_text_regions(self):
        from desicrew.regions import SAMPLE_REGION_TEMPLATES
        self.assertEqual(set(SAMPLE_REGION_TEMPLATES), set(DOCUMENT_SCHEMAS))
        self.assertEqual(sum(len(template["regions"]) for template in SAMPLE_REGION_TEMPLATES.values()), 42)
        for doc_type, template in SAMPLE_REGION_TEMPLATES.items():
            self.assertTrue(template["requires_alignment_review"])
            for region in template["regions"]:
                self.assertIn(region["field"], DOCUMENT_SCHEMAS[doc_type])
                self.assertTrue(all(0 <= coordinate <= 1 for coordinate in region["bbox_normalized"]))
        serialized = json.dumps(SAMPLE_REGION_TEMPLATES)
        self.assertNotIn("/workspace/", serialized)
        self.assertNotIn("crop_path", serialized)
        self.assertNotIn("reference_text", serialized)

    def test_roi_basename_fallback_requires_type_and_dimensions(self):
        processor = DocumentProcessor()
        processor.roi_documents = [{"file": "/original/source/form.png", "document_type": "pan_card", "image_size": [100, 50], "regions": [{"field": "pan_number", "bbox": [0, 0, 100, 50]}]}]
        regions = processor._regions_for_file(Path("/different/temp/form.png"), "pan_card", (100, 50))
        self.assertEqual(len(regions), 1)
        self.assertEqual(processor._regions_for_file(Path("/different/temp/form.png"), "pan_card", (101, 50)), [])
        self.assertEqual(processor._regions_for_file(Path("/different/temp/form.png"), "passport", (100, 50)), [])
        self.assertEqual(processor._regions_for_file(Path("/different/temp/another.png"), "pan_card", (100, 50)), [])

    def test_normalized_roi_template_scaling_and_aspect_guard(self):
        processor = DocumentProcessor(region_templates={"pan_card": {"aspect_ratio": 2., "regions": [{"field": "pan_number", "bbox_normalized": [.1, .2, .8, .6]}]}})
        regions = processor._regions_for_file(Path("upload.png"), "pan_card", (200, 100))
        self.assertEqual(regions[0]["bbox"], [20, 20, 160, 60])
        self.assertIn("requires_alignment_review", regions[0]["region_source"])
        self.assertEqual(processor._regions_for_file(Path("upload.png"), "pan_card", (200, 200)), [])

    def test_ambiguous_basename_roi_entries_are_not_arbitrarily_chosen(self):
        processor = DocumentProcessor()
        entry = {"file": "/original/source/form.png", "document_type": "pan_card", "image_size": [100, 50], "regions": [{"field": "pan_number", "bbox": [0, 0, 100, 50]}]}
        processor.roi_documents = [entry, {**entry, "file": "/another/source/form.png"}]
        self.assertEqual(processor._regions_for_file(Path("/temp/form.png"), "pan_card", (100, 50)), [])

    def test_printed_region_contract_preserves_candidates_and_forces_review(self):
        from PIL import Image
        from unittest.mock import patch
        import subprocess
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t1\t2\t10\t5\t99\tDEMPP4321Q\n"
        templates = {"pan_card": {"aspect_ratio": 2., "regions": [{"field": "pan_number", "bbox_normalized": [0., 0., 1., 1.]}]}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pan.png"
            Image.new("RGB", (100, 50), "white").save(path)
            processor = DocumentProcessor(.50, region_templates=templates, use_printed_region_ocr=True)
            with patch("desicrew.documents.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=tsv)):
                result = processor._augment_printed_regions(path, processor.process_text("PAN Card"))
            field = result["fields"]["pan_number"]
            self.assertEqual(field["value"], "DEMPP4321Q")
            self.assertTrue(field["needs_review"])
            self.assertLessEqual(field["confidence"], .85)
            self.assertEqual(len(field["region_candidates"]), 2)
            self.assertIn("explicit_region_alignment_requires_review", field["validation_issues"])

    def test_printed_name_region_consensus_resolves_other_segmentation_pass(self):
        from PIL import Image
        from unittest.mock import patch
        import subprocess
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t1\t2\t30\t10\t95\tExample\n"
        templates = {"pan_card": {"aspect_ratio": 2., "regions": [{"field": "full_name", "bbox_normalized": [0., 0., 1., 1.]}]}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pan.png"
            Image.new("RGB", (100, 50), "white").save(path)
            processor = DocumentProcessor(region_templates=templates, use_printed_region_ocr=True)
            result = processor.process_text("PAN Card\nFull Name: Example\nPAN Number: DEMPP4321Q")
            result["fields"]["full_name"].update(value=None, status="unreadable", validation_issues=["ocr_segmentation_passes_disagree"])
            with patch("desicrew.documents.subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=tsv)):
                result = processor._augment_printed_regions(path, result)
            field = result["fields"]["full_name"]
            self.assertEqual(field["value"], "Example")
            self.assertTrue(field["needs_review"])
            self.assertIn("localized_printed_consensus_resolves_segmentation_disagreement", field["validation_issues"])

    def test_very_weak_handwritten_image_ocr_retains_evidence_but_not_value(self):
        processor = DocumentProcessor()
        weak = processor._field_result("place", [("vr", _Line("Place: vr", confidence=.205), "label_same_line")], True, True)
        self.assertIsNone(weak["value"])
        self.assertEqual(weak["source"]["candidate_text"], "vr")
        self.assertIn("handwritten_ocr_evidence_too_weak_for_candidate_value", weak["validation_issues"])
        strong = processor._field_result("place", [("Pune", _Line("Place: Pune", confidence=.72), "label_same_line")], True, True)
        self.assertEqual(strong["value"], "Pune")
        self.assertTrue(strong["needs_review"])

    def test_name_and_place_remove_only_terminal_punctuation(self):
        processor = DocumentProcessor()
        self.assertEqual(processor._validate("policyholder_name", "Ms. Example.")[0], "Ms. Example")
        self.assertEqual(processor._validate("place", "Sample District,")[0], "Sample District")
        self.assertEqual(processor._validate("address", "Sample District,")[0], "Sample District,")

    def test_guarded_roi_model_use_from_temporary_basename(self):
        from PIL import Image
        class FakeRecognizer:
            def recognize(self, image, **kwargs):
                return {"text": "Ananya Demo", "confidence": .99, "model": "synthetic-fake", "validation_issues": []}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "form.png"
            Image.new("RGB", (100, 50), "white").save(path)
            processor = DocumentProcessor(handwriting_backend=FakeRecognizer())
            processor.roi_documents = [{"file": "/original/source/form.png", "document_type": "moral_hazard_questionnaire", "image_size": [100, 50], "regions": [{"field": "name_of_life_assured", "bbox": [0, 0, 100, 30]}]}]
            result = processor._augment_handwriting(path, processor.process_text("Moral Hazard Questionnaire"))
            self.assertEqual(result["fields"]["name_of_life_assured"]["value"], "Ananya Demo")
            self.assertTrue(result["fields"]["name_of_life_assured"]["needs_review"])

    def test_insurance_identifier_cannot_be_a_model_generated_plain_word(self):
        result = self.processor.process_text("FATCA Annexure Form\nPolicy Number: illustrative")
        self.assertIsNone(result["fields"]["policy_number"]["value"])
        self.assertIn("identifier_has_no_digits", result["fields"]["policy_number"]["validation_issues"])
        result = self.processor.process_text("Benefit Illustration Declaration\nApplication Number: DEMO00001")
        self.assertEqual(result["fields"]["application_number"]["value"], "DEMO00001")

    def test_diagnostic_comparison_does_not_fill_or_mutate_outputs(self):
        from scripts.evaluate_documents import compare
        reference = {"documents": [{"file": "synthetic.png", "document_type": "pan_card", "fields": {"full_name": {"value": "Ananya Demo", "readable": True}, "date_of_birth": {"value": None, "readable": False}}}]}
        result = [{"source_file": "synthetic.png", "fields": {"full_name": {"value": None, "source": {"candidate_text": "ANANYA DEMO _"}, "validation_issues": ["conflicting_candidates"]}}}]
        before = json.dumps(result, sort_keys=True)
        report = compare(reference, result)
        self.assertEqual(report["counts"]["readable_references"], 1)
        self.assertEqual(report["counts"]["excluded_references"], 1)
        self.assertEqual(report["counts"]["normalized_final_exact_matches"], 0)
        self.assertEqual(report["counts"]["correct_but_rejected_raw_candidate_fields"], 1)
        self.assertEqual(json.dumps(result, sort_keys=True), before)

    def test_diagnostic_normalization_does_not_repair_characters_or_guess_century(self):
        from scripts.evaluate_documents import normalize
        self.assertNotEqual(normalize("bank_account_number", "O0012345"), normalize("bank_account_number", "00012345"))
        self.assertNotEqual(normalize("date", "19/03/26"), normalize("date", "2026-03-19"))
        self.assertEqual(normalize("date", "19/03/2026"), normalize("date", "2026-03-19"))
        self.assertEqual(normalize("amount_figures", "INR 25,000"), normalize("amount_figures", "25000"))

    def test_declaration_prose_is_not_a_name(self):
        result = self.processor.process_text("Suitability Profiler Declaration\nI confirm the name of life assured is entered below")
        self.assertIsNone(result["fields"]["name_of_life_assured"]["value"])

    def test_blank_unknown(self):
        result = self.processor.process_text("")
        self.assertEqual(result["document_type"], "unknown")
        self.assertTrue(result["needs_review"])

    def test_type_tie_is_ambiguous(self):
        result = self.processor.process_text("PAN Card\nPassport")
        self.assertEqual(result["document_type"], "unknown")
        self.assertTrue(result["needs_review"])

    def test_type_hint_does_not_fabricate_confidence(self):
        result = self.processor.process_text("", "PAN Card")
        self.assertEqual(result["document_type"], "pan_card")
        self.assertEqual(result["classification_confidence"], 0.)
        self.assertTrue(result["needs_review"])

    def test_wrong_type_hint_flags(self):
        result = self.processor.process_text("Passport", "PAN Card")
        self.assertIn("document_type_hint_conflicts_with_detected_content", result["review_reasons"])

    def test_review_report_has_field_level_entries_and_rationale(self):
        result = self.processor.process_text("NACH ECS Mandate")
        report = self.processor.review_report([result])
        self.assertEqual(report["review_threshold"], .90)
        self.assertEqual(report["fields_needing_review"], 5)
        self.assertEqual(len(report["fields"]), 5)
        self.assertIn("not a calibrated", report["threshold_rationale"])

    def test_selected_threshold_rationale_is_dynamic(self):
        result = DocumentProcessor(.85).process_text("PAN Card")
        report = review_report([result], .85)
        self.assertEqual(report["review_threshold"], .85)
        self.assertTrue(report["threshold_rationale"].startswith("0.85 "))
        self.assertIn("85%", report["threshold_rationale"])
        self.assertTrue(all(item["threshold"] == .85 for item in report["fields"]))

    def test_mixed_threshold_report_preserves_extraction_thresholds(self):
        results = [DocumentProcessor(.85).process_text("PAN Card", source_name="a"), DocumentProcessor(.95).process_text("Passport", source_name="b")]
        report = review_report(results, .90)
        self.assertEqual({item["threshold"] for item in report["fields"]}, {.85, .95})
        self.assertEqual(report["review_threshold"], .90)

    def test_low_confidence_labelled_field_is_flagged(self):
        result = self.processor.process_text(fixture_texts()["pan_card"], ocr_confidence=.60)
        self.assertTrue(all(item["needs_review"] for item in result["fields"].values()))

    def test_threshold_and_hint_validation(self):
        for threshold in (0, -1, 1.1):
            with self.assertRaises(ValueError):
                DocumentProcessor(threshold)
        with self.assertRaises(ValueError):
            self.processor.process_text("test", "unrecognized")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.process_file("/nonexistent/example.png")

    def test_corrupted_png_has_review_failure_instead_of_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken.png"
            path.write_bytes(b"not an image")
            result = self.processor.process_file(path, "pan_card")
            self.assertTrue(result["needs_review"])
            self.assertIn("file_read_or_ocr_failed", result["review_reasons"])
            self.assertEqual(len(result["fields"]), 4)

    def test_tsv_confidence_uses_minimum_not_label_average(self):
        tsv = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n5\t1\t1\t1\t1\t1\t1\t2\t3\t4\t99\tIFSC\n5\t1\t1\t1\t1\t2\t5\t2\t9\t4\t40\tDEMO0123456\n"
        lines, conf = self.processor._parse_tsv(tsv, 1)
        self.assertEqual(lines[0].confidence, .40)
        self.assertEqual(lines[0].bbox, [1, 2, 14, 6])


class CheckboxPixelTests(unittest.TestCase):
    @staticmethod
    def box(kind="empty"):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (40, 40), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((1, 1, 38, 38), outline="black", width=2)
        if kind == "tick":
            draw.line([(12, 20), (17, 26), (30, 13)], fill="black", width=2)
        elif kind == "filled":
            draw.rectangle((6, 6, 34, 34), fill="black")
        elif kind == "rule":
            draw.line((0, 20, 39, 20), fill="black", width=2)
        elif kind == "noise":
            for point in [(16, 16), (23, 22), (19, 24), (13, 20)]:
                draw.point(point, fill="black")
        elif kind == "ambiguous":
            draw.line((20, 14, 20, 26), fill="black", width=2)
        return image

    def test_empty_box_border_is_not_selection(self):
        from desicrew.checkboxes import inspect_box
        self.assertEqual(inspect_box(self.box())["status"], "unselected")

    def test_tick_and_filled_box_are_detected(self):
        from desicrew.checkboxes import inspect_box
        self.assertEqual(inspect_box(self.box("tick"))["status"], "selected")
        self.assertEqual(inspect_box(self.box("filled"))["status"], "selected")

    def test_horizontal_rule_and_small_speckles_are_not_selection(self):
        from desicrew.checkboxes import inspect_box
        self.assertEqual(inspect_box(self.box("rule"))["status"], "unselected")
        self.assertEqual(inspect_box(self.box("noise"))["status"], "unselected")

    def test_unclear_mark_and_tiny_crop_remain_ambiguous(self):
        from PIL import Image
        from desicrew.checkboxes import inspect_box
        self.assertEqual(inspect_box(self.box("ambiguous"))["status"], "ambiguous")
        self.assertEqual(inspect_box(Image.new("RGB", (7, 7), "white"))["status"], "ambiguous")

    def test_multiple_or_ambiguous_boxes_return_null(self):
        from desicrew.checkboxes import selected_option
        self.assertEqual(selected_option([{"label":"A", "status":"selected"}, {"label":"B", "status":"selected"}])[0], None)
        self.assertEqual(selected_option([{"label":"A", "status":"selected"}, {"label":"B", "status":"ambiguous"}])[0], None)
        self.assertEqual(selected_option([{"label":"A", "status":"unselected"}])[0], None)

    def test_pipeline_selects_from_pixels_and_always_requires_review(self):
        from PIL import Image
        image = Image.new("RGB", (100, 50), "white")
        image.paste(self.box("tick"), (0, 0))
        image.paste(self.box("empty"), (50, 0))
        template = {"multiple_policies_consent_form": {"aspect_ratio": 2., "checkbox_fields": {"reason_for_multiple_policies": [{"label":"Option A", "bbox_normalized":[0,0,.4,.8]}, {"label":"Option B", "bbox_normalized":[.5,0,.9,.8]}]}}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "boxes.png"
            image.save(path)
            processor = DocumentProcessor(.50, region_templates=template, use_checkbox_detection=True)
            result = processor._augment_checkboxes(path, processor.process_text("Multiple Policies Consent Form"))
            item = result["fields"]["reason_for_multiple_policies"]
            self.assertEqual(item["value"], "Option A")
            self.assertTrue(item["needs_review"])
            self.assertEqual(item["confidence"], .55)
            self.assertEqual(len(item["checkbox_candidates"]), 2)
            self.assertEqual(item["source"]["label_source"], "geometry_only_template_option_labels")


class DocumentLocalSmokeTests(unittest.TestCase):
    def test_native_pdf_local_extraction(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_pan.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((40, 40), "SYNTHETIC PRINTED FIXTURE\n" + fixture_texts()["pan_card"], fontsize=12)
            pdf.save(path)
            pdf.close()
            result = DocumentProcessor().process_file(path)
            self.assertEqual(result["document_type"], "pan_card")
            self.assertEqual(result["fields"]["pan_number"]["value"], "DEMPP4321Q")
            self.assertEqual(result["ocr"]["engine"], "native_pdf_text")

    def test_cumulative_pdf_limit_rejects_small_pages_without_large_allocation(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bounded.pdf"
            with fitz.open() as pdf:
                pdf.new_page(width=50, height=50)
                pdf.new_page(width=50, height=50)
                pdf.save(path)
            result = DocumentProcessor(max_total_pixels=70_000).process_file(path, "pan_card")
            self.assertIn("file_read_or_ocr_failed", result["review_reasons"])
            self.assertTrue(any("cumulative" in warning for warning in result["ocr"]["warnings"]))

    def test_cumulative_tiff_limit_rejects_small_frames_without_large_allocation(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bounded.tiff"
            frames = [Image.new("RGB", (30, 30), "white") for _ in range(2)]
            frames[0].save(path, save_all=True, append_images=frames[1:])
            result = DocumentProcessor(max_total_pixels=1500).process_file(path, "pan_card")
            self.assertIn("file_read_or_ocr_failed", result["review_reasons"])
            self.assertTrue(any("cumulative" in warning for warning in result["ocr"]["warnings"]))

    @unittest.skipUnless(shutil.which("tesseract"), "Local Tesseract not installed")
    def test_blank_scan_returns_nulls_and_review(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "blank.png"
            Image.new("RGB", (600, 400), "white").save(path)
            result = DocumentProcessor().process_file(path, "pan_card")
            self.assertTrue(result["needs_review"])
            self.assertIn("no_readable_document_text", result["review_reasons"])
            self.assertTrue(all(item["value"] is None for item in result["fields"].values()))

    @unittest.skipUnless(shutil.which("tesseract"), "Local Tesseract not installed")
    def test_synthetic_printed_png_local_ocr(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as temp:
            create_fixtures(Path(temp), render_images=True)
            result = DocumentProcessor().process_file(Path(temp) / "pan_card.png")
            self.assertEqual(result["document_type"], "pan_card")
            self.assertEqual(result["fields"]["pan_number"]["value"], "DEMPP4321Q")
            self.assertEqual(result["ocr"]["engine"], "local_tesseract")
            self.assertEqual(len(result["ocr"]["passes"]), 2)


if __name__ == "__main__":
    unittest.main()
