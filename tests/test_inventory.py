import io
import json
import unittest

import pandas as pd

from desicrew.inventory import InventoryAgent, WorkbookContext, load_workbook


class FakeModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.messages = []

    def complete(self, messages, json_mode=False):
        self.messages.append((messages, json_mode))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class FakeSearch:
    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [{"title": "Inventory definitions", "url": "https://example.org/inventory", "snippet": "A reorder point triggers replenishment."}]


def workbook_bytes():
    frame = pd.DataFrame({"SKU": ["A", "B", "C"], "Category": ["Tools", "Tools", "Home"], "Quantity": [2, 8, 10], "Unit Price": [5, 8, 9], "Reorder Level": [3, 5, 4]})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Inventory", index=False)
        pd.DataFrame({"Quantity": [7]}).to_excel(writer, sheet_name="Other", index=False)
    return buffer.getvalue()


class WorkbookTests(unittest.TestCase):
    def test_loads_all_sheets_schema_and_preview(self):
        context = load_workbook(workbook_bytes(), "sample.xlsx")
        self.assertIsInstance(context, WorkbookContext)
        self.assertEqual(list(context.sheets), ["Inventory", "Other"])
        self.assertEqual(context.schema["Inventory"]["row_count"], 3)
        self.assertEqual(context.preview["Inventory"][0]["SKU"], "A")
        json.dumps(context.schema, allow_nan=False)
        json.dumps(context.preview, allow_nan=False)

    def test_csv_and_null_preview(self):
        context = load_workbook(b"SKU,Quantity\nA,2\nB,\n", "inventory.csv")
        self.assertIsNone(context.preview["Inventory"][1]["Quantity"])

    def test_decorative_excel_titles_and_leading_blank_column(self):
        rows = [[None, "Example workbook", None, None], [None, None, None, None], [None, "Product ID", "Hand-In-\nStock", "Cost Price\nTotal (USD)"], [None, "S1", 7, 28], [None, "S2", 3, 21]]
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame(rows).to_excel(writer, index=False, header=False)
        context = load_workbook(buffer.getvalue(), "decorated.xlsx")
        schema = context.schema["Sheet1"]
        self.assertEqual(schema["header_row"], 3)
        self.assertEqual(schema["row_count"], 2)
        self.assertEqual(list(context.sheets["Sheet1"].columns), ["Product ID", "Hand-In- Stock", "Cost Price Total (USD)"])
        self.assertEqual(InventoryAgent(context).ask("Total stock quantity")["result"]["value"], 10)
        self.assertEqual(InventoryAgent(context).ask("Total inventory value")["result"]["value"], 49)

    def test_missing_identifiable_header_fails_explicitly(self):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame([[1, 2], [3, 4]]).to_excel(writer, index=False, header=False)
        with self.assertRaisesRegex(ValueError, "identify.*header"):
            load_workbook(buffer.getvalue(), "no_header.xlsx")

    def test_invalid_empty_and_unsupported_files(self):
        for data, filename in [(b"", "inventory.xlsx"), (b"not a workbook", "inventory.xlsx"), (b"hello", "inventory.xls"), (b"Quantity\n", "empty.csv")]:
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    load_workbook(data, filename)


class LocalSummaryAuditTests(unittest.TestCase):
    def test_actual_scalar_and_explained_query_parameter_pass(self):
        from scripts.evaluate_inventory_llm import audit_summary_numbers
        audit = audit_summary_numbers("There are 23 units; round(2) was used.", {"type": "scalar", "value": 23}, 23, "result = df['Quantity'].sum().round(2)")
        self.assertTrue(audit["passed"])

    def test_added_numeric_fact_and_missing_expected_total_fail(self):
        from scripts.evaluate_inventory_llm import audit_summary_numbers
        for answer in ["There are 23 units and 99 orders.", "There are 99 units.", "Inventory exists."]:
            with self.subTest(answer=answer):
                audit = audit_summary_numbers(answer, {"type": "scalar", "value": 23}, 23)
                self.assertFalse(audit["passed"])


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.workbook = load_workbook(workbook_bytes(), "sample.xlsx")

    def test_unconfigured_mode_is_explicit_and_computes_uploaded_data(self):
        result = InventoryAgent(self.workbook).ask("What is the total inventory value?")
        self.assertEqual(result["mode"], "synthetic-demo")
        self.assertEqual(result["result"]["value"], 164)
        self.assertIn("no LLM", result["warnings"][0])
        self.assertTrue(result["code"])

    def test_demo_grouping_low_stock_and_selected_sheet(self):
        agent = InventoryAgent(self.workbook)
        result = agent.ask("Total quantity by category")
        self.assertEqual(result["result"]["rows"], [{"Category": "Home", "Total": 10}, {"Category": "Tools", "Total": 10}])
        self.assertEqual(agent.ask("Which items are below reorder level?")["result"]["rows"][0]["SKU"], "A")
        self.assertEqual(agent.ask("Total quantity in Other")["result"]["value"], 7)

    def test_unknown_demo_question_does_not_invent_answer(self):
        result = InventoryAgent(self.workbook).ask("Predict next quarter's stock demand")
        self.assertIsNone(result["result"])
        self.assertIn("Configure an LLM", result["answer"])

    def test_configured_model_generates_executes_and_summarizes_real_code(self):
        model = FakeModel(json.dumps({"code": "result = df['Quantity'].sum()", "search_terms": [], "assumptions": []}), "The workbook contains 20 units in stock.")
        result = InventoryAgent(self.workbook, model_client=model).ask("How many units are in stock?")
        self.assertEqual(result["mode"], "llm")
        self.assertEqual(result["result"]["value"], 20)
        self.assertEqual(len(model.messages), 2)
        self.assertTrue(model.messages[0][1])
        self.assertFalse(model.messages[1][1])
        summary_data = json.loads(model.messages[1][0][-1]["content"])
        self.assertEqual(summary_data["result"]["value"], 20)

    def test_single_bare_model_expression_is_transparently_normalized(self):
        raw_code = "df['Quantity'].sum()"
        model = FakeModel(json.dumps({"code": raw_code}), "There are twenty units.")
        result = InventoryAgent(self.workbook, model_client=model).ask("Total quantity")
        self.assertEqual(result["result"]["value"], 20)
        self.assertEqual(result["raw_model_code"], raw_code)
        self.assertTrue(result["code_normalized"])
        self.assertTrue(result["code"].startswith("result = ("))
        self.assertTrue(any("single bare expression" in warning for warning in result["warnings"]))

    def test_normalized_expression_still_cannot_read_files(self):
        raw_code = "pd.read_csv('/etc/passwd')"
        model = FakeModel(json.dumps({"code": raw_code}), json.dumps({"code": raw_code}))
        result = InventoryAgent(self.workbook, model_client=model).ask("Read a file")
        self.assertIsNone(result["result"])
        self.assertTrue(result["code_normalized"])
        self.assertEqual(result["raw_model_code"], raw_code)
        self.assertIn("couldn't safely answer", result["answer"])

    def test_multiple_statements_are_not_format_normalized(self):
        raw_code = "df.head()\nlen(df)"
        model = FakeModel(json.dumps({"code": raw_code}), json.dumps({"code": raw_code}))
        result = InventoryAgent(self.workbook, model_client=model).ask("Show data")
        self.assertFalse(result["code_normalized"])
        self.assertEqual(result["code"], raw_code)
        self.assertIsNone(result["result"])

    def test_multiturn_history_reaches_model_as_bounded_context(self):
        model = FakeModel(json.dumps({"code": "result = len(df)"}), "There are three rows.")
        history = [{"role": "user", "content": "Focus on Tools"}, {"role": "assistant", "content": "Tools has two records."}]
        InventoryAgent(self.workbook, model_client=model).ask("What about the total?", history=history)
        self.assertEqual(model.messages[0][0][1:3], history)

    def test_streamlit_turn_history_reaches_model(self):
        model = FakeModel(json.dumps({"code": "result = len(df)"}), "There are three rows.")
        history = [{"question": "Focus on Tools", "response": {"answer": "Tools has two records.", "code": "result = df[df['Category'] == 'Tools']"}}]
        InventoryAgent(self.workbook, model_client=model).ask("What about the total?", history=history)
        messages = model.messages[0][0]
        self.assertEqual(messages[1], {"role": "user", "content": "Focus on Tools"})
        self.assertIn("Tools has two records.", messages[2]["content"])
        self.assertIn("df['Category']", messages[2]["content"])

    def test_invalid_code_gets_one_safe_repair(self):
        model = FakeModel(json.dumps({"code": "result = open('/etc/passwd').read()"}), json.dumps({"code": "result = len(df)"}), "There are three rows.")
        result = InventoryAgent(self.workbook, model_client=model).ask("Count inventory rows")
        self.assertEqual(result["result"]["value"], 3)
        self.assertEqual(len(model.messages), 3)
        self.assertIn("restricted query failed", model.messages[1][0][-1]["content"])

    def test_repeated_invalid_code_is_blocked(self):
        invalid = json.dumps({"code": "result = pd.read_csv('/etc/passwd')"})
        result = InventoryAgent(self.workbook, model_client=FakeModel(invalid, invalid)).ask("Read a file")
        self.assertIsNone(result["result"])
        self.assertIn("couldn't safely answer", result["answer"])

    def test_search_receives_only_fixed_generic_definitions(self):
        search = FakeSearch()
        model = FakeModel(json.dumps({"code": "result = len(df)", "search_terms": ["reorder point", "Alice bank account 123456", "SKU A"]}), "Three rows. The reorder point triggers replenishment [1].")
        result = InventoryAgent(self.workbook, model_client=model, search_client=search).ask("Check this inventory")
        self.assertEqual(search.queries, ["reorder point inventory definition"])
        self.assertEqual(len(result["search_sources"]), 1)
        self.assertTrue(any("blocked" in warning for warning in result["warnings"]))

    def test_generic_definition_is_sourced_when_search_is_present(self):
        search = FakeSearch()
        model = FakeModel("A reorder point triggers replenishment [1].")
        result = InventoryAgent(self.workbook, model_client=model, search_client=search).ask("What is a reorder point?")
        self.assertEqual(result["result"]["type"], "context")
        self.assertEqual(search.queries, ["reorder point inventory definition"])
        self.assertIn("[1]", result["answer"])

    def test_silent_search_failure_is_visible(self):
        class FailedSearch:
            last_error = "Public definition search unavailable (TimeoutError)"
            def search(self, query):
                return []
        result = InventoryAgent(self.workbook, search_client=FailedSearch()).ask("What is SKU?")
        self.assertEqual(result["search_sources"], [])
        self.assertIn("TimeoutError", " ".join(result["warnings"]))
        self.assertIn("built-in general context", " ".join(result["warnings"]))

    def test_model_failure_does_not_leak_exception_secret(self):
        model = FakeModel(RuntimeError("test-secret-token-from-http-client"))
        result = InventoryAgent(self.workbook, model_client=model).ask("Count rows")
        self.assertNotIn("test-secret-token", result["answer"])
        self.assertIn("RuntimeError", result["answer"])

    def test_summary_failure_keeps_computed_result(self):
        model = FakeModel(json.dumps({"code": "result = len(df)"}), RuntimeError("unavailable"))
        result = InventoryAgent(self.workbook, model_client=model).ask("Count rows")
        self.assertEqual(result["result"]["value"], 3)
        self.assertIn("Computed inventory result", result["answer"])
        self.assertTrue(any("summary was unavailable" in warning for warning in result["warnings"]))

    def test_nested_truncation_is_visible_before_model_summary(self):
        frame = pd.DataFrame({"SKU": [f"S{n}" for n in range(250)]})
        workbook = WorkbookContext({"Inventory": frame}, "large.csv", {}, {})
        model = FakeModel(json.dumps({"code": "result = {'skus': df['SKU'].to_list()}"}), "The displayed list contains these SKUs.")
        result = InventoryAgent(workbook, model_client=model).ask("List all SKUs")
        self.assertTrue(result["result"]["truncated"])
        self.assertFalse(result["result"]["complete"])
        self.assertTrue(result["answer"].startswith("This is a partial, bounded result."))
        self.assertIn("collection was limited to 200 items", " ".join(result["warnings"]))
        self.assertEqual(result["summary_mode"], "deterministic-structured")
        self.assertEqual(len(model.messages), 1)
        self.assertIn("Treat it as incomplete", " ".join(result["warnings"]))

    def test_scalar_clipping_warning_is_not_a_row_limit_claim(self):
        frame = pd.DataFrame({"SKU": ["S" * 2_500]})
        workbook = WorkbookContext({"Inventory": frame}, "long.csv", {}, {})
        model = FakeModel(json.dumps({"code": "result = df['SKU'].iloc[0]"}), "The displayed SKU text is shortened.")
        result = InventoryAgent(workbook, model_client=model).ask("What is the SKU text?")
        self.assertIn("text was limited to 2,000 characters", " ".join(result["warnings"]))
        self.assertNotIn("200 rows", " ".join(result["warnings"]))

    def test_table_summary_bypasses_invented_model_labels(self):
        model = FakeModel(json.dumps({"code": "result = df[['SKU', 'Quantity']].head(2)"}), "Product Z has 999 units.")
        result = InventoryAgent(self.workbook, model_client=model).ask("Show two records")
        self.assertEqual(result["summary_mode"], "deterministic-structured")
        self.assertEqual(len(model.messages), 1)
        self.assertNotIn("Product Z", result["answer"])
        self.assertNotIn("999", result["answer"])
        self.assertIn("SKU: A", result["answer"])
        self.assertIn("Quantity: 2", result["answer"])
        self.assertEqual(result["mode"], "llm")

    def test_numeric_summary_fabrication_falls_back_to_actual_scalar(self):
        for summary in ["The total is 999 units.", "There are ninety-nine units.", "There are 20 units from Product Z.", "There are 20 units and 999 orders."]:
            with self.subTest(summary=summary):
                model = FakeModel(json.dumps({"code": "result = df['Quantity'].sum()"}), summary)
                result = InventoryAgent(self.workbook, model_client=model).ask("Total quantity")
                self.assertEqual(result["summary_mode"], "deterministic-fallback")
                self.assertTrue(result["model_summary_rejected"])
                self.assertEqual(result["answer"], "Computed inventory result: 20.")
                self.assertTrue(any("unsupported numbers or labels" in warning for warning in result["warnings"]))
                self.assertEqual(result["result"]["value"], 20)

    def test_supported_numeric_summary_remains_model_written(self):
        model = FakeModel(json.dumps({"code": "result = df['Quantity'].sum().round(2)"}), "There are twenty units; round(2) was used.")
        result = InventoryAgent(self.workbook, model_client=model).ask("Total quantity")
        self.assertEqual(result["summary_mode"], "llm-numeric-checked")
        self.assertFalse(result["model_summary_rejected"])
        self.assertIn("twenty units", result["answer"])

    def test_grounded_description_does_not_fix_wrong_query_direction(self):
        raw = "result = df.nlargest(2, 'Quantity')"
        model = FakeModel(json.dumps({"code": raw}), json.dumps({"code": raw}))
        result = InventoryAgent(self.workbook, model_client=model).ask("Show the two lowest stock products")
        self.assertEqual(result["code"], raw)
        self.assertIsNone(result["result"])
        self.assertTrue(result["analysis_validation"]["rejected"])
        self.assertEqual(result["analysis_validation"]["status"], "inconclusive")
        self.assertIn("couldn't safely answer", result["answer"])


class RankingIntentTests(unittest.TestCase):
    def setUp(self):
        self.workbook = load_workbook(workbook_bytes(), "sample.xlsx")

    def ask(self, question, *codes, workbook=None):
        model = FakeModel(*(json.dumps({"code": code}) for code in codes))
        response = InventoryAgent(workbook or self.workbook, model_client=model).ask(question)
        return response, model

    def test_descending_for_lowest_is_rejected_and_model_gets_one_correction(self):
        result, model = self.ask("Show two products with the lowest stock", "result = df.nlargest(2, 'Quantity')", "result = df.nsmallest(2, 'Quantity')")
        self.assertEqual(result["analysis_validation"]["status"], "passed")
        self.assertEqual(result["result"]["rows"][0]["SKU"], "A")
        self.assertEqual(len(model.messages), 2)
        self.assertTrue(result["analysis_validation_attempts"][0]["rejected"])
        self.assertIn("Sort 'Quantity' ascending", model.messages[1][0][-1]["content"])

    def test_repeated_wrong_direction_returns_failure_without_result(self):
        code = "result = df.nlargest(2, 'Quantity')"
        result, model = self.ask("Show two products with the lowest quantity", code, code)
        self.assertIsNone(result["result"])
        self.assertTrue(result["analysis_validation"]["rejected"])
        self.assertEqual(result["analysis_validation"]["status"], "inconclusive")
        self.assertEqual(len(model.messages), 2)
        self.assertIn("couldn't safely answer", result["answer"])

    def test_correct_highest_and_explicit_prefix_size_pass(self):
        result, _ = self.ask("Show the two highest stock products", "result = df.nlargest(2, 'Quantity')")
        self.assertEqual(result["analysis_validation"]["status"], "passed")
        self.assertEqual(result["analysis_validation"]["requested_limit"], 2)
        self.assertEqual(result["analysis_validation"]["direction"], "highest")

    def test_any_genuine_cutoff_tie_is_allowed_and_missing_stock_is_excluded(self):
        frame = pd.DataFrame({"SKU": ["A", "B", "C", "D", "E"], "Quantity": [1, 2, 2, None, 9]})
        workbook = WorkbookContext({"Inventory": frame}, "ties.xlsx", {}, {})
        result, _ = self.ask("Show two products with the lowest stock", "result = df.iloc[[0, 2]]", workbook=workbook)
        self.assertEqual(result["analysis_validation"]["status"], "passed")
        self.assertEqual(result["analysis_validation"]["excluded_missing_or_nonfinite_metrics"], 1)
        self.assertEqual([row["SKU"] for row in result["result"]["rows"]], ["A", "C"])

    def test_tied_metric_does_not_allow_a_fabricated_identifier(self):
        code = "result = pd.DataFrame({'SKU': ['Z', 'A'], 'Quantity': [2, 8]})"
        result, _ = self.ask("Show two products with lowest stock", code, code)
        self.assertIsNone(result["result"])
        self.assertTrue(result["analysis_validation"]["rejected"])
        self.assertIn("identifier", result["analysis_validation"]["reason"])

    def test_omitted_metric_retries_then_requires_real_pairs(self):
        result, model = self.ask("Show two products with lowest quantity", "result = df[['SKU']].head(2)", "result = df.nsmallest(2, 'Quantity')")
        self.assertEqual(result["analysis_validation"]["status"], "passed")
        self.assertIn("omits the source metric", result["analysis_validation_attempts"][0]["reason"])
        self.assertEqual(len(model.messages), 2)

    def test_omitted_identifiers_are_not_presented_as_product_ranking(self):
        code = "result = df['Quantity'].nsmallest(2)"
        result, _ = self.ask("Show two products with lowest stock", code, code)
        self.assertIsNone(result["result"])
        self.assertIn("omits a source item identifier", result["analysis_validation"]["reason"])

    def test_ambiguous_stock_fields_are_inconclusive_without_silent_guess(self):
        frame = self.workbook.sheets["Inventory"].copy()
        frame["Stock"] = [99, 95, 90]
        workbook = WorkbookContext({"Inventory": frame}, "ambiguous.xlsx", {}, {})
        result, model = self.ask("Show two products with lowest stock", "result = df.nsmallest(2, 'Quantity')", workbook=workbook)
        self.assertEqual(result["analysis_validation"]["status"], "inconclusive")
        self.assertFalse(result["analysis_validation"]["rejected"])
        self.assertIn("unambiguous", result["analysis_validation"]["reason"])
        self.assertEqual(len(model.messages), 1)
        explicit, _ = self.ask("Show two products with lowest 'Quantity'", "result = df.nsmallest(2, 'Quantity')", workbook=workbook)
        self.assertEqual(explicit["analysis_validation"]["status"], "passed")

    def test_filtered_or_grouped_item_intent_is_inconclusive(self):
        for question in ["Show two products with lowest stock in warehouse West", "Show two products with lowest quantity by category", "Show two products with lowest stock from supplier A", "Show only two products with lowest stock"]:
            with self.subTest(question=question):
                result, model = self.ask(question, "result = df.nsmallest(2, 'Quantity')")
                self.assertEqual(result["analysis_validation"]["status"], "inconclusive")
                self.assertFalse(result["analysis_validation"]["rejected"])
                self.assertEqual(len(model.messages), 1)

    def test_category_rankings_and_non_rank_questions_are_not_applicable(self):
        result, _ = self.ask("Which two categories have lowest stock?", "result = df.groupby('Category')['Quantity'].sum().reset_index()")
        self.assertEqual(result["analysis_validation"]["status"], "not_applicable")
        result, _ = self.ask("Show inventory preview", "result = df.head(3)")
        self.assertEqual(result["analysis_validation"]["status"], "not_applicable")

    def test_wrong_identifier_metric_pair_is_rejected(self):
        code = "result = pd.DataFrame({'SKU': ['A', 'B'], 'Quantity': [1, 2]})"
        result, _ = self.ask("Show two products with lowest quantity", code, code)
        self.assertIsNone(result["result"])
        self.assertIn("pair does not match", result["analysis_validation"]["reason"])

    def test_all_missing_quantities_can_only_return_no_ranked_rows(self):
        frame = pd.DataFrame({"SKU": ["A", "B"], "Quantity": [None, None]})
        workbook = WorkbookContext({"Inventory": frame}, "empty-metric.xlsx", {}, {})
        result, _ = self.ask("Show two products with lowest stock", "result = df[df['Quantity'].notna()]", workbook=workbook)
        self.assertEqual(result["analysis_validation"]["status"], "passed")
        self.assertEqual(result["result"]["row_count"], 0)


if __name__ == "__main__":
    unittest.main()
