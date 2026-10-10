"""Independent answer checks; fixture code is not a real-model quality claim."""
import json
import unittest

import pandas as pd

from desicrew.inventory import InventoryAgent, WorkbookContext, _parse_plan, _bind_schema_columns
from desicrew.sandbox import execute_query
from scripts.inventory_oracle import total, verify_ranking, verify_scalar


class IndependentInventoryOracleTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"SKU": "A", "Quantity": "2", "Value": 1.25},
            {"SKU": "B", "Quantity": 8, "Value": 2.5},
            {"SKU": "C", "Quantity": 2, "Value": None},
            {"SKU": "D", "Quantity": None, "Value": 0.125},
            {"SKU": "E", "Quantity": float("inf"), "Value": 0.0},
        ]
        self.frame = pd.DataFrame(self.records)

    def table(self, rows):
        return {"type": "table", "columns": ["SKU", "Quantity"], "rows": rows, "row_count": len(rows), "complete": True}

    def test_reference_total_preserves_fractional_values(self):
        self.assertEqual(total(self.records, "Value"), 3.875)
        self.assertTrue(verify_scalar({"type": "scalar", "value": 3.875}, 3.875))
        self.assertFalse(verify_scalar({"type": "scalar", "value": 3}, 3.875))

    def test_equal_stock_identifiers_must_still_be_real_and_unique(self):
        for rows in [[{"SKU": "Z", "Quantity": 2}], [{"SKU": "A", "Quantity": 2}, {"SKU": "A", "Quantity": 2}]]:
            self.assertFalse(verify_ranking(self.table(rows), self.records, "SKU", "Quantity", len(rows)))

    def test_cutoff_tie_may_use_any_real_identifier(self):
        self.assertTrue(verify_ranking(self.table([{"SKU": "C", "Quantity": 2}]), self.records, "SKU", "Quantity", 1))

    def test_identifiers_alone_do_not_hide_changed_metric(self):
        self.assertFalse(verify_ranking(self.table([{"SKU": "A", "Quantity": 99}]), self.records, "SKU", "Quantity", 1))

    def test_wrong_order_and_wrong_prefix_are_independently_rejected(self):
        rows = [{"SKU": "B", "Quantity": 8}, {"SKU": "A", "Quantity": 2}]
        self.assertFalse(verify_ranking(self.table(rows), self.records, "SKU", "Quantity", 2))
        self.assertFalse(verify_ranking(self.table(rows[0:1]), self.records, "SKU", "Quantity", 1))

    def test_actual_sandbox_arithmetic_matches_python_reference(self):
        result = execute_query("result = df['Value'].sum()", {"Inventory": self.frame})
        self.assertTrue(verify_scalar(result, total(self.records, "Value")))

    def test_agent_fixture_generated_ranking_executes_and_matches_reference(self):
        # Controlled model transport fixture: tests the full generated-code path
        # and safeguards, never reported as measured real inference accuracy.
        class ModelFixture:
            def complete(self, messages, json_mode=False):
                return json.dumps({"code": "stock = pd.to_numeric(df['Quantity'], errors='coerce')\nmask = stock.notna() & (stock != float('inf')) & (stock != float('-inf'))\neligible = pd.DataFrame({'SKU': df['SKU'][mask], 'Quantity': stock[mask]})\nresult = eligible.sort_values('Quantity', ascending=True).head(2)"})
        context = WorkbookContext({"Inventory": self.frame}, "oracle-fixture.xlsx", {}, {})
        response = InventoryAgent(context, model_client=ModelFixture()).ask("Show two products with the lowest quantity")
        self.assertEqual(response["mode"], "llm")
        self.assertEqual(response["summary_mode"], "deterministic-structured")
        self.assertTrue(verify_ranking(response["result"], self.records, "SKU", "Quantity", 2))
        self.assertEqual(response["analysis_validation"]["status"], "passed")

    def test_schema_whitespace_binding_preserves_raw_code_and_wrong_direction(self):
        frame = pd.DataFrame({"SKU": ["A", "B"], "Hand-In- Stock": [2, 5]})
        workbook = WorkbookContext({"Inventory": frame}, "spaced.xlsx", {}, {})
        raw = "result = df.sort_values('Hand-In-Stock', ascending=False).head(1)"
        plan = _bind_schema_columns(_parse_plan(json.dumps({"code": raw})), workbook)
        self.assertEqual(plan["raw_model_code"], raw)
        self.assertIn("ascending=False", plan["code"])
        self.assertIn("'Hand-In- Stock'", plan["code"])
        self.assertEqual(plan["schema_bindings_normalized"][0]["rule"], "unambiguous-whitespace-only")

    def test_schema_binding_never_changes_predicate_values_or_ambiguous_columns(self):
        frame = pd.DataFrame({"SKU": ["Hand-In-Stock"], "Hand-In- Stock": [2], "Hand-In -Stock": [3]})
        workbook = WorkbookContext({"Inventory": frame}, "ambiguous.xlsx", {}, {})
        for code in ["result = df[df['SKU'] == 'Hand-In-Stock']", "result = df['Hand-In-Stock'].sum()"]:
            plan = _bind_schema_columns(_parse_plan(json.dumps({"code": code})), workbook)
            self.assertEqual(plan["code"], code)
            self.assertEqual(plan["schema_bindings_normalized"], [])

    def test_schema_bound_model_fixture_executes_but_wrong_rank_is_withheld(self):
        frame = pd.DataFrame({"SKU": ["A", "B"], "Hand-In- Stock": [2, 5]})
        workbook = WorkbookContext({"Inventory": frame}, "spaced.xlsx", {}, {})
        class ModelFixture:
            def complete(self, messages, json_mode=False):
                return json.dumps({"code": "result = df.sort_values('Hand-In-Stock', ascending=False).head(1)"})
        response = InventoryAgent(workbook, model_client=ModelFixture()).ask("Show one product with the lowest stock")
        self.assertIsNone(response["result"])
        self.assertTrue(response["analysis_validation"]["rejected"])
        self.assertTrue(response["schema_bindings_normalized"])
        self.assertIn("ascending=False", response["code"])

    def test_literal_df_sheet_rebinding_uses_the_correct_second_schema(self):
        first = pd.DataFrame({"SKU": ["First"], "Hand-In- Stock": [99]})
        second = pd.DataFrame({"SKU": ["Second"], "Hand-In -Stock": [2]})
        workbook = WorkbookContext({"First": first, "Second": second}, "two-sheets.xlsx", {}, {})
        raw = "df = sheets['Second']\nresult = df.nsmallest(1, 'Hand-In-Stock')"
        plan = _bind_schema_columns(_parse_plan(json.dumps({"code": raw})), workbook)
        self.assertEqual(plan["schema_bindings_normalized"][0]["bound"], "Hand-In -Stock")
        self.assertNotIn("'Hand-In- Stock'", plan["code"])
        result = execute_query(plan["code"], workbook.sheets)
        self.assertTrue(verify_ranking(result, second.to_dict(orient="records"), "SKU", "Hand-In -Stock", 1))

    def test_ambiguous_or_complex_df_rebinding_skips_schema_normalization(self):
        first = pd.DataFrame({"SKU": ["First"], "Hand-In- Stock": [99]})
        second = pd.DataFrame({"SKU": ["Second"], "Hand-In -Stock": [2]})
        workbook = WorkbookContext({"First": first, "Second": second}, "two-sheets.xlsx", {}, {})
        for raw in ["df = sheets['Second'].copy()\nresult = df['Hand-In-Stock']", "df = sheets['Second']\ndf = df.copy()\nresult = df['Hand-In-Stock']"]:
            plan = _bind_schema_columns(_parse_plan(json.dumps({"code": raw})), workbook)
            self.assertEqual(plan["code"], raw)
            self.assertEqual(plan["schema_bindings_normalized"], [])


if __name__ == "__main__":
    unittest.main()
