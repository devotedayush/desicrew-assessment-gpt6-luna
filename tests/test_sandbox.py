"""No model/API calls: run with python -m unittest discover -s tests."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from desicrew.sandbox import execute_query, validate_code, SandboxValidationError, SandboxExecutionError


class ValidationTests(unittest.TestCase):
    def test_ordinary_queries_are_allowed(self):
        for code in [
            "result = df['Quantity'].sum()",
            "result = df.groupby('Category').agg({'Quantity': ['sum', 'mean']})",
            "result = df.groupby('Category').agg(total=('Quantity', 'sum'))",
            "result = df[df['Quantity'] < 5].head(10)",
            "result = pd.to_numeric(df['Quantity'], errors='coerce').sum(min_count=1)",
        ]:
            with self.subTest(code=code):
                validate_code(code)

    def test_dangerous_queries_are_rejected(self):
        attacks = [
            "import os\nresult = os.environ",
            "result = __import__('os').environ",
            "result = open('/etc/passwd').read()",
            "result = eval('1+1')",
            "result = df.__class__",
            "result = df.to_csv('/tmp/escape.csv')",
            "result = pd.read_csv('/etc/passwd')",
            "result = pd.read_pickle('/tmp/payload')",
            "result = df.query('Quantity > 0')",
            "result = df.eval('Quantity')",
            "result = df.apply(str)",
            "result = df['Quantity'].map(str)",
            "result = df.groupby('Category').agg('to_csv')",
            "result = df.groupby('Category').agg({'Quantity': 'to_pickle'})",
            "result = df.groupby('Category').agg(lambda x: x.sum())",
            "result = [x for x in df['Quantity']]",
            "for x in df: pass\nresult = 1",
            "while True: pass\nresult = 1",
            "result = df['Quantity'].astype('untrusted.module.Type')",
            "pd = df\nresult = 1",
            "sheets = {}\nresult = 1",
            "df['Quantity'] = 1\nresult = df",
            "f = len\nresult = f(df)",
            "result = df.sum(**{'axis': 0})",
        ]
        for code in attacks:
            with self.subTest(code=code):
                with self.assertRaises(SandboxValidationError):
                    validate_code(code)

    def test_requires_result_and_bounded_code(self):
        for code in ["df.head()", "", "result = 1\n" + "#padding" * 2_000]:
            with self.subTest(code=code[:30]):
                with self.assertRaises(SandboxValidationError):
                    validate_code(code)


@unittest.skipUnless(sys.platform == "linux", "The execution sandbox requires Linux")
class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.sheets = {"Inventory": pd.DataFrame({"SKU": ["A", "B", "C"], "Category": ["Tools", "Tools", "Home"], "Quantity": [2, 8, 10]})}

    def test_scalar_grouped_and_filtered_results(self):
        self.assertEqual(execute_query("result = df['Quantity'].sum()", self.sheets)["value"], 20)
        grouped = execute_query("result = df.groupby('Category')['Quantity'].sum().reset_index()", self.sheets)
        self.assertEqual(grouped["rows"], [{"Category": "Home", "Quantity": 10}, {"Category": "Tools", "Quantity": 10}])
        filtered = execute_query("result = df[df['Quantity'] < 5]", self.sheets)
        self.assertEqual(filtered["rows"][0]["SKU"], "A")

    def test_parent_dataframe_is_unchanged(self):
        before = self.sheets["Inventory"].copy(deep=True)
        execute_query("result = df.drop(columns=['Quantity'], inplace=True)", self.sheets)
        pd.testing.assert_frame_equal(before, self.sheets["Inventory"])

    def test_result_is_bounded_and_non_finite_values_become_null(self):
        large = {"Inventory": pd.DataFrame({"Quantity": range(250)})}
        result = execute_query("result = df", large)
        self.assertEqual(result["row_count"], 250)
        self.assertEqual(len(result["rows"]), 200)
        self.assertTrue(result["truncated"])
        self.assertIsNone(execute_query("result = float('nan')", self.sheets)["value"])

    def test_nested_collections_report_recursive_truncation(self):
        large = {"Inventory": pd.DataFrame({"SKU": [f"S{n}" for n in range(250)]})}
        result = execute_query("result = {'skus': df['SKU'].to_list()}", large)
        self.assertEqual(len(result["value"]["skus"]), 200)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["complete"])
        self.assertIn("collection_limit", result["truncation_reasons"])

    def test_scalar_and_table_strings_report_clipping(self):
        for code in ["result = df['SKU'].iloc[0]", "result = df.head(1)"]:
            with self.subTest(code=code):
                result = execute_query(code, {"Inventory": pd.DataFrame({"SKU": ["s" * 2_500]})})
                self.assertTrue(result["truncated"])
                self.assertFalse(result["complete"])
                self.assertIn("string_limit", result["truncation_reasons"])

    def test_depth_and_key_clipping_are_explicit(self):
        nested = "'value'"
        for _ in range(10):
            nested = "{'nested': " + nested + "}"
        deep = execute_query("result = " + nested, self.sheets)
        self.assertTrue(deep["truncated"])
        self.assertIn("depth_limit", deep["truncation_reasons"])
        clipped_key = execute_query("result = {'" + "x" * 250 + "': 1}", self.sheets)
        self.assertTrue(clipped_key["truncated"])
        self.assertIn("key_limit", clipped_key["truncation_reasons"])

    def test_small_nested_result_is_complete(self):
        result = execute_query("result = {'skus': df['SKU'].to_list()}", self.sheets)
        self.assertFalse(result["truncated"])
        self.assertTrue(result["complete"])
        self.assertEqual(result["truncation_reasons"], [])

    def test_runtime_failure_is_reported_without_traceback(self):
        with self.assertRaises(SandboxExecutionError) as caught:
            execute_query("result = df['Not a column'].sum()", self.sheets)
        self.assertNotIn("Traceback", str(caught.exception))

    def test_timeout_stops_child(self):
        with self.assertRaisesRegex(SandboxExecutionError, "time limit"):
            execute_query("result = len(df)", self.sheets, timeout=0.01)

    def test_application_secrets_are_not_inherited(self):
        original_popen = subprocess.Popen
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret-not-a-real-key", "AWS_SECRET_ACCESS_KEY": "other-test-secret"}):
            with patch("desicrew.sandbox.subprocess.Popen", wraps=original_popen) as mocked:
                self.assertEqual(execute_query("result = len(df)", self.sheets)["value"], 3)
                env = mocked.call_args.kwargs["env"]
                self.assertNotIn("OPENAI_API_KEY", env)
                self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
                self.assertTrue(mocked.call_args.kwargs["close_fds"])
                self.assertIn("-I", mocked.call_args.args[0])

    def test_os_filter_denies_file_open_network_and_subprocess(self):
        # Trusted test code intentionally probes operations that generated code
        # cannot reach through the AST allowlist. Each must also fail at OS level.
        root = str(Path(__file__).resolve().parents[1])
        script = f"""
import sys, socket, subprocess, json, os
sys.path.insert(0, {root!r})
from desicrew._sandbox_runner import _install_seccomp
_install_seccomp()
checks = {{}}
for name, action in [
    ('file_read', lambda: open('/etc/passwd').read()),
    ('file_write', lambda: open('/tmp/desicrew-sandbox-test-escape', 'w')),
    ('network', lambda: socket.socket()),
    ('subprocess', lambda: subprocess.Popen(['/bin/true'])),
]:
    try:
        action()
        checks[name] = False
    except PermissionError:
        checks[name] = True
print(json.dumps(checks))
"""
        response = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=5, check=True)
        self.assertEqual(json.loads(response.stdout), {"file_read": True, "file_write": True, "network": True, "subprocess": True})


if __name__ == "__main__":
    unittest.main()
