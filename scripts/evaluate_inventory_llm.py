"""Optional real local-LLM inventory QA and loopback-only serving example.

Weights and torch/transformers are deliberately separate from the baseline app.
No model downloads, credentials, paid APIs, or private data transmission occur
here. Serve a previously downloaded safetensors model from a local directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]


def audit_summary_numbers(answer, result, expected_scalar=None, code=""):
    """Conservative QA check, not a complete semantic/factuality evaluator."""
    allowed = []
    def visit(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            allowed.append(float(value))
        elif isinstance(value, dict):
            for key, part in value.items():
                if key not in {"truncated", "complete"}:
                    visit(part)
        elif isinstance(value, list):
            for part in value:
                visit(part)
        elif isinstance(value, str):
            for match in re.finditer(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
                allowed.append(float(match.group().replace(",", "")))
    visit(result or {})
    # A summary may explain literal parameters such as round(2) or head(5).
    # These are operation metadata, not newly invented workbook measurements.
    if code:
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                    visit(node.value)
        except SyntaxError:
            pass
    claims = []
    unsupported = []
    for match in re.finditer(r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?", answer):
        literal = match.group().replace(",", "")
        number = float(literal)
        claims.append(number)
        places = len(literal.split(".", 1)[1]) if "." in literal and "e" not in literal.lower() else None
        if not any(math.isclose(number, value, rel_tol=1e-10, abs_tol=1e-9) or (places is not None and math.isclose(number, round(value, places), abs_tol=1e-9)) for value in allowed):
            unsupported.append(number)
    mentioned = expected_scalar is None or any(math.isclose(number, float(expected_scalar), rel_tol=1e-10, abs_tol=1e-9) for number in claims)
    return {"passed": bool(claims and not unsupported and mentioned), "numeric_claims": claims, "unsupported_numeric_claims": unsupported, "expected_scalar_mentioned": mentioned, "limitation": "Literal-number consistency only; manually inspect meaning, units, rankings and unsupported categorical claims."}


def serve(args):
    # Explicitly disable implicit authentication/telemetry; load local files only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if not 1 <= args.max_input_tokens <= 8_000 or not 1 <= args.max_new_tokens <= 1_024:
        raise ValueError("Use at most 8,000 input tokens and 1,024 generated tokens")
    if not 0 < args.max_generation_seconds <= 60 or not 0 < args.serve_seconds <= 1_800:
        raise ValueError("Use at most 60 seconds per generation and 1,800 seconds serving")
    if not 1 <= args.port <= 65_535:
        raise ValueError("Invalid loopback port")
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (1_800, 1_801))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model_path = Path(args.model_dir).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32).to("cpu").eval()
    print(json.dumps({"event": "model_ready", "model": args.model_name, "threads": args.threads, "torch": torch.__version__}), flush=True)
    request_number = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass  # Do not place private prompts or values in public process logs.

        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ready", "model": args.model_name, "local_only": True})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            nonlocal request_number
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 500_000:
                self.reply(413, {"error": "Request exceeds the local size limit"})
                return
            started = time.monotonic()
            try:
                payload = json.loads(self.rfile.read(length))
                if payload.get("model") != args.model_name:
                    raise ValueError("Requested model is not the loaded model")
                messages = payload["messages"]
                if not isinstance(messages, list) or not 1 <= len(messages) <= 50:
                    raise ValueError("Invalid message count")
                if any(not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str) for message in messages):
                    raise ValueError("Invalid messages")
                if payload.get("response_format", {}).get("type") == "json_object":
                    data_request = {}
                    for candidate in reversed(messages):
                        try:
                            candidate_data = json.loads(candidate["content"])
                        except (json.JSONDecodeError, TypeError):
                            continue
                        if isinstance(candidate_data, dict) and "schema" in candidate_data and isinstance(candidate_data.get("question"), str):
                            data_request = candidate_data
                            break
                    reminder = "Return exactly one valid JSON object with code, search_terms, and assumptions. The code MUST use result = followed by the calculation requested by the actual question. Copy exact schema column names including internal spaces. Escape newlines inside the code string. Do not include markdown, imports or return statements."
                    if data_request:
                        # Repeat request/schema attributes near generation, never
                        # calculated reference answers or a replacement query.
                        reminder += " Actual question and available columns: " + json.dumps({"question": data_request["question"], "exact_column_names": data_request.get("exact_column_names", {})}, ensure_ascii=False)
                        constraints = data_request.get("planning_constraints", {})
                        if isinstance(constraints, dict) and isinstance(constraints.get("ascending"), bool):
                            reminder += " This ranking request must use ascending=" + str(constraints["ascending"]) + ". Retain actual source identifiers and the requested metric together in a DataFrame."
                    messages = [*messages, {"role": "user", "content": reminder}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt")
                input_tokens = inputs["input_ids"].shape[1]
                if input_tokens > args.max_input_tokens:
                    raise ValueError("Local prompt exceeds the token limit")
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens, max_time=args.max_generation_seconds, pad_token_id=tokenizer.eos_token_id)
                generated = output[0, input_tokens:]
                answer = tokenizer.decode(generated, skip_special_tokens=True)
                request_number += 1
                if args.trace_dir:
                    directory = Path(args.trace_dir)
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f"generation_{request_number:03d}.json").write_text(json.dumps({"model": args.model_name, "messages": messages, "completion": answer, "input_tokens": input_tokens, "output_tokens": len(generated), "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"event": "generation", "request": request_number, "input_tokens": input_tokens, "output_tokens": len(generated), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
                self.reply(200, {"model": args.model_name, "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "length" if len(generated) >= args.max_new_tokens else "stop"}]})
            except Exception as exc:
                # Error type only: no model state, private rows or filesystem paths.
                self.reply(400, {"error": type(exc).__name__})

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.timeout = 1
    server.socket.settimeout(15)
    expires = time.monotonic() + args.serve_seconds
    print(json.dumps({"event": "listening", "url": f"http://127.0.0.1:{args.port}/v1", "model": args.model_name}), flush=True)
    try:
        while time.monotonic() < expires:
            server.handle_request()
    finally:
        server.server_close()


def evaluate(args):
    import sys
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from desicrew.inventory import InventoryAgent, load_workbook
    from desicrew.llm import OpenAICompatibleClient
    from scripts.inventory_oracle import total, verify_scalar, verify_ranking

    workbook_path = Path(args.workbook).resolve(strict=True)
    context = load_workbook(workbook_path.read_bytes(), workbook_path.name)
    # Independently read the supplied worksheet table; no generated answers or
    # runtime planner code is used to construct the verification values.
    direct = pd.read_excel(workbook_path, header=args.header_row - 1, usecols=args.usecols)
    direct.columns = [re.sub(r"\s+", " ", str(name)).strip() for name in direct.columns]
    quantity = args.quantity_column
    value = args.value_column
    records = direct.to_dict(orient="records")
    targets = [
        ("row-count", "How many inventory records are there?", len(direct)),
        ("stock-total", f"What is the total recorded {quantity}?", total(records, quantity)),
        ("value-total", f"What is the total recorded {value}?", total(records, value)),
        ("lowest-stock", f"Show the five products with the lowest {quantity}, using that column for ranking.", None),
    ]
    if args.checks:
        targets = [target for target in targets if target[0] in args.checks]
    targets = targets[:args.questions]
    client = OpenAICompatibleClient(base_url=f"http://127.0.0.1:{args.port}/v1", model=args.model_name, timeout=args.request_timeout)
    agent = InventoryAgent(context, model_client=client)
    checks = []
    history = []
    for label, question, expected in targets:
        started = time.monotonic()
        response = agent.ask(question, history=history if args.history else None)
        result = response.get("result")
        if label == "lowest-stock":
            correct = verify_ranking(result, records, args.id_column, quantity, 5)
        else:
            correct = verify_scalar(result, expected)
        summary_present = bool(result is not None and response["answer"].strip() and response.get("summary_mode", "").startswith("llm"))
        audit = audit_summary_numbers(response["answer"], result, expected_scalar=expected if isinstance(expected, (int, float)) else None, code=response["code"])
        # Structured results intentionally have deterministic, source-grounded
        # summaries. Count verified model-generated query execution separately;
        # never falsely claim that deterministic wording was model written.
        grounded_summary = bool(result is not None and response["answer"].strip() and response.get("summary_mode") in {"deterministic-structured", "llm-numeric-checked"})
        entry = {"label": label, "question": question, "response": response, "query_result_verified": correct, "summary_model_returned": summary_present, "summary_method": response.get("summary_mode"), "summary_numeric_audit": audit, "end_to_end_numeric_pass": bool(correct and summary_present and audit["passed"]), "verified_query_with_grounded_summary": bool(correct and grounded_summary), "elapsed_seconds": time.monotonic() - started}
        checks.append(entry)
        history.append({"question": question, "response": response})
        print(json.dumps({"check": label, "query_result_verified": correct, "summary_model_returned": summary_present, "summary_numeric_audit_passed": audit["passed"], "elapsed_seconds": round(entry["elapsed_seconds"], 2)}), flush=True)
        Path(args.private_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.private_output).write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "model": args.model_name, "live_local_model": True, "deterministic_planner_fallback": False, "checks": checks}, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    passed = sum(item["query_result_verified"] for item in checks)
    summaries = sum(item["summary_model_returned"] for item in checks)
    numeric_passes = sum(item["end_to_end_numeric_pass"] for item in checks)
    print(json.dumps({"checks": len(checks), "verified_queries": passed, "model_summaries": summaries, "end_to_end_literal_number_passes": numeric_passes, "manual_semantic_review_required": True}), flush=True)
    return 0 if all(item["verified_query_with_grounded_summary"] for item in checks) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    server = sub.add_parser("serve", help="Serve local safetensors weights on loopback only")
    server.add_argument("--model-dir", required=True)
    server.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    server.add_argument("--port", type=int, default=8766)
    server.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    server.add_argument("--max-input-tokens", type=int, default=6_000)
    server.add_argument("--max-new-tokens", type=int, default=384)
    server.add_argument("--max-generation-seconds", type=float, default=40)
    server.add_argument("--serve-seconds", type=float, default=900)
    server.add_argument("--trace-dir", help="Optional private directory; raw prompts and values must not be packaged")
    evaluation = sub.add_parser("evaluate", help="Evaluate real agent questions through the existing adapter")
    evaluation.add_argument("--workbook", required=True)
    evaluation.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    evaluation.add_argument("--port", type=int, default=8766)
    evaluation.add_argument("--questions", type=int, choices=range(1, 5), default=4)
    evaluation.add_argument("--checks", nargs="+", choices=["row-count", "stock-total", "value-total", "lowest-stock"], help="Optional subset for a bounded regression rerun")
    evaluation.add_argument("--request-timeout", type=float, default=65)
    evaluation.add_argument("--header-row", type=int, default=6)
    evaluation.add_argument("--usecols", default="B:I")
    evaluation.add_argument("--quantity-column", default="Hand-In- Stock")
    evaluation.add_argument("--value-column", default="Cost Price Total (USD)")
    evaluation.add_argument("--id-column", default="Product ID")
    evaluation.add_argument("--history", action="store_true")
    evaluation.add_argument("--private-output", default=str(ROOT / "outputs/private_samples/local_inventory_llm_qa.json"))
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
        return 0
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
