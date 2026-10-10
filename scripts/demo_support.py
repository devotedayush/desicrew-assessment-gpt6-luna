"""Replay twelve actual user/assistant exchanges in one support session."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desicrew.support import SupportAssistant

DEMO_QUESTIONS = [
    "How do I invite teammates?",
    "When does the link expire?",
    "How do I replace an expired link?",
    "How do I invite teammates?",
    "How do I cancel my subscription?",
    "When does it take effect?",
    "Can renewal charges be refunded?",
    "Back to invitations: how do I resend an expired link?",
    "How do I export projects?",
    "What does it include?",
    "Do you support SAML single sign-on?",
    "What about the other link?",
]


def run_demo(kb_path: str | Path | None = None) -> dict:
    assistant = SupportAssistant(kb_path or ROOT / "docs" / "kb")
    exchanges = []
    for number, question in enumerate(DEMO_QUESTIONS, 1):
        response = assistant.ask(question)
        exchanges.append({"turn": number, "user": question, "assistant": response})
    return {
        "product": "MeadowDesk (fictional)",
        "mode": "offline deterministic extractive",
        "exchange_count": len(exchanges),
        "session_memory": "One SupportAssistant instance for all exchanges; no replayed canned responses.",
        "exchanges": exchanges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "support_demo.json")
    args = parser.parse_args()
    result = run_demo()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for exchange in result["exchanges"]:
        response = exchange["assistant"]
        print(f'{exchange["turn"]}. User: {exchange["user"]}')
        print(f'   Assistant: {response["answer"]}')
        print(f'   Topic: {response["topic"]}; status: {response["status"]}; already covered: {response["already_covered"]}')
        for citation in response["citations"]:
            print(f'   Source: {citation["document"]} > {citation["section"]}')
    print(f"\nSaved {result['exchange_count']} full exchanges to {args.output}")


if __name__ == "__main__":
    main()
