"""Check the saved twelve-turn support replay against hand-written expectations."""
from __future__ import annotations

import json
from pathlib import Path

from demo_support import ROOT, run_demo


# These expectations are independent of the retriever's ranking and saved JSON.
# Each quote is checked against the indicated Markdown source section.
EXPECTED = [
    ("answered", "Workspace invitations", "Workspace Owners and Admins can invite members from Settings > Members > Invite.", False),
    ("answered", "Workspace invitations", "Invitation links expire after 7 days.", False),
    ("answered", "Workspace invitations", "To replace an expired invitation link, an Owner or Admin must open Settings > Members and choose Resend invite.", False),
    ("answered", "Workspace invitations", "Workspace Owners and Admins can invite members from Settings > Members > Invite.", True),
    ("answered", "Cancel a subscription", "Only the workspace Owner can cancel a paid subscription from Settings > Billing > Cancel plan.", False),
    ("answered", "Cancel a subscription", "Cancellation takes effect at the end of the current billing period.", False),
    ("answered", "Refund policy", "Renewal charges are not eligible for refunds under this demo policy.", False),
    ("answered", "Workspace invitations", "To replace an expired invitation link, an Owner or Admin must open Settings > Members and choose Resend invite.", True),
    ("answered", "Export projects", "Owners and Admins can export project data from Settings > Data > Export.", False),
    ("answered", "Export projects", "An export is a ZIP file containing project CSV files and uploaded attachments.", False),
    ("no_answer", "Knowledge boundaries", "This assistant answers only from the supplied MeadowDesk support documents.", False),
    ("clarification", "Conversation clarification", "If a follow-up could refer to more than one subject, the assistant should ask which subject the user means before answering.", False),
]
EXPECTED_EXTRA = {
    4: ("Repeated questions", "When the requested information has already been given in this session, the assistant should refer the user to the earlier answer instead of repeating the same instructions."),
    8: ("Repeated questions", "When the requested information has already been given in this session, the assistant should refer the user to the earlier answer instead of repeating the same instructions."),
    11: ("Knowledge boundaries", "If those documents do not cover a requested fact, the assistant should say it does not know and ask the user to contact the fictional support address help@meadowdesk.example."),
}


def evaluate() -> dict:
    transcript = run_demo()
    saved = json.loads((ROOT / "artifacts" / "support_demo.json").read_text(encoding="utf-8"))
    assert transcript == saved, "Saved transcript differs from a live replay"
    assert len(transcript["exchanges"]) == len(EXPECTED) >= 10
    results = []
    for exchange, (status, section, quote, repeated) in zip(transcript["exchanges"], EXPECTED):
        answer = exchange["assistant"]
        citations = answer["citations"]
        assert answer["status"] == status, exchange
        assert answer["already_covered"] is repeated, exchange
        assert (citations[0]["section"], citations[0]["quote"]) == (section, quote), exchange
        expected_citations = [(section, quote)]
        if exchange["turn"] in EXPECTED_EXTRA:
            expected_citations.append(EXPECTED_EXTRA[exchange["turn"]])
        assert [(c["section"], c["quote"]) for c in citations] == expected_citations, exchange
        for citation in citations:
            source = ROOT / "docs" / "kb" / citation["document"]
            assert source.is_file() and citation["quote"] in source.read_text(encoding="utf-8"), exchange
        if repeated:
            assert quote not in answer["answer"], exchange
        if status == "answered" and not repeated:
            assert answer["answer"] == quote, exchange
        results.append({"turn": exchange["turn"], "user": exchange["user"],
                        "expected_status": status, "expected_section": section,
                        "expected_citations": [{"section": sec, "quote": quoted} for sec, quoted in expected_citations],
                        "observed_citations": citations, "repeat_expected": repeated,
                        "repeat_observed": answer["already_covered"], "passed": True})
    # These checks test conversation state, beyond individual answer grounding.
    checks = {
        "followups_resolve_to_current_topic": [2, 6, 10],
        "topic_switches": [5, 7, 8, 9],
        "repetitions_avoid_duplicate_instructions": [4, 8],
        "unsupported_question_declined": [11],
        "ambiguous_reference_clarified": [12],
    }
    for turn in checks["followups_resolve_to_current_topic"]:
        assert "(about " in transcript["exchanges"][turn - 1]["assistant"]["resolved_question"]
    for turn in checks["topic_switches"]:
        assert transcript["exchanges"][turn - 1]["assistant"]["topic"] != transcript["exchanges"][turn - 2]["assistant"]["topic"]
    return {"method": "hand-written turn expectations checked against live replay and source Markdown",
            "exchange_count": len(results), "passed_count": len(results), "failed_count": 0,
            "mode": "deterministic extractive; no generative answer model",
            "conversation_checks": checks, "turns": results}


if __name__ == "__main__":
    result = evaluate()
    output = ROOT / "artifacts" / "support_evaluation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Passed {result['passed_count']}/{result['exchange_count']} independent turn checks; saved {output}")
