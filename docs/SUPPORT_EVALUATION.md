# Q2 support assistant evaluation

The original fictional MeadowDesk Markdown files in `docs/kb/` are the knowledge base. The default assistant copies selected source sentences into answers and cites the file, level-two section, and exact quote. It is a deterministic extractive assistant, not a generative LLM. The optional model may select one allowed source sentence; it cannot write the answer.

`python3 scripts/demo_support.py` generated `artifacts/support_demo.json` by calling one `SupportAssistant` instance for all 12 questions. `python3 scripts/evaluate_support.py` replayed the same questions and compared each response with hand-written expectations and the actual source files. Result: **12/12 turns passed**; the detailed expected and observed citations are in `artifacts/support_evaluation.json`.

| Turn | Expected behavior | Expected cited section | Result |
| --- | --- | --- | --- |
| 1 | Give invitation instructions | `01-workspace.md` → Workspace invitations | Pass |
| 2 | Resolve “the link” to invitations; give 7-day expiry | `01-workspace.md` → Workspace invitations | Pass |
| 3 | Give expired-invitation replacement detail | `01-workspace.md` → Workspace invitations | Pass |
| 4 | Recognize repeated invitation request without repeating instructions | `01-workspace.md` → Workspace invitations; `00-scope.md` → Repeated questions | Pass |
| 5 | Switch to cancellation | `02-billing.md` → Cancel a subscription | Pass |
| 6 | Resolve “it” to cancellation; give effective date | `02-billing.md` → Cancel a subscription | Pass |
| 7 | Switch to refund eligibility | `02-billing.md` → Refund policy | Pass |
| 8 | Return to invitations and recognize previously covered resend instruction | `01-workspace.md` → Workspace invitations; `00-scope.md` → Repeated questions | Pass |
| 9 | Switch to export instructions | `04-data.md` → Export projects | Pass |
| 10 | Resolve “it” to project export contents | `04-data.md` → Export projects | Pass |
| 11 | Decline undocumented SAML support | `00-scope.md` → Knowledge boundaries | Pass |
| 12 | Ask which earlier link is meant | `00-scope.md` → Conversation clarification | Pass |

Additional regression cases verify that “How long is the password reset link valid?” returns the documented 30-minute expiry, while “Can admins download invoices?” and “Can editors export projects?” do not borrow a different role's permission. These are conservative knowledge-boundary responses: the source documents do not establish those roles' rights for the requested actions.

The fixed vocabulary and sentence ranking limit the range of natural paraphrases this offline assistant handles. An unknown wording may receive a cited knowledge-boundary response even when a human can infer the answer. Citations prove the quoted text exists in the cited section; they do not establish live account state or product behavior beyond these fictional documents.
