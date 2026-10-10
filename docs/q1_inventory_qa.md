# Supplied inventory workbook QA

- Parsed 1 worksheet, 46 inventory rows and 8 columns
- Detected the actual table header after decorative title/blank rows; normalized whitespace and removed all-empty columns
- Loaded data exactly matches an independent direct-pandas read: PASS
- Representative bounded sandbox query checks: 18/18 PASS
- Coverage includes counts, totals, ranking, filters, missing values, duplicates, grouped totals, reconciliation and configured-agent transport with a local model fixture
- Stock arithmetic reconciliation discrepancy rows: 12; source values were preserved, never silently corrected
- Recorded valuation reconciliation discrepancy rows: 0
- Query execution uses Linux seccomp; the same filesystem/network/process restrictions remain enforced
- No paid live-LLM calls were made and no workbook contents were shared externally
- This 18-query QA uses controlled local model fixtures and measures execution, not LLM generation quality. Separate genuine inference measurements and their current limitations are in `LOCAL_MODEL_QA.md` and `artifacts/inventory_model_final_qa_public.json`
- Detailed query outputs remain in local private artifacts; this report contains aggregate counts and status only
- Real-workbook Streamlit AppTest selection, executed-query display and chat-clear flow: PASS
