"""Streamlit presentation layer; core workflows are independent of Streamlit."""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from desicrew.llm import OpenAICompatibleClient

ROOT = Path(__file__).resolve().parents[1]

def sidebar_settings():
    with st.sidebar:
        st.title("◈ AI workbench")
        st.caption("DesiCrew Version 1 • GPT-6 Luna for Q1/Q2")
        entered_key = st.text_input("OpenAI API key", type="password", key="openai_luna_key",
                                    help="Kept in this session only. Alternatively set OPENAI_API_KEY before starting.")
        try:
            if entered_key.strip():
                from desicrew.openai_luna import OpenAILunaClient
                configured = OpenAILunaClient(api_key=entered_key.strip())
            else:
                configured = OpenAICompatibleClient.from_env()
        except ValueError as exc:
            st.warning(str(exc))
            configured = None
        st.session_state["configured_model"] = configured
        st.toggle("Enable configured LLM", value=False, disabled=configured is None, key="enable_llm",
                  help="No model requests are made until enabled. Remote providers may charge for requests.")
        if configured:
            from urllib.parse import urlparse
            destination = urlparse(configured.base_url)
            st.caption(f"Configured model: {configured.model}")
            st.caption("Model destination: " + destination.scheme + "://" + destination.netloc)
            if not configured.is_local:
                st.checkbox("Allow inventory schema, preview and query results to this provider", key="allow_remote_inventory")
                st.caption("Support mode sends fictional KB snippets and allowlisted KB query concepts. Raw support questions and identity/bank documents stay local.")
        else:
            st.caption("Enter an OpenAI API key to use GPT-6 Luna for Q1 and Q2. Q3 remains local.")
        st.divider()
        st.caption("Do not publish identity files, extracted values, credentials, or private workbooks.")

def _model(inventory=False):
    client = st.session_state.get("configured_model")
    if not st.session_state.get("enable_llm") or not client:
        return None
    if inventory and not client.is_local and not st.session_state.get("allow_remote_inventory"):
        return None
    return client

def _header(kicker, title, description):
    st.caption(kicker)
    st.title(title)
    st.write(description)

def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)

def inventory_page():
    from desicrew.inventory import InventoryAgent, load_workbook
    from desicrew.search import PublicDefinitionSearch
    _header("Q1 · GENERATED QUERIES", "Ask your inventory", "Inspect a workbook, ask a question, and audit the code and executed result behind the answer.")
    available = [ROOT / "data/synthetic/inventory.xlsx"] + sorted((ROOT / "data/private").rglob("*.xlsx"))
    available = [p for p in available if p.is_file()]
    col1, col2 = st.columns([2, 3])
    with col1:
        selected = st.selectbox("Workbook", available, format_func=lambda p: "Synthetic demo inventory" if "synthetic" in p.parts else p.name)
    with col2:
        uploaded = st.file_uploader("Or upload the supplied workbook", type=["xlsx", "csv"], key="inventory_upload")
    raw = uploaded.getvalue() if uploaded else selected.read_bytes()
    name = uploaded.name if uploaded else selected.name
    synthetic = not uploaded and "synthetic" in selected.parts
    if synthetic:
        st.warning("SYNTHETIC DEMONSTRATION. This workbook is not the supplied assessment dataset.")
    model = _model(inventory=True)
    if model is None:
        st.info("Offline planner demonstration: code is really executed, but no LLM is being used. Enable a configured model for the required LLM workflow.")
    else:
        st.success("LLM workflow enabled. Generated code runs in a restricted Linux subprocess.")
    identity = (hashlib.sha256(raw).hexdigest(), bool(model), getattr(model, "model", ""))
    if st.session_state.get("inventory_identity") != identity:
        try:
            context = load_workbook(raw, name)
            st.session_state["inventory_context"] = context
            st.session_state["inventory_agent"] = InventoryAgent(context, model_client=model, search_client=PublicDefinitionSearch())
            st.session_state["inventory_history"] = []
            st.session_state["inventory_identity"] = identity
        except Exception as exc:
            st.error(f"Could not load workbook: {exc}")
            return
    context = st.session_state["inventory_context"]
    with st.expander("Workbook schema and preview", expanded=False):
        st.json(context.schema)
        for sheet_name, frame in context.sheets.items():
            st.caption(sheet_name)
            st.dataframe(frame.head(10), hide_index=True)
    left, right = st.columns([3,1])
    with left:
        st.caption("Try: How many products are there? · Which products need reordering? · What is total inventory value?")
    with right:
        if st.button("Clear inventory chat"):
            st.session_state["inventory_history"] = []
            st.rerun()
    for turn in st.session_state["inventory_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            _inventory_answer(turn["response"])
    if question := st.chat_input("Ask about the selected inventory workbook", key="inventory_question"):
        with st.spinner("Planning and executing a bounded query…"):
            try:
                result = st.session_state["inventory_agent"].ask(question, history=st.session_state["inventory_history"])
            except Exception as exc:
                result = {"answer": "The query could not be completed safely.", "code": "", "result": None, "mode": "error", "warnings": [str(exc)], "search_sources": []}
        st.session_state["inventory_history"].append({"question": question, "response": result})
        st.rerun()
    if st.session_state["inventory_history"]:
        st.download_button("Download this inventory transcript", _json(st.session_state["inventory_history"]), "inventory_transcript.json", "application/json")

def _inventory_answer(result):
    st.write(result.get("answer", "No answer returned"))
    st.caption("Execution mode: " + str(result.get("mode", "unknown")))
    if result.get("summary_mode"):
        st.caption("Summary mode: " + str(result["summary_mode"]))
    for warning in result.get("warnings", []):
        st.warning(warning)
    with st.expander("Generated code and executed result", expanded=True):
        st.code(result.get("code", ""), language="python")
        if result.get("result") is None:
            st.info("No executed result was produced. Review the query error above.")
        else:
            st.json(result["result"])
    sources = result.get("search_sources", [])
    if sources:
        with st.expander("Public definition search sources"):
            for source in sources:
                st.link_button(source.get("title", "Source"), source["url"])
                st.caption(source.get("snippet", ""))

def support_page():
    from desicrew.support import SupportAssistant
    _header("Q2 · GROUNDED SESSION MEMORY", "Support that remembers", "Answers come from an original, fictional MeadowDesk knowledge base. Each response names the supporting document section.")
    model = _model()
    identity = (bool(model), getattr(model,"model", ""))
    if st.session_state.get("support_identity") != identity:
        st.session_state["support_agent"] = SupportAssistant(ROOT / "docs/kb", model_client=model)
        st.session_state["support_history"] = []
        st.session_state["support_identity"] = identity
    cols = st.columns([2,2,3])
    if cols[0].button("Replay 12-turn demonstration"):
        from scripts.demo_support import DEMO_QUESTIONS
        st.session_state["support_agent"].reset()
        st.session_state["support_history"] = []
        with st.spinner("Replaying a full multi-turn support session…"):
            for prompt in DEMO_QUESTIONS:
                response = st.session_state["support_agent"].ask(prompt)
                st.session_state["support_history"].append({"question":prompt,"response":response})
        st.rerun()
    if cols[1].button("Reset support session"):
        st.session_state["support_agent"].reset()
        st.session_state["support_history"] = []
        st.rerun()
    cols[2].caption(f"{len(st.session_state['support_history'])} full exchanges this session • " + ("configured LLM" if model else "local extractive retrieval"))
    with st.expander("Read the source knowledge base"):
        for doc in sorted((ROOT / "docs/kb").glob("*.md")):
            st.markdown(doc.read_text())
    for turn in st.session_state["support_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            response = turn["response"]
            st.write(response["answer"])
            st.caption("Sources: " + "; ".join(
                f"{citation.get('document')} · {citation.get('section')}"
                for citation in response.get("citations", [])))
            with st.expander("Evidence and session context"):
                for citation in response.get("citations", []):
                    st.caption(f"{citation.get('document')} · {citation.get('section')}")
                    st.write(citation.get("quote", ""))
                st.caption("Resolved question: " + response.get("resolved_question", turn["question"]))
                st.caption("Topic: " + str(response.get("topic", "")))
    if question := st.chat_input("Ask about MeadowDesk policies", key="support_question"):
        try:
            response = st.session_state["support_agent"].ask(question)
            st.session_state["support_history"].append({"question":question,"response":response})
            st.rerun()
        except Exception as exc:
            st.error(f"Support response could not be completed: {exc}")
    if st.session_state["support_history"]:
        st.download_button("Download cited session transcript", _json(st.session_state["support_history"]), "support_transcript.json", "application/json")

def documents_page():
    from desicrew.documents import DocumentProcessor, DOCUMENT_SCHEMAS, DOCUMENT_LABELS, review_report
    _header("Q3 · PRIVATE OCR + HUMAN REVIEW", "Review document extraction", "Classify mixed identity and insurance forms, inspect all required fields, and route uncertainty to a human reviewer.")
    st.info("Local processing only. Tesseract reads printed text; handwriting is conservatively flagged. Confidence scores are heuristics, not calibrated probabilities.")
    options = ["Automatic classification"] + list(DOCUMENT_SCHEMAS)
    left, right = st.columns([3,1])
    with left:
        selected_type = st.selectbox("Optional document type override", options, format_func=lambda k: DOCUMENT_LABELS.get(k,k))
    with right:
        threshold = st.number_input("Review threshold", min_value=0.50, max_value=0.99, value=0.90, step=0.01)
    uploaded = st.file_uploader("Upload documents for local processing", type=["pdf","png","jpg","jpeg","tif","tiff","txt"], accept_multiple_files=True, key="document_upload")
    staged_targets = ROOT / "data/private/target_images"
    if staged_targets.is_dir():
        private_files = sorted(p for p in staged_targets.iterdir()
                               if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
    else:
        private_files = sorted(p for p in (ROOT / "data/private").rglob("*")
                               if p.is_file() and p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
                               and any(part.lower().replace(" ", "") == "question3" for part in p.parts))
    use_private = st.checkbox("Use supplied local document files", value=False, disabled=not private_files,
                              help="Process the privately imported document folder; no external upload")
    if use_private:
        st.caption(f"{len(private_files)} local files selected; original paths are kept private")
    use_synthetic = st.checkbox("Use ten synthetic text fixtures instead", value=False)
    with st.expander("Optional local value-line recognition"):
        use_regions = st.checkbox("Use inspected supplied-layout region templates", value=False,
                                  help="Opt-in geometry for the supplied layouts. Aspect-ratio guards and alignment-review flags remain mandatory")
        detect_marks = st.checkbox("Detect marks in inspected option boxes", value=False, disabled=not use_regions,
                                   help="Local pixel evidence only; ambiguous or multiple marks remain unresolved and require review")
        import os
        import importlib.util
        import sys
        model_path = st.text_input("Local TrOCR model directory", value=os.getenv("TROCR_MODEL_PATH", str(ROOT / "data/private/models/trocr-base-handwritten")))
        hwr_ready = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
                         and (Path(model_path) / "model.safetensors").is_file())
        enable_hwr = st.checkbox("Enable offline TrOCR on value-line regions", value=False, key="enable_trocr_regions",
                                disabled=not hwr_ready or not use_regions or st.session_state.get("enable_macos_vision_regions", False),
                                help="Local CPU model only; no downloads or hosted OCR. Experimental and review-only")
        vision_ready = sys.platform == "darwin" and bool(importlib.util.find_spec("Vision") and importlib.util.find_spec("Foundation"))
        enable_vision = st.checkbox("Enable local macOS Vision on value-line regions", value=False,
                                    key="enable_macos_vision_regions",
                                    disabled=not vision_ready or not use_regions or enable_hwr,
                                    help="Runs only in the local macOS Vision framework. Handwritten candidates remain review-required.")
        if not hwr_ready:
            st.caption("Optional CPU dependencies or local safe weights are missing; see docs/HANDWRITING_MODEL.md")
        if sys.platform == "darwin" and not vision_ready:
            st.caption("Optional macOS Vision bridge is missing; see requirements-macos-vision.txt")
        st.caption("The small-model diagnostic was poor on the supplied forms. More non-null candidates do not mean more correct fields. Verify every value locally.")
    if use_synthetic:
        st.warning("SYNTHETIC TEXT DEMONSTRATION. These files do not measure OCR or handwriting accuracy on the assessment scans.")
    if st.button("Classify and extract", type="primary", disabled=not uploaded and not use_synthetic and not use_private):
        options = {}
        if use_regions:
            from desicrew.regions import SAMPLE_REGION_TEMPLATES
            options.update(region_templates=SAMPLE_REGION_TEMPLATES, use_printed_region_ocr=True,
                           use_handwritten_region_ocr=not (enable_hwr or enable_vision),
                           use_checkbox_detection=detect_marks)
        if enable_hwr:
            from desicrew.handwriting import LocalHandwritingRecognizer
            signature = str(Path(model_path).resolve())
            if st.session_state.get("handwriting_model_signature") != signature:
                st.session_state["handwriting_backend"] = LocalHandwritingRecognizer(model_path)
                st.session_state["handwriting_model_signature"] = signature
            options["handwriting_backend"] = st.session_state["handwriting_backend"]
        elif enable_vision:
            from desicrew.mac_vision import LocalMacVisionRecognizer
            options["handwriting_backend"] = LocalMacVisionRecognizer()
        processor = DocumentProcessor(review_threshold=float(threshold), **options)
        results = []
        previews = []
        doc_type = None if selected_type == "Automatic classification" else selected_type
        with st.spinner("Processing locally and validating extracted fields…"):
            if use_synthetic:
                for path in sorted((ROOT / "data/synthetic/documents").glob("*.txt")):
                    results.append(_safe_document_result(processor, path, doc_type))
                    previews.append({"kind": "text", "content": path.read_text()[:12000]})
            elif use_private:
                for path in private_files:
                    results.append(_safe_document_result(processor, path, doc_type))
                    previews.append(_local_source_preview(path.read_bytes(), path.suffix.lower()))
            else:
                for file in uploaded:
                    if file.size > 20_000_000:
                        st.error("Skipped a file over the 20 MB limit")
                        continue
                    # No filenames from uploads are interpreted as paths.
                    with tempfile.TemporaryDirectory(prefix="desicrew-document-") as temp:
                        path = Path(temp) / Path(file.name).name
                        path.write_bytes(file.getvalue())
                        result = _safe_document_result(processor, path, doc_type)
                        result["source_file"] = Path(file.name).name
                        results.append(result)
                        previews.append(_local_source_preview(file.getvalue(), Path(file.name).suffix.lower()))
        st.session_state["document_previews"] = previews
        st.session_state["document_results"] = results
        st.session_state["document_report"] = review_report(results, float(threshold))
    results = st.session_state.get("document_results", [])
    if results:
        st.caption("Displayed results were extracted with review threshold " + str(results[0].get("review_threshold", "unknown")))
        count_review = sum(bool(r.get("needs_review")) for r in results)
        col_a,col_b,col_c=st.columns(3)
        col_a.metric("Documents", len(results))
        col_b.metric("Fields represented", sum(len(r.get("fields",{})) for r in results))
        col_c.metric("Documents for review", count_review)
        for index, result in enumerate(results):
            dtype = result.get("document_type", "unknown")
            with st.expander(f"{index+1}. {DOCUMENT_LABELS.get(dtype,dtype)} · " + ("review required" if result.get("needs_review") else "passes current heuristics"), expanded=index==0):
                previews = st.session_state.get("document_previews", [])
                if index < len(previews) and previews[index]:
                    preview = previews[index]
                    with st.expander("Local source preview"):
                        if preview["kind"] == "image":
                            st.image(preview["content"], caption="Original local source, first page only for PDFs", width=700)
                        else:
                            st.text(preview["content"])
                rows=[]
                for field, details in result.get("fields",{}).items():
                    rows.append({"Field":field,"Value":details.get("value"),"Confidence":details.get("confidence"),"Status":details.get("status"),"Issues":"; ".join(details.get("validation_issues",[]))})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                for reason in result.get("review_reasons",[]):
                    st.warning(str(reason))
                with st.expander("Structured extraction and provenance"):
                    st.json(result)
        st.download_button("Download extraction JSON", _json(results), "extractions.json", "application/json")
        st.download_button("Download flagging report", _json(st.session_state["document_report"]), "review_report.json", "application/json")
        st.caption("Downloads can contain identity and banking data. Keep them private and exclude them from public source repositories and screenshots.")


def _safe_document_result(processor, path, document_type=None):
    """Keep one unreadable upload from interrupting the whole batch."""
    try:
        return processor.process_file(path, document_type=document_type)
    except Exception as exc:
        result = processor.process_text("", document_type=document_type, source_name=Path(path).name)
        result["needs_review"] = True
        result["review_reasons"].append("file_processing_failed:" + type(exc).__name__)
        result["ocr"]["engine"] = "unreadable_input"
        result["ocr"]["warnings"].append("File could not be processed; inspect the original locally")
        return result


def _local_source_preview(raw, suffix):
    """Bounded, in-memory local preview; never saved or transmitted externally."""
    import io
    try:
        if suffix == ".txt":
            return {"kind": "text", "content": raw[:12000].decode("utf-8", errors="replace")}
        from PIL import Image
        if suffix == ".pdf":
            import fitz
            with fitz.open(stream=raw, filetype="pdf") as pdf:
                if pdf.needs_pass or not len(pdf):
                    return None
                page = pdf[0]
                scale = min(1.5, 1500 / max(page.rect.width, page.rect.height))
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                return {"kind": "image", "content": pixmap.tobytes("png")}
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 40_000_000:
                return None
            image.thumbnail((1500, 1500))
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
        return {"kind": "image", "content": output.getvalue()}
    except Exception:
        return None
