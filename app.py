"""
Kestrel Research Assistant — Streamlit UI
Connects to the real multi-agent backend defined in the notebook.
All response fields are read directly from the AgentState returned by app.invoke().
"""
import streamlit as st
import time
import os
import re

# ---------------------------------------------------------
# PAGE CONFIGURATION & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="Kestrel Research Assistant",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0B0F19;
        color: #F3F4F6;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #311042 100%);
        padding: 24px 28px;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 24px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        background-color: #064E3B;
        color: #34D399;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #059669;
        letter-spacing: 0.02em;
    }
    .verdict-supported {
        background: #064E3B; color: #34D399;
        border: 1px solid #059669;
        padding: 5px 14px; border-radius: 8px;
        font-weight: 600; font-size: 0.88rem;
        display: inline-block;
    }
    .verdict-partial {
        background: #451A03; color: #FCD34D;
        border: 1px solid #D97706;
        padding: 5px 14px; border-radius: 8px;
        font-weight: 600; font-size: 0.88rem;
        display: inline-block;
    }
    .verdict-conflict {
        background: #431407; color: #FB923C;
        border: 1px solid #EA580C;
        padding: 5px 14px; border-radius: 8px;
        font-weight: 600; font-size: 0.88rem;
        display: inline-block;
    }
    .verdict-insufficient {
        background: #1E1B4B; color: #A5B4FC;
        border: 1px solid #6366F1;
        padding: 5px 14px; border-radius: 8px;
        font-weight: 600; font-size: 0.88rem;
        display: inline-block;
    }
    .citation-chip {
        display: inline-block;
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 6px;
        padding: 4px 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.80rem;
        color: #93C5FD;
        margin: 3px 4px 3px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# BACKEND BOOTSTRAP (loads the pipeline once per session)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="⚙️ Initialising multi-agent pipeline…")
def load_pipeline():
    """
    Import and initialise the real multi-agent pipeline.
    Extracts all cell sources from the notebook and executes them
    in order so that llm, vectorstore, app (LangGraph) etc. are
    available without duplicating code.
    """
    import json, types, sys

    nb_path = os.path.join(os.path.dirname(__file__), "kestrel-research-assistant.ipynb")
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Create an isolated module namespace for the pipeline
    ns = types.ModuleType("kestrel_pipeline")
    ns.__dict__["__builtins__"] = __builtins__

    errors = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])

        # Skip cells that write files, run evals, or zip artifacts
        skip_markers = [
            "from google.colab",   # Colab-only secret access
            "shutil.make_archive",  # zip export cell
            "eval_results_path",   # evaluation runner
            "metrics_summary",     # metrics aggregation
            "improvement_content", # improvement.md writer
            "readme_content",      # README writer
            "app.invoke(initial_state)",  # standalone test runs
            "conversation_turns",  # multi-turn test cell
        ]
        if any(marker in src for marker in skip_markers):
            continue

        # Replace Colab userdata secret calls with env-var fallbacks
        src = src.replace(
            "userdata.get('GEMINI_API_KEY')",
            "os.environ.get('GEMINI_API_KEY', '')"
        )
        src = src.replace(
            "userdata.get('LANGSMITH_API_KEY')",
            "os.environ.get('LANGSMITH_API_KEY', '')"
        )
        src = src.replace(
            "userdata.get('LANGSMITH_TRACING')",
            "os.environ.get('LANGSMITH_TRACING', 'false')"
        )

        try:
            exec(compile(src, f"<cell_{i}>", "exec"), ns.__dict__)
        except Exception as e:
            errors.append((i, str(e)))

    pipeline_app = ns.__dict__.get("app")       # compiled LangGraph app
    vs = ns.__dict__.get("vectorstore")         # Chroma vectorstore (for chunk title lookup)
    chunks_meta = ns.__dict__.get("chunks", []) # raw corpus list for metadata

    return pipeline_app, vs, chunks_meta, errors


# ---------------------------------------------------------
# HELPER: build chunk_id → title map from corpus
# ---------------------------------------------------------
def _build_title_map(chunks_meta: list) -> dict:
    """Returns {chunk_id: title} from the raw corpus list."""
    return {c.get("chunk_id", ""): c.get("title", c.get("doc_id", "")) for c in chunks_meta}


# ---------------------------------------------------------
# HELPER: run the full multi-agent pipeline
# ---------------------------------------------------------
def run_pipeline(pipeline_app, user_question: str, conversation_history: list) -> dict:
    """
    Constructs the initial AgentState and calls app.invoke().
    Returns the full final state dict.
    """
    initial_state = {
        "messages": [],
        "user_question": user_question,
        "conversation_history": conversation_history,
        "question_type": "",
        "retrieval_strategy": "",
        "research_plan": "",
        "sub_questions": [],
        "retrieval_queries": [],
        "retrieved_chunks": [],
        "retrieved_docs_raw": [],
        "evidence": "",
        "claims": [],
        "verifier_findings": "",
        "verifier_verdict": "",
        "final_answer": "",
        "citations": [],
    }
    return pipeline_app.invoke(initial_state)


# ---------------------------------------------------------
# HELPER: render verdict badge
# ---------------------------------------------------------
def _render_verdict(verdict: str):
    verdict_map = {
        "supported": ("🟢", "Supported by Evidence", "verdict-supported"),
        "partially_supported": ("🟡", "Partially Supported", "verdict-partial"),
        "conflicting_evidence": ("🟠", "Conflicting Evidence Detected", "verdict-conflict"),
        "insufficient_evidence": ("🔵", "Insufficient Evidence — Unsupported Query", "verdict-insufficient"),
    }
    icon, label, css_class = verdict_map.get(
        verdict, ("⚪", verdict or "Unknown", "verdict-insufficient")
    )
    st.markdown(
        f'<span class="{css_class}">{icon} {label}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# HELPER: render citations as chips
# ---------------------------------------------------------
def _render_citations(citations: list, title_map: dict):
    if not citations:
        return
    chips = ""
    for cid in citations:
        title = title_map.get(cid, "")
        label = f"{cid} — {title}" if title else cid
        chips += f'<span class="citation-chip">{label}</span> '
    st.markdown(chips, unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPER: clean answer text
# Strip leading "### ✦ Answer" / "### Answer" headings that
# the synthesizer occasionally prepends, since we show our own.
# ---------------------------------------------------------
def _clean_answer(text: str) -> str:
    # Remove common synthesizer heading prefixes
    text = re.sub(r"^#{1,4}\s*[✦✧★*]*\s*Answer\s*\n+", "", text.strip(), flags=re.IGNORECASE)
    # Remove raw [svg](...) / image links
    text = re.sub(r"!\[.*?\]\(.*?\.svg.*?\)", "", text)
    text = re.sub(r"\[svg\]\(.*?\)", "", text)
    return text.strip()


# ---------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []      # [{role, content, meta}]
if "conv_history" not in st.session_state:
    st.session_state.conv_history = []  # [{role, content}] for pipeline
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

# ---------------------------------------------------------
# LOAD PIPELINE (once)
# ---------------------------------------------------------
pipeline_app, vectorstore, chunks_meta, boot_errors = load_pipeline()
title_map = _build_title_map(chunks_meta)
pipeline_ok = pipeline_app is not None

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🦅 Kestrel Labs")
    st.markdown("**Multi-Agent Research Assistant**")
    st.markdown("Evidence-grounded answers from Kestrel Labs internal documentation.")
    st.divider()

    st.markdown("#### 📊 System")
    st.markdown("""
    - **Architecture:** Multi-Agent RAG
    - **Agents:** Planner · Researcher · Verifier · Synthesizer
    - **Vector Store:** Chroma (local)
    - **Orchestration:** LangGraph
    - **Observability:** LangSmith
    """)
    st.divider()

    st.markdown("#### ⚡ Status")
    if pipeline_ok:
        st.markdown("🟢 **Pipeline:** Ready")
    else:
        st.markdown("🔴 **Pipeline:** Not loaded")
        if boot_errors:
            with st.expander("Boot errors"):
                for idx, err in boot_errors:
                    st.caption(f"Cell {idx}: {err}")

    corpus_size = len(chunks_meta)
    st.markdown(f"📚 **Corpus:** {corpus_size} chunks")
    st.divider()

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conv_history = []
        st.rerun()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; color: #FFFFFF;">🦅 Kestrel Research Assistant</h1>
                <p style="margin: 6px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
                    Trusted, evidence-grounded answers from your internal knowledge base.
                </p>
            </div>
            <div>
                <span class="status-badge">● System Online</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# WELCOME / SUGGESTION CARDS
# ---------------------------------------------------------
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Try a Research Query")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📦 **Beacons & Alerting**\n\nWhat are the alerting specifications and limits for Beacons?", use_container_width=True):
            st.session_state.current_query = "What are the alerting specifications and limits for Beacons?"
    with col2:
        if st.button("💰 **Pricing & Plans**\n\nWhat are the pricing tiers and event ingestion rate limits?", use_container_width=True):
            st.session_state.current_query = "What are the pricing tiers and event ingestion rate limits on each plan?"
    with col3:
        if st.button("🚫 **Unsupported Query**\n\nWhat is the quantum encryption key rotation schedule for Enterprise data?", use_container_width=True):
            st.session_state.current_query = "What is the quantum encryption key rotation schedule for Enterprise data in Kestrel?"
    st.divider()

# ---------------------------------------------------------
# RENDER CHAT HISTORY
# Each stored message has: role, content, meta (optional dict)
# ---------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            meta = message.get("meta", {})

            # 1. Answer
            st.markdown(_clean_answer(meta.get("final_answer", message["content"])))

            # 2. Verification badge
            verdict = meta.get("verifier_verdict", "")
            if verdict:
                st.markdown("**Verification:**")
                _render_verdict(verdict)

            # 3. Citations
            citations = meta.get("citations", [])
            if citations:
                st.markdown("**Sources:**")
                _render_citations(citations, title_map)

            # 4. Evidence expander
            evidence = meta.get("evidence", "")
            if evidence:
                with st.expander("📂 View Evidence & Sources"):
                    st.markdown(evidence)

            # 5. Agent trace expander
            planner_trace = {k: meta.get(k) for k in [
                "question_type", "retrieval_strategy", "research_plan",
                "sub_questions", "retrieval_queries", "verifier_findings",
                "verifier_verdict",
            ] if meta.get(k)}
            if planner_trace:
                with st.expander("🛠️ View Agent Trace"):
                    st.json(planner_trace)

# ---------------------------------------------------------
# QUERY INPUT
# ---------------------------------------------------------
user_input = st.chat_input("Ask a question about Kestrel Labs documentation…")

# Suggestion card clicked
if st.session_state.current_query:
    user_input = st.session_state.current_query
    st.session_state.current_query = None

# ---------------------------------------------------------
# PROCESS QUERY
# ---------------------------------------------------------
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Stage indicators while pipeline runs
        stage_placeholder = st.empty()

        if not pipeline_ok:
            st.error("Pipeline failed to load. Check sidebar for boot errors.")
            st.stop()

        # Show live agent stages
        with st.status("🤖 **Multi-Agent Pipeline Running…**", expanded=True) as status:
            st.write("🧭 **Planner:** Analysing query and deciding retrieval strategy…")
            # Kick off the real pipeline in the background while stages animate
            import threading
            result_holder = {}

            def _run():
                try:
                    result_holder["state"] = run_pipeline(
                        pipeline_app,
                        user_input,
                        st.session_state.conv_history,
                    )
                except Exception as e:
                    result_holder["error"] = str(e)

            thread = threading.Thread(target=_run)
            thread.start()

            # Animate stages while waiting (purely cosmetic; real work is in thread)
            time.sleep(0.6)
            st.write("🔎 **Researcher:** Querying Chroma vector store…")
            time.sleep(0.4)
            st.write("🛡️ **Verifier:** Checking claims and detecting conflicts…")
            time.sleep(0.4)
            st.write("✍️ **Synthesizer:** Building grounded answer with citations…")

            thread.join(timeout=120)   # Wait up to 2 min for real result
            status.update(label="✨ **Research complete!**", state="complete", expanded=False)

        stage_placeholder.empty()

        # Handle pipeline error
        if "error" in result_holder:
            st.error(f"Pipeline error: {result_holder['error']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Pipeline error: {result_holder['error']}",
                "meta": {}
            })
            st.stop()

        final_state = result_holder.get("state", {})

        # Extract fields from the real AgentState
        final_answer      = final_state.get("final_answer", "")
        verifier_verdict  = final_state.get("verifier_verdict", "")
        verifier_findings = final_state.get("verifier_findings", "")
        citations         = final_state.get("citations", [])
        evidence          = final_state.get("evidence", "")
        question_type     = final_state.get("question_type", "")
        retrieval_strategy= final_state.get("retrieval_strategy", "")
        research_plan     = final_state.get("research_plan", "")
        sub_questions     = final_state.get("sub_questions", [])
        retrieval_queries = final_state.get("retrieval_queries", [])

        # --- 1. ANSWER ---
        clean_answer = _clean_answer(final_answer) if final_answer else "No answer was generated."
        st.markdown(clean_answer)

        # --- 2. VERIFICATION STATUS ---
        if verifier_verdict:
            st.markdown("**Verification:**")
            _render_verdict(verifier_verdict)

        # Spacing
        st.markdown("")

        # --- 3. SOURCES (citations as chips) ---
        if citations:
            st.markdown("**Sources:**")
            _render_citations(citations, title_map)

        # --- 4. EVIDENCE EXPANDER ---
        if evidence:
            with st.expander("📂 View Evidence & Sources"):
                st.markdown(evidence)

        # --- 5. AGENT TRACE EXPANDER ---
        trace_data = {}
        if question_type:
            trace_data["question_type"] = question_type
        if retrieval_strategy:
            trace_data["retrieval_strategy"] = retrieval_strategy
        if research_plan:
            trace_data["research_plan"] = research_plan
        if sub_questions:
            trace_data["sub_questions"] = sub_questions
        if retrieval_queries:
            trace_data["retrieval_queries"] = retrieval_queries
        if verifier_verdict:
            trace_data["verifier_verdict"] = verifier_verdict
        if verifier_findings:
            trace_data["verifier_findings"] = verifier_findings

        if trace_data:
            with st.expander("🛠️ View Agent Trace"):
                st.json(trace_data)

    # ---------------------------------------------------------
    # UPDATE CONVERSATION HISTORY (for multi-turn follow-ups)
    # ---------------------------------------------------------
    st.session_state.conv_history.append({"role": "user", "content": user_input})
    st.session_state.conv_history.append({"role": "assistant", "content": clean_answer})

    # Store full meta so history re-renders correctly
    st.session_state.messages.append({
        "role": "assistant",
        "content": clean_answer,
        "meta": {
            "final_answer": final_answer,
            "verifier_verdict": verifier_verdict,
            "verifier_findings": verifier_findings,
            "citations": citations,
            "evidence": evidence,
            "question_type": question_type,
            "retrieval_strategy": retrieval_strategy,
            "research_plan": research_plan,
            "sub_questions": sub_questions,
            "retrieval_queries": retrieval_queries,
        }
    })
