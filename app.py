"""
Kestrel Research Assistant — Streamlit UI
Imports from backend.py (self-contained pipeline module).
All response fields are read directly from the AgentState returned by app.invoke().
"""
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import time
import os
import re
from pathlib import Path

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
    .status-badge-err {
        display: inline-flex;
        align-items: center;
        background-color: #450A0A;
        color: #FCA5A5;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #EF4444;
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
# PIPELINE BOOTSTRAP (loads once per session via cache)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="⚙️ Initialising multi-agent pipeline…")
def load_pipeline(cache_key: str = "v8_dual_groq_gemini"):
    """
    Import backend.py and build the pipeline.
    Returns (pipeline_app, vectorstore, chunks, error_message_or_None).
    """
    try:
        # Check st.secrets explicitly for Streamlit Cloud
        for k in ["GEMINI_API_KEY", "GROQ_API_KEY", "GEMINI_MODEL", "GROQ_MODEL"]:
            if k not in os.environ:
                try:
                    if k in st.secrets:
                        os.environ[k] = st.secrets[k]
                except Exception:
                    pass

        from backend import build_pipeline
        corpus_path = Path(__file__).parent / "corpus.jsonl"
        app, vectorstore, chunks = build_pipeline(corpus_path)
        return app, vectorstore, chunks, None
    except Exception as e:
        import traceback
        # Return full traceback so we can see if the model is failing
        return None, None, [], traceback.format_exc()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _build_title_map(chunks: list) -> dict:
    return {c.get("chunk_id", ""): c.get("title", c.get("doc_id", "")) for c in chunks}


def _render_verdict(verdict: str):
    verdict_map = {
        "supported":            ("🟢", "Supported by Evidence",               "verdict-supported"),
        "partially_supported":  ("🟡", "Partially Supported",                  "verdict-partial"),
        "conflicting_evidence": ("🟠", "Conflicting Evidence Detected",        "verdict-conflict"),
        "insufficient_evidence":("🔵", "Insufficient Evidence — Query Not Supported", "verdict-insufficient"),
    }
    icon, label, css = verdict_map.get(verdict, ("⚪", verdict or "Unknown", "verdict-insufficient"))
    st.markdown(f'<span class="{css}">{icon} {label}</span>', unsafe_allow_html=True)


def _render_citations(citations: list, title_map: dict):
    if not citations:
        return
    chips = ""
    for cid in citations:
        title = title_map.get(cid, "")
        label = f"{cid} — {title}" if title else cid
        chips += f'<span class="citation-chip">{label}</span> '
    st.markdown(chips, unsafe_allow_html=True)


def _clean_answer(text: str) -> str:
    """Strip stray synthesizer headings and broken image markup."""
    text = re.sub(r"^#{1,4}\s*[✦✧★*]*\s*Answer\s*\n+", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"!\[.*?\]\(.*?\.svg.*?\)", "", text)
    text = re.sub(r"\[svg\]\(.*?\)", "", text)
    return text.strip()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_history" not in st.session_state:
    st.session_state.conv_history = []
if "current_query" not in st.session_state:
    st.session_state.current_query = None

# ---------------------------------------------------------
# LOAD PIPELINE
# ---------------------------------------------------------
pipeline_app, vectorstore, chunks_meta, boot_error = load_pipeline(cache_key="v8_dual_groq_gemini")
title_map  = _build_title_map(chunks_meta)
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
    - **Vector Store:** Chroma (in-memory)
    - **Orchestration:** LangGraph
    - **Observability:** LangSmith (optional)
    """)
    st.divider()

    st.markdown("#### ⚡ Status")
    if pipeline_ok:
        st.markdown("🟢 **Pipeline:** Ready")
        st.markdown(f"📚 **Corpus:** {len(chunks_meta)} chunks loaded")
    else:
        st.markdown("🔴 **Pipeline:** Failed to load")
        st.markdown(f"📚 **Corpus:** {len(chunks_meta)} chunks found")
        if boot_error:
            with st.expander("🔍 Boot error details"):
                st.code(boot_error, language="python")

    st.divider()
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conv_history = []
        st.rerun()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
online_badge = (
    '<span class="status-badge">● System Online</span>'
    if pipeline_ok else
    '<span class="status-badge-err">● Pipeline Error</span>'
)
st.markdown(f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 1.8rem; color: #FFFFFF;">🦅 Kestrel Research Assistant</h1>
                <p style="margin: 6px 0 0 0; color: #9CA3AF; font-size: 0.95rem;">
                    Trusted, evidence-grounded answers from your internal knowledge base.
                </p>
            </div>
            <div>{online_badge}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Show a clear error banner if pipeline failed (not just buried in sidebar)
if not pipeline_ok:
    st.error(
        "**Pipeline failed to initialise.** "
        "Check that `GEMINI_API_KEY` is set in Streamlit Cloud secrets → "
        "Settings → Secrets. See the sidebar for the full error trace.",
        icon="🔴",
    )

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
        if st.button("💰 **Pricing & Plans**\n\nWhat are the pricing tiers and event ingestion rate limits on each plan?", use_container_width=True):
            st.session_state.current_query = "What are the pricing tiers and event ingestion rate limits on each plan?"
    with col3:
        if st.button("🚫 **Unsupported Query**\n\nWhat is the quantum encryption key rotation schedule for Enterprise data?", use_container_width=True):
            st.session_state.current_query = "What is the quantum encryption key rotation schedule for Enterprise data in Kestrel?"
    st.divider()

# ---------------------------------------------------------
# RENDER CHAT HISTORY
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
            trace = {k: meta[k] for k in (
                "question_type", "retrieval_strategy", "research_plan",
                "sub_questions", "retrieval_queries",
                "verifier_verdict", "verifier_findings",
            ) if meta.get(k)}
            if trace:
                with st.expander("🛠️ View Agent Trace"):
                    st.json(trace)

# ---------------------------------------------------------
# QUERY INPUT
# ---------------------------------------------------------
user_input = st.chat_input("Ask a question about Kestrel Labs documentation…")

if st.session_state.current_query:
    user_input = st.session_state.current_query
    st.session_state.current_query = None

# ---------------------------------------------------------
# PROCESS QUERY
# ---------------------------------------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if not pipeline_ok:
            st.error(
                "Cannot answer — the pipeline failed to load. "
                "Please add your `GEMINI_API_KEY` to Streamlit secrets and restart the app."
            )
        else:
            final_state = None
            pipeline_err = None
            with st.status("🤖 **Multi-Agent Pipeline Running…**", expanded=True) as status:
                st.write("🧭 **Planner:** Analysing query and deciding retrieval strategy…")
                try:
                    from backend import make_initial_state
                    history_snapshot = list(st.session_state.conv_history)
                    initial = make_initial_state(user_input, history_snapshot)
                    st.write("🔎 **Researcher:** Querying Chroma vector store…")
                    st.write("🛡️ **Verifier:** Checking claims and detecting conflicts…")
                    st.write("✍️ **Synthesizer:** Building grounded answer with citations…")
                    final_state = pipeline_app.invoke(initial)
                    status.update(label="✨ **Research complete!**", state="complete", expanded=False)
                except Exception as exc:
                    import traceback
                    pipeline_err = traceback.format_exc()
                    status.update(label="❌ **Execution error**", state="error", expanded=True)

            # Handle pipeline error
            if pipeline_err or not final_state:
                st.error(f"**Pipeline error during query execution:**\n\n```\n{pipeline_err}\n```")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "❌ An error occurred during query execution. See the error above.",
                    "meta": {},
                })
            else:

                # Extract all AgentState fields
                final_answer       = final_state.get("final_answer", "")
                verifier_verdict   = final_state.get("verifier_verdict", "")
                verifier_findings  = final_state.get("verifier_findings", "")
                citations          = final_state.get("citations", [])
                evidence           = final_state.get("evidence", "")
                question_type      = final_state.get("question_type", "")
                retrieval_strategy = final_state.get("retrieval_strategy", "")
                research_plan      = final_state.get("research_plan", "")
                sub_questions      = final_state.get("sub_questions", [])
                retrieval_queries  = final_state.get("retrieval_queries", [])

                clean_answer = _clean_answer(final_answer) if final_answer else "No answer was generated."

                # 1. Answer
                st.markdown(clean_answer)

                # 2. Verification badge
                if verifier_verdict:
                    st.markdown("**Verification:**")
                    _render_verdict(verifier_verdict)
                st.markdown("")

                # 3. Citations
                if citations:
                    st.markdown("**Sources:**")
                    _render_citations(citations, title_map)

                # 4. Evidence expander
                if evidence:
                    with st.expander("📂 View Evidence & Sources"):
                        st.markdown(evidence)

                # 5. Agent trace expander
                trace_data = {}
                for key, val in [
                    ("question_type",      question_type),
                    ("retrieval_strategy", retrieval_strategy),
                    ("research_plan",      research_plan),
                    ("sub_questions",      sub_questions),
                    ("retrieval_queries",  retrieval_queries),
                    ("verifier_verdict",   verifier_verdict),
                    ("verifier_findings",  verifier_findings),
                ]:
                    if val:
                        trace_data[key] = val
                if trace_data:
                    with st.expander("🛠️ View Agent Trace"):
                        st.json(trace_data)

                # Update conversation history for multi-turn follow-ups
                st.session_state.conv_history.append({"role": "user",      "content": user_input})
                st.session_state.conv_history.append({"role": "assistant", "content": clean_answer})

                # Store full meta for history replay
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": clean_answer,
                    "meta": {
                        "final_answer":       final_answer,
                        "verifier_verdict":   verifier_verdict,
                        "verifier_findings":  verifier_findings,
                        "citations":          citations,
                        "evidence":           evidence,
                        "question_type":      question_type,
                        "retrieval_strategy": retrieval_strategy,
                        "research_plan":      research_plan,
                        "sub_questions":      sub_questions,
                        "retrieval_queries":  retrieval_queries,
                    },
                })
