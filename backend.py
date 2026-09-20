"""
backend.py — Kestrel Research Assistant pipeline
Self-contained importable module. No exec() of notebook cells needed.
All logic mirrors the notebook exactly; prompts/agents are unchanged.
"""
from __future__ import annotations

import json
import os
import re
import operator
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

# ── typing_extensions fallback ────────────────────────────────────────
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore

# ── LangSmith: make tracing completely optional ───────────────────────
_LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING", "false").lower() == "true"
_LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")
if _LANGSMITH_TRACING and _LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
    os.environ.setdefault("LANGCHAIN_API_KEY", _LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGSMITH_PROJECT", "kestrel-research-assistant"))
else:
    # Disable tracing to avoid errors if credentials absent
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Imports (all must be in requirements.txt) ─────────────────────────
from langchain_core.documents import Document                            # langchain-core
from langchain_community.embeddings import HuggingFaceEmbeddings         # langchain-community
from langchain_community.vectorstores import Chroma                      # langchain-community + chromadb
from langchain_google_genai import ChatGoogleGenerativeAI                # langchain-google-genai
from langgraph.graph import StateGraph, END                              # langgraph


# ─────────────────────────────────────────────────────────────────────
# 0. CORPUS LOADING
# ─────────────────────────────────────────────────────────────────────
def _load_corpus(corpus_path: str | Path) -> List[Dict]:
    corpus_path = Path(corpus_path)
    chunks: List[Dict] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return chunks


def _chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    docs: List[Document] = []
    for chunk in chunks:
        docs.append(Document(
            page_content=chunk.get("text", ""),
            metadata={
                "chunk_id":   chunk.get("chunk_id"),
                "doc_id":     chunk.get("doc_id"),
                "title":      chunk.get("title"),
                "category":   chunk.get("category"),
                "owner":      chunk.get("owner", "Unknown"),
                "source_url": chunk.get("source_url", ""),
                "published":  chunk.get("published", ""),
                "version":    chunk.get("version", ""),
            }
        ))
    return docs


# ─────────────────────────────────────────────────────────────────────
# 1. AGENT STATE (matches notebook AgentState TypedDict exactly)
# ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_question: str
    conversation_history: List[Dict[str, str]]
    question_type: str
    retrieval_strategy: str
    research_plan: str
    sub_questions: List[str]
    retrieval_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    retrieved_docs_raw: List[Any]
    evidence: str
    claims: List[str]
    verifier_findings: str
    verifier_verdict: str
    final_answer: str
    citations: List[str]


# ─────────────────────────────────────────────────────────────────────
# 2. HELPER
# ─────────────────────────────────────────────────────────────────────
def _get_content(response) -> str:
    """Safely extract string content from a LangChain chat response."""
    content = response.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
        return "".join(parts).strip()
    return str(content).strip()


# ─────────────────────────────────────────────────────────────────────
# 3. PLANNER / ROUTER AGENT
# Unchanged from notebook cell 19 — only the f-string bug is fixed.
# ─────────────────────────────────────────────────────────────────────
def planner_node(state: AgentState, llm) -> Dict[str, Any]:
    question = state["user_question"]
    history = state.get("conversation_history", [])

    system_prompt = """You are the Planner and Router agent for Kestrel Labs' internal research assistant.

ROLE & OBJECTIVE:
- Understand the user's intent within the context of any prior conversation history.
- For follow-up questions, resolve pronouns/references (e.g. "which one", "them", "what about") into explicit entity names.
- Classify question into one of: "single_hop", "multi_hop", "conflicting", "unsupported", "follow_up".
- Decide retrieval strategy:
    * "single_shot": Simple direct factual questions requiring a single lookup.
    * "parallel": Independent multi-part or comparison questions (e.g. "What are pricing plans AND engineering limits?").
    * "sequential": Dependent multi-hop questions where finding the answer requires a first lookup to identify an entity/component followed by a second targeted query.
- Create a clear, structured research plan.
- Formulate 1 to 3 targeted, standalone retrieval queries.

ALLOWED ACTIONS:
- Analyze conversational context and rewrite coreferences.
- Decompose complex or multi-aspect queries into discrete sub-questions.
- Output a structured research plan and targeted search queries.

FORBIDDEN ACTIONS:
- NEVER attempt to answer the user question yourself.
- NEVER invent facts, chunks, or internal documentation.

OUTPUT FORMAT:
Respond with ONLY a valid JSON object with the following structure:
{
  "question_type": "single_hop" | "multi_hop" | "conflicting" | "unsupported" | "follow_up",
  "retrieval_strategy": "single_shot" | "parallel" | "sequential",
  "research_plan": "1-2 sentence description of what needs to be retrieved",
  "sub_questions": ["sub-question 1", "sub-question 2"],
  "retrieval_queries": ["query 1", "query 2"]
}"""

    # Fixed: was f"""...{question}"""" (4 quotes) — now correctly closed
    user_prompt = (
        "Conversation History:\n"
        + json.dumps(history, indent=2)
        + '\n\nCurrent User Question: "'
        + question
        + '"'
    )

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])
    content = _get_content(response)

    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()

    try:
        parsed = json.loads(content)
        q_type = parsed.get("question_type", "single_hop")
        strat   = parsed.get("retrieval_strategy", "single_shot")
        plan    = parsed.get("research_plan", f"Retrieve documentation for: {question}")
        sub_qs  = parsed.get("sub_questions", [question])
        queries = parsed.get("retrieval_queries", [question])
    except Exception:
        q_type  = "single_hop"
        strat   = "single_shot"
        plan    = f"Retrieve documentation for: {question}"
        sub_qs  = [question]
        queries = [question]

    return {
        "question_type":      q_type,
        "retrieval_strategy": strat,
        "research_plan":      plan,
        "sub_questions":      sub_qs,
        "retrieval_queries":  queries,
    }


# ─────────────────────────────────────────────────────────────────────
# 4. RESEARCHER / RETRIEVER AGENT
# Unchanged from notebook cell 19.
# ─────────────────────────────────────────────────────────────────────
def researcher_node(state: AgentState, retriever, llm) -> Dict[str, Any]:
    queries  = state.get("retrieval_queries", []) or [state["user_question"]]
    strategy = state.get("retrieval_strategy", "single_shot")

    all_chunks: List[Any]        = []
    seen_chunk_ids: set          = set()
    structured_chunks: List[Dict] = []

    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            cid = doc.metadata.get("chunk_id")
            if cid and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                all_chunks.append(doc)
                structured_chunks.append({
                    "query":      q,
                    "chunk_id":   cid,
                    "doc_id":     doc.metadata.get("doc_id"),
                    "title":      doc.metadata.get("title"),
                    "category":   doc.metadata.get("category"),
                    "owner":      doc.metadata.get("owner", "Unknown"),
                    "source_url": doc.metadata.get("source_url", ""),
                    "published":  doc.metadata.get("published"),
                    "version":    doc.metadata.get("version"),
                    "text":       doc.page_content,
                })

    # Sequential multi-hop: generate a targeted Step-2 query from Step-1 evidence
    if strategy == "sequential" and all_chunks:
        summary = "\n".join(
            f"[{c['chunk_id']} | {c['title']}]: {c['text'][:200]}"
            for c in structured_chunks[:4]
        )
        hop_prompt = (
            'You are the Researcher agent executing a sequential multi-hop investigation.\n'
            f'User Question: "{state["user_question"]}"\n\n'
            f"Step 1 Retrieved Evidence:\n{summary}\n\n"
            "Based on this Step 1 evidence, identify the specific entity, architecture component, "
            "or technical term that requires a secondary lookup to complete the answer.\n"
            "Return a targeted search query for Step 2.\n"
            "Respond with ONLY the search query string, nothing else."
        )
        try:
            hop_res     = llm.invoke(hop_prompt)
            step2_query = _get_content(hop_res).strip().strip('"')
            if step2_query and len(step2_query) < 120 and step2_query not in queries:
                for doc in retriever.invoke(step2_query):
                    cid = doc.metadata.get("chunk_id")
                    if cid and cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        all_chunks.append(doc)
                        structured_chunks.append({
                            "query":      f"[Multi-Hop Step 2] {step2_query}",
                            "chunk_id":   cid,
                            "doc_id":     doc.metadata.get("doc_id"),
                            "title":      doc.metadata.get("title"),
                            "category":   doc.metadata.get("category"),
                            "owner":      doc.metadata.get("owner", "Unknown"),
                            "source_url": doc.metadata.get("source_url", ""),
                            "published":  doc.metadata.get("published"),
                            "version":    doc.metadata.get("version"),
                            "text":       doc.page_content,
                        })
        except Exception:
            pass

    # Build grouped evidence string
    docs_by_id: Dict[str, List] = {}
    for sc in structured_chunks:
        docs_by_id.setdefault(sc["doc_id"], []).append(sc)

    sections = []
    for did, doc_chunks in docs_by_id.items():
        header = (
            f"=== DOCUMENT: {did} ({doc_chunks[0]['title']}) "
            f"| Version: {doc_chunks[0]['version']} "
            f"| Published: {doc_chunks[0]['published']} ==="
        )
        lines = [
            f"[Chunk ID: {c['chunk_id']} | Source: {c['source_url']} | Query: {c['query']}]\n{c['text']}"
            for c in doc_chunks
        ]
        sections.append(header + "\n" + "\n\n".join(lines))

    formatted_evidence = "\n\n" + ("\n\n" + "=" * 70 + "\n\n").join(sections)

    return {
        "retrieved_chunks":   structured_chunks,
        "retrieved_docs_raw": all_chunks,
        "evidence":           formatted_evidence,
    }


# ─────────────────────────────────────────────────────────────────────
# 5. VERIFIER / CRITIC AGENT
# Unchanged from notebook cell 21.
# ─────────────────────────────────────────────────────────────────────
def verifier_node(state: AgentState, llm) -> Dict[str, Any]:
    question  = state["user_question"]
    evidence  = state.get("evidence", "")
    raw_docs  = state.get("retrieved_docs_raw", [])

    if not raw_docs or not evidence.strip():
        return {
            "verifier_verdict":  "insufficient_evidence",
            "verifier_findings": "No documentation chunks were retrieved for this query.",
            "claims":            [],
        }

    system_prompt = """You are the Verifier and Critic agent for Kestrel Labs' research assistant.

ROLE & OBJECTIVE:
- Critically evaluate the retrieved evidence against the user question.
- Check whether the evidence contains sufficient facts to directly answer the user query.
- Detect contradictory or conflicting documentation across different document versions or publication dates.
- Assign EXACTLY one verdict from the allowed list.

ALLOWED VERDICTS (MUST return EXACTLY one):
- "supported": The retrieved evidence directly, unambiguously, and sufficiently supports answering the user question.
- "partially_supported": The retrieved evidence answers part of the question, but key aspects are missing.
- "conflicting_evidence": The retrieved chunks contain conflicting or contradictory claims, specifications, or version differences.
- "insufficient_evidence": The retrieved chunks DO NOT contain sufficient evidence to answer the question, or the question asks about something not in the corpus.

FORBIDDEN ACTIONS:
- DO NOT assume or extrapolate facts not explicitly stated in the retrieved chunks.
- DO NOT invent or fabricate facts or citations.

OUTPUT FORMAT:
Respond with ONLY a valid JSON object with the following structure:
{
  "overall_verdict": "supported" | "partially_supported" | "conflicting_evidence" | "insufficient_evidence",
  "reasoning": "Detailed explanation addressing evidence coverage, missing aspects, or contradictions.",
  "claims": ["list of verified factual claims supported by the evidence"],
  "conflicts": ["description of conflicting points and respective chunk_ids/versions if applicable"]
}"""

    user_prompt = f'User Question: "{question}"\n\nRetrieved Evidence Chunks:\n{evidence}'

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])
    content = _get_content(response)

    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()

    try:
        parsed  = json.loads(content)
        verdict = parsed.get("overall_verdict", "insufficient_evidence")
        if verdict not in ("supported", "partially_supported", "conflicting_evidence", "insufficient_evidence"):
            verdict = "insufficient_evidence"
        findings = parsed.get("reasoning", "")
        claims   = parsed.get("claims", [])
    except Exception:
        verdict  = "insufficient_evidence"
        findings = "Failed to parse verification output safely."
        claims   = []

    return {
        "verifier_verdict":  verdict,
        "verifier_findings": findings,
        "claims":            claims,
    }


# ─────────────────────────────────────────────────────────────────────
# 6. SYNTHESIZER AGENT + CITATION VALIDATOR
# Unchanged from notebook cell 21.
# ─────────────────────────────────────────────────────────────────────
def synthesizer_node(state: AgentState, llm) -> Dict[str, Any]:
    question  = state["user_question"]
    evidence  = state.get("evidence", "")
    verdict   = state.get("verifier_verdict", "insufficient_evidence")
    findings  = state.get("verifier_findings", "")
    raw_docs  = state.get("retrieved_docs_raw", [])

    retrieved_chunk_ids = {
        doc.metadata.get("chunk_id")
        for doc in raw_docs
        if doc.metadata.get("chunk_id")
    }

    if verdict == "insufficient_evidence" or not retrieved_chunk_ids:
        return {
            "final_answer": (
                "The available Kestrel documentation does not provide sufficient evidence "
                "to answer this question. No supporting details were found in the internal corpus."
            ),
            "citations": [],
        }

    system_prompt = """You are the Synthesizer agent for Kestrel Labs.

ROLE & OBJECTIVE:
- Generate a clear, precise, and completely grounded answer to the user question using ONLY the provided evidence.
- Every material factual statement MUST include an explicit inline citation in the format [chunk_id | Title].
- Respect the Verifier's findings and verdict.

CRITICAL GUARDRAILS & INSTRUCTIONS:
1. CITATION GUARDRAIL: Cite ONLY chunk IDs present in the retrieved evidence. Format: [chunk_id | Title]. Never invent chunk IDs.
2. CONFLICT GUARDRAIL: If the verifier verdict is "conflicting_evidence":
   - Do NOT silently pick one source.
   - Explicitly identify both pieces of conflicting evidence and their respective chunk IDs/titles.
   - Compare publication dates and/or versions from the chunk metadata.
   - Explain which source appears more current or authoritative based on documented metadata.
3. UNSUPPORTED GUARDRAIL: Never use outside model knowledge to fill missing corpus information.

OUTPUT FORMAT:
Provide the grounded answer clearly structured with inline citations [chunk_id | Title]."""

    user_prompt = (
        f'User Question: "{question}"\n\n'
        f"Verifier Status: {verdict}\n"
        f"Verifier Reasoning: {findings}\n\n"
        f"Retrieved Evidence Chunks:\n{evidence}"
    )

    response     = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ])
    draft_answer = _get_content(response)

    # Citation validation guardrail
    valid_citations: List[str] = []
    found_in_brackets = re.findall(r'\[([a-zA-Z0-9\-_]+:[0-9]+)[^\]]*\]', draft_answer)
    direct_mentions   = re.findall(r'\b([a-zA-Z0-9\-_]+:[0-9]+)\b', draft_answer)

    for cid in found_in_brackets + direct_mentions:
        if cid in retrieved_chunk_ids and cid not in valid_citations:
            valid_citations.append(cid)

    if not valid_citations and verdict in ("supported", "conflicting_evidence"):
        valid_citations = [
            doc.metadata.get("chunk_id")
            for doc in raw_docs[:2]
            if doc.metadata.get("chunk_id")
        ]

    return {
        "final_answer": draft_answer,
        "citations":    valid_citations,
    }


# ─────────────────────────────────────────────────────────────────────
# 7. LANGGRAPH PIPELINE FACTORY
# ─────────────────────────────────────────────────────────────────────
def build_pipeline(corpus_path: str | Path):
    """
    Builds and returns the complete multi-agent pipeline.

    Returns
    -------
    app          : compiled LangGraph CompiledGraph
    vectorstore  : Chroma instance
    chunks       : raw corpus list
    """
    # ── Load corpus ──────────────────────────────────────────────────
    corpus_path = Path(corpus_path)
    chunks      = _load_corpus(corpus_path)
    docs        = _chunks_to_documents(chunks)

    # ── Embeddings (local, no API) ────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Chroma vector store (in-memory; works on Streamlit Cloud) ────
    # Do NOT pass persist_directory — ephemeral in-memory is safe for cloud.
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="kestrel_kb",
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    # ── LLM (Gemini) ─────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to Streamlit Cloud secrets as GEMINI_API_KEY."
        )
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        max_output_tokens=2048,
        google_api_key=api_key,
    )

    # ── Wire node closures (inject dependencies via closure) ─────────
    def _planner(state):    return planner_node(state, llm)
    def _researcher(state): return researcher_node(state, retriever, llm)
    def _verifier(state):   return verifier_node(state, llm)
    def _synthesizer(state):return synthesizer_node(state, llm)

    # ── Build LangGraph ───────────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("planner",     _planner)
    workflow.add_node("researcher",  _researcher)
    workflow.add_node("verifier",    _verifier)
    workflow.add_node("synthesizer", _synthesizer)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner",    "researcher")
    workflow.add_edge("researcher", "verifier")
    workflow.add_edge("verifier",   "synthesizer")
    workflow.add_edge("synthesizer", END)

    app = workflow.compile()
    return app, vectorstore, chunks


# ─────────────────────────────────────────────────────────────────────
# 8. INITIAL STATE FACTORY (used by app.py)
# ─────────────────────────────────────────────────────────────────────
def make_initial_state(user_question: str, conversation_history: list) -> AgentState:
    return {
        "messages":            [],
        "user_question":       user_question,
        "conversation_history": conversation_history,
        "question_type":       "",
        "retrieval_strategy":  "",
        "research_plan":       "",
        "sub_questions":       [],
        "retrieval_queries":   [],
        "retrieved_chunks":    [],
        "retrieved_docs_raw":  [],
        "evidence":            "",
        "claims":              [],
        "verifier_findings":   "",
        "verifier_verdict":    "",
        "final_answer":        "",
        "citations":           [],
    }
