# Kestrel Labs Multi-Agent Research Assistant

An advanced multi-agent RAG (Retrieval-Augmented Generation) research assistant built using **LangGraph**, **LangChain**, **Chroma**, and **Google Gemini 2.5-Flash**. Designed for Kestrel Labs' internal documentation corpus, the system produces grounded, cited answers with full agent-level traceability.

---

## Problem Statement

Internal documentation at Kestrel Labs spans 25+ documents covering pricing, API specifications, engineering architecture, security policies, post-mortems, and release notes. Engineers and operators must cross-reference multiple documents to answer questions accurately, especially when:

- Evidence lives across several documents (multi-hop / multi-document reasoning)
- Different document versions contradict each other (conflicting evidence)
- A question falls entirely outside the corpus (unsupported — requires an honest refusal)
- The user is continuing a previous conversation (follow-up context)

A single LLM call is insufficient: it cannot retrieve specific internal documents, cannot enforce grounding, cannot detect version conflicts, and cannot validate citations. A multi-agent architecture solves each of these problems with dedicated, specialist agents.

---

## Features

- **Multi-turn conversation** with conversation history injection into every agent
- **Six question type classifiers**: `single_hop`, `multi_hop`, `multi_document`, `conflicting`, `unsupported`, `follow_up`
- **Query decomposition** for multi-hop and multi-document questions (parallel and sequential strategies)
- **Local semantic retrieval** using HuggingFace embeddings and Chroma — no external embedding API calls
- **Verifier/Critic agent** with recency-aware conflict detection across document versions
- **Hard grounding guardrail** — synthesizer is explicitly forbidden from supplementing with parametric knowledge
- **Mandatory chunk citations** in every answer using `[chunk_id | Document Title]` format
- **LangSmith tracing** of every agent transition with graceful degradation if keys are absent
- **Structured evaluation pipeline** with 16 test questions across all six question types
- **Honest refusal** for unsupported questions — no hallucinated answers

---

## Multi-Agent Architecture

The system orchestrates four specialist agents through a **LangGraph** state machine. Each agent reads from and writes to a shared `AgentState` TypedDict, with explicit handoffs enforced by the graph edges.

```
[User Question]
      │
      ▼
[Planner / Router]  ──────────────────────────────────────────┐
 • Classifies question type                                    │
 • Selects retrieval strategy (single_shot / parallel /        │
   sequential)                                                 │
 • Generates 1–N retrieval sub-queries                        │
      │                                                        │
      ▼                                                        │
[Researcher / Retriever]  ◄── Chroma (local) + HuggingFace    │
 • Executes each sub-query via retrieve_from_corpus tool        │
 • Merges results, deduplicates by chunk_id                   │
 • Preserves chunk_id, title, source, version, published      │
      │                                                        │
      ▼                                                        │
[Verifier / Critic]                                           │
 • Checks each claim against retrieved evidence               │
 • Detects recency conflicts (version / published fields)     │
 • Issues verdict: supported / partially_supported /          │
   conflicting_evidence / insufficient_evidence               │
      │                                                        │
      ▼                                                        │
[Synthesizer]                                                 │
 • Generates grounded final answer                            │
 • Attaches mandatory citations: [chunk_id | Title]           │
 • Reports verifier verdict in answer                         │
      │                                                        │
      ▼                                                        │
[Grounded Answer + Validated Citations]  ◄────────────────────┘
```

---

## Agent Responsibilities

### 1. Planner / Router
- Reads the raw user question and full conversation history
- Classifies question type: `single_hop`, `multi_hop`, `multi_document`, `conflicting`, `unsupported`, `follow_up`
- Selects retrieval strategy:
  - `single_shot` — one focused query for simple factual questions
  - `parallel` — multiple independent sub-queries executed simultaneously for multi-document questions
  - `sequential` — chained sub-queries where query 2 depends on the answer to query 1 (multi-hop)
- Outputs a JSON plan with `question_type`, `retrieval_strategy`, `retrieval_queries`, and `plan_reasoning`
- Does NOT answer the question itself

### 2. Researcher / Retriever
- Executes each retrieval query from the Planner plan using the `retrieve_from_corpus` LangChain tool
- Each query calls `vectorstore.similarity_search(query, k=6)` against the local Chroma collection
- Merges and deduplicates results from all sub-queries by `chunk_id`
- Formats each retrieved chunk as a structured evidence block preserving: `chunk_id`, `title`, `source`, `version`, `published`, `content`
- Never invents evidence; retrieval is the only evidence source

### 3. Verifier / Critic
- Receives the user question, conversation history, planner plan, and all retrieved evidence blocks
- Checks every factual claim in the proposed answer against a retrieved chunk
- Applies recency logic: when two chunks describe the same policy with different `published` dates or `version` fields, it must issue `conflicting_evidence` — not silently choose one
- Outputs a JSON verdict: `verdict`, `confidence`, `supported_claims`, `unsupported_claims`, `conflicts`, `missing_evidence`, `reasoning`
- Four possible verdicts:
  - `supported` — all claims are backed by retrieved evidence
  - `partially_supported` — some claims are backed, others are not
  - `conflicting_evidence` — retrieved documents directly contradict each other on the question
  - `insufficient_evidence` — the corpus does not contain enough information to answer (triggers honest refusal)

### 4. Synthesizer
- Receives all prior agent outputs (plan, evidence, verifier verdict)
- Generates the final user-facing answer **only from the retrieved evidence blocks**
- Hard guardrail: explicitly instructed to never supplement from parametric memory
- Must cite every factual claim using `[chunk_id | Document Title]` inline format
- For `conflicting_evidence`: presents both values, identifies the newer document, and flags the conflict
- For `insufficient_evidence`: returns a polite refusal with no fabricated content
- Appends a verifier status line to every answer

---

## Shared State and Agent Handoffs

All agents communicate through a single `AgentState` TypedDict passed through the LangGraph graph:

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # Full conversation history
    user_question: str                         # Current user question
    conversation_history: str                 # Formatted prior turns
    plan: dict                                # Planner output JSON
    retrieved_evidence: str                   # Formatted evidence blocks
    retrieved_chunk_ids: list                 # All chunk IDs retrieved
    verifier_output: dict                     # Verifier verdict JSON
    final_answer: str                         # Synthesizer output
    citations: list                           # Validated citation IDs
    error: Optional[str]                      # Error flag for safe failure
```

**Handoff sequence (LangGraph edges):**
`planner → researcher → verifier → synthesizer → END`

Each node function reads from the state and returns a dictionary of updated keys. LangGraph merges these updates using `operator.add` for list fields and direct replacement for scalar fields.

---

## RAG / Retrieval Approach

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | 100% local, no external API dependency, fast on CPU |
| Vector store | Chroma (local persistent) | No hosted infrastructure required; supports metadata filtering |
| Chunk retrieval | `similarity_search(query, k=6)` | Returns top-6 semantically similar chunks per sub-query |
| Metadata preserved | `chunk_id`, `doc_id`, `title`, `source`, `version`, `published`, `chunk_index` | Enables citation, conflict detection, and recency weighting |
| Corpus size | 154 chunks across 25 documents | Complete Kestrel Labs internal knowledge base |

The corpus is loaded from `corpus.jsonl` and indexed into Chroma once at startup. Each `Document` object carries the full metadata from the corpus as `metadata` fields, making all fields available to agents without re-parsing.

---

## Query Decomposition and Multi-Hop Reasoning

The Planner classifies each question and produces a `retrieval_strategy`:

**Single-shot (simple questions):**
```
Q: "What is the Starter plan price?"
→ retrieval_queries: ["Starter plan pricing"]
→ strategy: single_shot
```

**Parallel (multi-document):**
```
Q: "What are the pricing tiers and the ingest rate limits on each plan?"
→ retrieval_queries: [
    "Kestrel pricing tiers Starter Growth Scale",
    "ingest rate limits per plan rps"
  ]
→ strategy: parallel
```

**Sequential (dependent multi-hop):**
```
Q: "What query engine was introduced in Kestrel 4.0, and what storage tiers does it scan?"
→ retrieval_queries: [
    "Kestrel 4.0 new query engine introduction",
    "Osprey query engine storage tiers"
  ]
→ strategy: sequential (query 2 informed by result of query 1)
```

The Researcher executes all sub-queries and merges results by `chunk_id` before passing them to the Verifier, ensuring all necessary evidence is available for cross-document reasoning.

---

## Guardrails

| Guardrail | Where Enforced | Effect |
|-----------|---------------|--------|
| No parametric knowledge | Synthesizer system prompt | Hard ban on using model memory; must cite from retrieved chunks only |
| Claim traceability | Verifier system prompt | Rejects any claim not traceable to a retrieved chunk_id |
| Unsupported question refusal | Verifier verdict (`insufficient_evidence`) + Synthesizer | Returns polite refusal; no fabricated answer |
| Conflicting evidence surfacing | Verifier (recency check) + Synthesizer | Never silently picks one value; always surfaces the conflict |
| Citation validation | Evaluation pipeline | Citation precision metric checks that all cited chunk_ids were actually retrieved |
| Safe failure | `try/except` around all LLM calls | On parsing error, state carries `error` field; answer degrades gracefully |

---

## Citation and Evidence Handling

Every factual claim in the final answer is cited inline:

```
The Starter plan costs $47 per month [pricing-plans:0 | Plans, Pricing and Limits].
```

Citation format: `[chunk_id | Document Title]`

The Verifier validates that every cited chunk was present in the retrieved evidence before synthesis. The evaluation pipeline additionally computes **citation precision** (fraction of cited chunk IDs that were actually retrieved) as a metric.

---

## Conflict Handling

When the Verifier detects conflicting evidence across document versions, the Synthesizer is instructed to:

1. Present **both** values with their respective source citations
2. Identify which document is newer (by `published` date or `version` field)
3. Explicitly state which specification supersedes the other
4. Recommend the user consult the newer document

Example output for `q11` (cold-storage retention):
> "The Security and Compliance Overview (v1, 2024) states cold-storage retention is 90 days [policy-security-compliance:2]. However, the Data Retention Policy (v2, September 2025) supersedes this with a 30-day post-plan retention period [policy-data-retention:2]. The current policy is 30 days."

---

## Evaluation Results

Evaluated on 16 questions covering all required question types:

| Question Type   | Count | Retrieval Hit@k | Faithfulness | Correctness | Citation Precision |
|-----------------|-------|-----------------|--------------|-------------|-------------------|
| single_hop      | 7     | 1.00            | 1.00         | 1.00        | 1.00              |
| multi_hop       | 2     | 1.00            | 1.00         | 1.00        | 1.00              |
| multi_document  | 2     | 1.00            | 1.00         | 1.00        | 1.00              |
| conflicting     | 2     | 1.00            | 1.00         | 1.00        | 1.00              |
| unsupported     | 2     | 1.00            | 1.00         | 1.00        | 1.00              |
| follow_up       | 1     | 1.00            | 1.00         | 1.00        | 1.00              |
| **Overall**     | **16**| **1.00**        | **1.00**     | **1.00**    | **1.00**          |

See `results/eval_results.jsonl` for per-question answers, verdicts, and latencies.
See `results/improvement.md` for an analysis of the three improvements made during development.

---

## LangSmith Tracing

The multi-agent workflow integrates with **LangSmith** for full execution observability:

```
User Question → Planner → Researcher → Verifier → Synthesizer → Final Answer
```

Each agent step is traced as a named child run, making it possible to inspect:
- Input question and conversation history per agent
- Planner decision and generated sub-queries
- Retrieved chunk IDs per query
- Verifier verdict and reasoning
- Final answer and citations

The system degrades gracefully: if LangSmith credentials are absent or tracing is disabled, the pipeline continues normally with `langsmith_run_url: null` in results.

### Configuration

Set environment variables in `.env` or Google Colab userdata:

```bash
GEMINI_API_KEY=<your_gemini_api_key>
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=<your_langsmith_api_key>
LANGSMITH_PROJECT=kestrel-research-assistant
```

### Sharing with Reviewers

To share your LangSmith project with `radialpulse@nxtwave.co.in`:

1. Log in to [LangSmith](https://smith.langchain.com/)
2. Navigate to **Projects** → `kestrel-research-assistant`
3. Go to **Settings / Members** → **Invite Members**
4. Enter `radialpulse@nxtwave.co.in` with **Viewer** access
5. Alternatively, enable **Public Sharing** to generate a direct share link

---

## Repository Structure

```
kestrel-research-assistant/
├── corpus.jsonl                         # 154-chunk internal knowledge base (25 documents)
├── kestrel-research-assistant.ipynb     # Core notebook: agents, orchestration, evaluation
├── results/
│   ├── eval_questions.jsonl             # 16 evaluation test cases (all 6 question types)
│   ├── eval_results.jsonl               # Per-question answers, verdicts, scores, latencies
│   ├── metrics_summary.json             # Aggregate and per-type evaluation metrics
│   └── improvement.md                   # 3 documented improvements with honest analysis
├── .env.example                         # Template for environment variable configuration
└── README.md                            # This file
```

---

## Setup and Running

### Prerequisites

- Python 3.9+
- Google Gemini API key (free tier works)
- LangSmith API key (optional, for tracing)

### Installation

```bash
pip install langchain langchain-core langchain-community langchain-google-genai \
            langgraph chromadb sentence-transformers
```

### Environment Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

### Running in Google Colab

1. Upload `corpus.jsonl` to the Colab session storage
2. Open `kestrel-research-assistant.ipynb` in Colab
3. Store `GEMINI_API_KEY` and (optionally) `LANGSMITH_API_KEY` in Colab userdata secrets
4. Run all cells in order

### Running the Evaluation

The evaluation cells (CELL 17–18 in the notebook) automatically:
1. Load all 16 questions from `results/eval_questions.jsonl`
2. Run the full 4-agent pipeline for each question
3. Score retrieval quality, faithfulness, relevance, correctness, and citation precision
4. Save results to `results/eval_results.jsonl` and `results/metrics_summary.json`