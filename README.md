# Kestrel Labs Multi-Agent Research Assistant

An advanced, production-grade multi-agent RAG (Retrieval-Augmented Generation) research assistant built using **LangGraph**, **LangChain**, **Chroma**, and **Google Gemini 2.5-Flash**. Designed specifically for Kestrel Labs' internal documentation corpus, this system features robust multi-agent orchestration, local vector retrieval, verifier-led truth checking, conflict resolution, and a comprehensive evaluation suite.

---

## 🏛️ Architecture & Multi-Agent Workflow

The system coordinates four specialized agents via **LangGraph**:

1. **Planner / Router (`planner`)**: Analyzes the incoming user query and conversation history, identifies question type (`single_hop`, `multi_hop`, `conflicting`, `unsupported`, `follow_up`), and generates targeted retrieval queries.
2. **Researcher / Retriever (`researcher`)**: Executes searches against the local Chroma vector store using custom tools and compiles structured evidence blocks containing chunk IDs, titles, publication dates, and versions.
3. **Verifier / Critic (`verifier) & Conflict Resolver**: Inspects claims against retrieved evidence, accounts for document recency and versioning, detects missing data or contradictions, and assigns an overall verification verdict (`supported`, `partially_supported`, `conflicting_evidence`, `insufficient_evidence`).
4. **Synthesizer (`synthesizer`)**: Generates a precise, grounded final answer complete with mandatory chunk citations and verifier status reporting.

[User Query] ---> [Planner / Router]
│
▼
[Researcher] <--- (Chroma Vector Store & Local Embeddings)
│
▼
[Verifier] ---- (Conflict / Recency Check)
│
▼
[Synthesizer] ---> [Grounded Answer + Citations]

---

## 🚀 Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLM:** Google Gemini (`gemini-2.5-flash`)
- **Embeddings:** HuggingFace Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`) — *100% local, no hosted embedding APIs*
- **Vector Store:** Chroma (Local persistent storage)
- **Tracing & Monitoring:** LangSmith

---

## 📁 Repository Structure

├── corpus.jsonl               # Internal knowledge base corpus (154 chunks, 25 docs)
├── results/
│   ├── eval_questions.jsonl   # 15 evaluation test cases
│   ├── eval_results.jsonl     # Execution outputs, verdicts, and latencies
│   ├── metrics_summary.json   # Aggregate evaluation metrics & breakdown
│   └── improvement.md         # Recency-aware conflict resolution analysis
└── README.md                  # Project documentation
---

## 📊 Evaluation Results & Performance

- **Mean Faithfulness:** 0.94 (Optimized via recency weighting)
- **Mean Relevance:** 1.00
- **Supported Question Types:** Single-hop, multi-hop, conflicting specs, unsupported queries, and multi-turn follow-ups.