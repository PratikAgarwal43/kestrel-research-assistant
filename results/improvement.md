# Improvement Report: Multi-Agent Kestrel Research Assistant

---

## Observed Issue 1: Retrieval Without Recency Weighting

### Problem
During development, queries about retention policies and rate limits retrieved semantically similar but chronologically older chunks. The verifier had no instruction to consider document version or publication date, so it could not distinguish current specifications from deprecated ones. For example, a question about cold-storage retention duration retrieved both `policy-security-compliance:2` (v1, 2024, stating 90-day cold retention) and `policy-data-retention:2` (v2, Sept 2025, stating 30-day cold retention) — semantically similar chunks from conflicting document versions with no resolution guidance.

### Improvement Made
Added explicit recency-aware reasoning instructions to both the Verifier and Synthesizer system prompts:

- **Verifier**: When two retrieved chunks describe the same policy but carry different `published` dates or `version` fields, the Verifier must label the result `conflicting_evidence`, identify which document is newer, and note which supersedes the other.
- **Synthesizer**: When the verifier verdict is `conflicting_evidence`, the Synthesizer must surface both values with source attribution and explicitly state which specification is current.

### Why
Assignment requirement: the verifier must detect conflicting evidence across document versions and surface this to the user. Without recency-weighted instructions, the synthesizer would silently pick one value, giving users incorrect information. Surfacing the conflict is the correct, grounded behavior.

### Before (No Real Numeric Baseline Available)
No automated baseline evaluation run was executed prior to adding the recency-aware instructions, because the evaluation pipeline was built after the verifier prompts were already strengthened. Fabricating a before-number would be dishonest and has been avoided.

**Limitation**: Because the evaluation was run only once (after all prompt improvements were in place), there is no true A/B baseline. This is a known limitation of the evaluation design.

### After (Current System Evaluation)
System evaluated on 16 questions (see `eval_results.jsonl`). Results on conflicting-evidence questions:

| Question ID | Type        | Verifier Verdict       | Faithfulness | Correctness |
|-------------|-------------|------------------------|--------------|-------------|
| q11         | conflicting | conflicting_evidence   | 1.0          | 1.0         |
| q12         | conflicting | conflicting_evidence   | 1.0          | 1.0         |

Both conflicting questions received the correct verdict and full scores, confirming the recency-aware logic works as intended.

### Interpretation
The recency-aware verifier instructions ensure users are never silently given a stale policy value. Instead they are shown both values, which document is newer, and which should be trusted.

---

## Observed Issue 2: Overly Permissive Synthesizer (Hallucination Risk)

### Problem
The original synthesizer system prompt did not explicitly forbid the model from using its parametric knowledge. Without a hard guardrail, the LLM could supplement missing retrieved evidence with memorised text and still produce confident-sounding answers.

### Improvement Made
Added the following hard grounding rule to the Synthesizer system prompt:

> "You MUST NOT use any knowledge not present in the retrieved evidence blocks above. If the evidence is insufficient, state that clearly — do not supplement from memory."

Added a parallel instruction to the Verifier:

> "Reject any answer claim that cannot be directly traced to a retrieved chunk_id."

### Why
The system must be grounded — answers must come only from the retrieved corpus. Hallucination is the single biggest failure mode for RAG systems in production.

### After
All 16 evaluation questions produced answers whose claims are directly traceable to cited chunk IDs. Unsupported questions (q13, q14) correctly returned `insufficient_evidence` verdicts with no fabricated content.

---

## Observed Issue 3: No Query Decomposition for Multi-Hop Questions

### Problem
Multi-hop questions (e.g., "What query engine was introduced in Kestrel 4.0, and what storage tiers does it scan?") require evidence from multiple distinct documents. A single retrieval query targeting the combined question often fails to surface all necessary chunks.

### Improvement Made
Added explicit query decomposition logic to the Planner system prompt:

- For `multi_hop` and `multi_document` question types, the Planner generates multiple independent sub-queries.
- The Researcher executes each sub-query separately and merges the retrieved chunks before passing them to the Verifier.
- The `retrieval_strategy` field in `AgentState` distinguishes `single_shot` (one query) from `parallel` (multiple concurrent queries) and `sequential` (dependent queries where query 2 depends on the answer to query 1).

### Why
The system must support multi-hop reasoning — questions requiring sequential or cross-document evidence.

### After
Multi-hop questions (q7, q8) and multi-document questions (q9, q10) all received `supported` verdicts with all expected chunk IDs present in `retrieved_chunk_ids`, confirming that decomposed retrieval surfaces the necessary evidence.

---

## Summary of Current Evaluation Metrics

| Metric                  | Value |
|-------------------------|-------|
| Retrieval Hit@k         | 1.00  |
| Mean Faithfulness       | 1.00  |
| Mean Relevance          | 1.00  |
| End-to-End Correctness  | 1.00  |
| Citation Precision      | 1.00  |

> **Note on perfect scores**: These scores reflect a well-constructed evaluation set with clear expected answers and a system with strong grounding guardrails. A more adversarial test set with paraphrased questions or partial corpus coverage would likely reveal lower scores. The value of this evaluation is primarily in confirming that the system handles all required question types (single-hop, multi-hop, multi-document, conflicting, unsupported, follow-up) correctly, not in claiming production-grade robustness.