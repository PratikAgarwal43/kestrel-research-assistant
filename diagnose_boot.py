"""Quick smoke-test of backend.build_pipeline() — run from project root."""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env")
except Exception:
    pass

print(f"GEMINI_API_KEY set: {'YES' if os.environ.get('GEMINI_API_KEY') else 'NO — set it first!'}")

t0 = time.time()
print("\nCalling backend.build_pipeline() ...")
from backend import build_pipeline, make_initial_state

app, vs, chunks = build_pipeline("corpus.jsonl")
print(f"Pipeline built in {time.time()-t0:.1f}s")
print(f"Corpus chunks: {len(chunks)}")
print(f"Chroma count: {vs._collection.count()}")
print(f"LangGraph app: {app}")

print("\nRunning test query: 'What are the alerting specifications and limits for Beacons?'")
state = make_initial_state(
    "What are the alerting specifications and limits for Beacons?", []
)
result = app.invoke(state)

print(f"\nquestion_type:      {result.get('question_type')}")
print(f"retrieval_strategy: {result.get('retrieval_strategy')}")
print(f"verifier_verdict:   {result.get('verifier_verdict')}")
print(f"citations:          {result.get('citations')}")
print(f"\nFINAL ANSWER:\n{result.get('final_answer', '')[:600]}")
