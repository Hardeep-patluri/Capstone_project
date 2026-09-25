# Zepto Support Assistant — RAG Module

A small, fully offline-gradable GenAI service: an 8-document Zepto policy
corpus embedded into ChromaDB, a LangGraph-orchestrated intent router +
retriever, a Pydantic-enforced structured output, and a FastAPI wrapper.

**Grading note:** every LLM call is gated behind `MOCK_LLM` (unset or `"1"`
= deterministic offline mock, the graded baseline; `"0"` = optional real-LLM
extension). All example transcripts below were run with `MOCK_LLM` left at
its default — no signup, API key, or network call to any LLM provider was
used to produce them.

## 1. Setup

```bash
cd support_assistant
pip install -r requirements.txt
python ingest.py          # builds & sanity-checks the ChromaDB collection
uvicorn main:app --host 0.0.0.0 --port 7860
```

`ingest.py`'s first run downloads the `all-MiniLM-L6-v2` weights once
(cached locally afterward by `sentence-transformers`/Hugging Face's local
cache — no API key required, just a one-time model-file download) and
persists the Chroma index under `support_assistant/chroma_db/`.

## 2. Architecture (ingestion → embedding → retrieval → generation)

```
docs/doc_01.txt ... doc_08.txt
        |
        v
ingest.load_and_chunk_documents()      [ingestion + chunking]
  - reads each doc_*.txt
  - treats each short policy doc as a single chunk (id: "<doc_id>_chunk_0")
        |
        v
ingest.get_embedder() -> SentenceTransformer("all-MiniLM-L6-v2")
ingest.build_collection()              [embedding]
  - embeds every chunk locally, no API key / no network call to an LLM
  - stores vectors + text + {"doc_id": ...} metadata in the ChromaDB
    PersistentClient collection "zepto_policies" (support_assistant/chroma_db/)
        |
        v
ingest.retrieve_top_k(query, k=3)      [retrieval]
  - called from graph_app.py's `retrieve_and_answer` node
  - embeds the incoming query with the same MiniLM model, queries the
    "zepto_policies" Chroma collection via cosine similarity, returns the
    top-3 chunks. This step always runs for real, in BOTH MOCK_LLM modes.
        |
        v
graph_app.py — LangGraph StateGraph(GraphState)   [orchestration + generation]
  - classify_intent   -> keyword heuristic (mock) / LLM call (MOCK_LLM=0),
                          sets state["intent"]
  - conditional edge   -> retrieve_and_answer  (intent == "policy_question")
                        -> direct_answer       (intent == "general_question")
  - retrieve_and_answer -> calls ingest.retrieve_top_k, then GENERATES the
                            final answer: canned "Based on the retrieved
                            context: ..." template from the top chunk (mock,
                            graded baseline) OR prompts.build_rag_prompt() +
                            a real LLM call (MOCK_LLM=0 extension)
  - direct_answer       -> fixed canned string (mock, graded baseline) OR a
                            direct LLM prompt with no retrieval (MOCK_LLM=0)
        |
        v
schemas.AnswerSchema (Pydantic)        [structured output]
  - mock mode: populated deterministically in code (sources = retrieved
    chunk ids or [], confidence = 1.0) — no LLM output exists to validate
  - MOCK_LLM=0: the LLM's raw JSON is parsed and validated against this
    schema; on failure, graph_app._generate_with_retries() retries up to 2
    more times with prompts.SCHEMA_RETRY_SUFFIX before returning a
    confidence=0.0 error-marked AnswerSchema
        |
        v
main.py — FastAPI POST /ask            [API layer]
  - AskRequest{query} in -> AskResponse{answer, sources, confidence} out
```

**What changes between the two `MOCK_LLM` states:** only the *generation*
step inside `classify_intent`, `retrieve_and_answer`, and `direct_answer`
branches on it (per node, checked via `graph_app.mock_llm_enabled()`).
Chunking, embedding, and Chroma retrieval are identical and always real in
both states. The conditional-edge routing logic itself is also identical in
both states — it only reads `state["intent"]`, which the classify step set.

## 3. Structured prompt template (Task 2)

See `prompts.py` — `RAG_PROMPT_TEMPLATE`. It contains all five
role / context / task / format / length sections, one explicit **negative
constraint** ("Do not answer using information not present in the provided
context..."), and one embedded **few-shot example** (the damaged-item
Q&A pair). It is used by the optional `MOCK_LLM=0` extension inside
`graph_app._llm_generate_grounded_answer`.

## 4. LangGraph graph (Task 3)

`graph_app.build_graph()` builds a `StateGraph(GraphState)` (a `TypedDict`)
with exactly 3 nodes — `classify_intent`, `retrieve_and_answer`,
`direct_answer` — and one conditional edge out of `classify_intent`
(`graph.add_conditional_edges(...)`) routing on `state["intent"]`.

## 5. Example calls (recorded with `MOCK_LLM` left at its default)

Run via FastAPI's `TestClient` against the local app (equivalent to
`curl -X POST http://localhost:7860/ask -d '{"query": "..."}'`).

**Call 1 — triggers retrieval** (`policy_question`, keyword: "delivery"):

Request:
```json
{"query": "What is Zepto's delivery fee for small orders?"}
```

Response:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01_chunk_0", "doc_05_chunk_0", "doc_08_chunk_0"],
  "confidence": 1.0
}
```
The top retrieved/used chunk is `doc_01_chunk_0` (the Delivery Policy doc),
which does match the question asked, and no LLM/network call was made.

**Call 2 — does not trigger retrieval** (`general_question`, no policy
keyword present):

Request:
```json
{"query": "Who won the cricket world cup in 2011?"}
```

Response:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

Additional keyword-routing spot checks (`classify_intent`, mock mode, no
LLM call in either case):
- `"How much does membership cost?"` → `intent = "policy_question"` (keyword: "membership")
- `"Tell me a joke."` → `intent = "general_question"`

## 6. Docker (Task 6 — required, graded baseline)

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
# then: curl -X POST http://localhost:7860/ask -d '{"query": "How do gift cards work?"}'
```

The container runs with `MOCK_LLM` at its default (mock mode) unless you
override it. This local build-and-run is the required, graded baseline —
no push to a registry or hosting platform is required.

*(Optional, ungraded stretch, not attempted in this submission: deploy the
same `Dockerfile` to Hugging Face Spaces' free community CPU tier, storing
`GROQ_API_KEY` as a Space secret rather than committing it.)*

## 7. Optional `MOCK_LLM=0` extension (not required for grading)

Not exercised for this graded submission (grading runs with `MOCK_LLM` at
its default). To try it:

```bash
export MOCK_LLM=0
export GROQ_API_KEY=your_free_tier_key   # console.groq.com, no card required
uvicorn main:app --host 0.0.0.0 --port 7860
```

This switches `classify_intent`, `retrieve_and_answer`, and `direct_answer`
to their real-LLM branches (`graph_app._llm_classify_intent`,
`_llm_generate_grounded_answer`, `_llm_generate_direct_answer`), using the
Task 2 prompt template and the schema-validation retry loop
(`_generate_with_retries`, up to 2 retries with `SCHEMA_RETRY_SUFFIX`).

## Repository layout

```
support_assistant/
├── docs/doc_01.txt ... doc_08.txt   # Task: corpus (verbatim as specified)
├── ingest.py                        # Task 1: load, chunk, embed, store
├── prompts.py                       # Task 2: structured prompt template
├── graph_app.py                     # Task 3+4: LangGraph graph + schema
├── schemas.py                       # Task 4: Pydantic AnswerSchema/API models
├── main.py                          # Task 5: FastAPI app (POST /ask)
├── requirements.txt
├── Dockerfile                       # Task 6
└── README.md                        # this file (Task 7)
```
