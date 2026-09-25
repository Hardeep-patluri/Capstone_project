"""Task 3: LangGraph StateGraph orchestration.

State: a TypedDict carrying the query through the graph plus whatever each
node fills in along the way.

Nodes (>= 3, as required):
  - classify_intent      -> sets state["intent"] to "policy_question" or
                             "general_question"
  - retrieve_and_answer   -> retrieves top-3 chunks + produces the answer for
                             policy_question queries
  - direct_answer         -> produces the answer for general_question queries
                             (no retrieval)

A conditional edge out of classify_intent routes to retrieve_and_answer or
direct_answer based on state["intent"], mirroring a graph-based intent
router. That routing logic never depends on MOCK_LLM - only the
generation step *inside* retrieve_and_answer / direct_answer branches on it.
"""
import os
import json
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END

from ingest import retrieve_top_k
from prompts import build_rag_prompt, SCHEMA_RETRY_SUFFIX
from schemas import AnswerSchema

# --------------------------------------------------------------------------
# MOCK_LLM toggle (see module README for the full explanation).
# Unset, or "1" -> deterministic offline mock (graded baseline, default).
# "0"           -> optional real-LLM extension (Groq or any free-tier API).
# --------------------------------------------------------------------------
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def mock_llm_enabled() -> bool:
    """Returns True only when MOCK_LLM=0 is explicitly set (real-LLM mode)."""
    return os.environ.get("MOCK_LLM", "1") == "0"


class GraphState(TypedDict, total=False):
    query: str
    intent: str  # "policy_question" | "general_question"
    retrieved_chunks: List[dict]
    answer: Optional[AnswerSchema]


# --------------------------------------------------------------------------
# Node 1: classify_intent
# --------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if mock_llm_enabled():
        # Optional MOCK_LLM=0 extension: call the LLM to classify instead.
        intent = _llm_classify_intent(query)
    else:
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"

    return {**state, "intent": intent}


def _route_from_intent(state: GraphState) -> str:
    """Conditional-edge selector function (not itself MOCK_LLM-dependent)."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# --------------------------------------------------------------------------
# Node 2: retrieve_and_answer (policy_question path)
# --------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real in both modes: local embedding model +
    # ChromaDB need no API key and no network call to any LLM provider.
    retrieved = retrieve_top_k(query, k=3)

    if mock_llm_enabled():
        answer_schema = _llm_generate_grounded_answer(query, retrieved)
    else:
        # Mock mode (graded baseline): canned templated answer, no LLM call.
        top_chunk = retrieved[0]
        top_chunk_snippet = top_chunk["text"][:200]
        answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
        answer_schema = AnswerSchema(
            answer=answer_text,
            sources=[r["id"] for r in retrieved],
            confidence=1.0,
        )

    return {**state, "retrieved_chunks": retrieved, "answer": answer_schema}


# --------------------------------------------------------------------------
# Node 3: direct_answer (general_question path, no retrieval)
# --------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if mock_llm_enabled():
        answer_schema = _llm_generate_direct_answer(query)
    else:
        # Mock mode (graded baseline): fixed canned string, no LLM call.
        answer_schema = AnswerSchema(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )

    return {**state, "retrieved_chunks": [], "answer": answer_schema}


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_from_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AnswerSchema:
    graph = get_graph()
    final_state = graph.invoke({"query": query})
    return final_state["answer"]


# --------------------------------------------------------------------------
# Optional MOCK_LLM=0 extension: real LLM calls (Groq free tier by default,
# any OpenAI-compatible free-tier endpoint works too). Only imported/used
# when MOCK_LLM=0 is explicitly set, so this never runs during grading.
# --------------------------------------------------------------------------
def _get_llm_client():
    """Returns a Groq client configured from env vars. Import is local so
    the `groq` package is only required for the optional extension."""
    from groq import Groq  # pip install groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires GROQ_API_KEY to be set (console.groq.com free tier)."
        )
    return Groq(api_key=api_key)


LLM_MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")


def _call_llm_raw(prompt: str) -> str:
    client = _get_llm_client()
    completion = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content


def _parse_and_validate(raw_text: str) -> AnswerSchema:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    data = json.loads(cleaned)
    return AnswerSchema(**data)


def _generate_with_retries(base_prompt: str, max_retries: int = 2) -> AnswerSchema:
    """Task 4: retry up to 2 additional times with a corrective instruction
    before giving up and returning a clearly marked error response."""
    prompt = base_prompt
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = _call_llm_raw(prompt)
            return _parse_and_validate(raw)
        except Exception as exc:  # JSON decode error or Pydantic ValidationError
            last_error = exc
            prompt = base_prompt + SCHEMA_RETRY_SUFFIX

    return AnswerSchema(
        answer=f"[ERROR] LLM output failed schema validation after {max_retries + 1} attempts: {last_error}",
        sources=[],
        confidence=0.0,
    )


def _llm_classify_intent(query: str) -> str:
    classify_prompt = (
        "Classify the following customer question as exactly one word: "
        "'policy_question' if it asks about Zepto's delivery, returns, "
        "refunds, membership, order tracking, cancellation, gift cards, or "
        "support hours; otherwise 'general_question'. "
        f"Question: \"{query}\"\nRespond with only the single classification word."
    )
    raw = _call_llm_raw(classify_prompt).strip().lower()
    return "policy_question" if "policy_question" in raw else "general_question"


def _llm_generate_grounded_answer(query: str, retrieved: list) -> AnswerSchema:
    prompt = build_rag_prompt(query, retrieved)
    return _generate_with_retries(prompt)


def _llm_generate_direct_answer(query: str) -> AnswerSchema:
    prompt = (
        "You are Zepto's customer support assistant. Answer the following "
        "general question concisely and respond with ONLY a JSON object "
        '{"answer": "...", "sources": [], "confidence": <0-1 float>}.\n'
        f"Question: {query}"
    )
    return _generate_with_retries(prompt)
