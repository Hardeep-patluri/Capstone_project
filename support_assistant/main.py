"""Task 5: FastAPI wrapper around the LangGraph pipeline.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 7860 --reload

Then:
    curl -X POST http://localhost:7860/ask \\
         -H "Content-Type: application/json" \\
         -d '{"query": "How much does delivery cost?"}'
"""
from fastapi import FastAPI, HTTPException

from schemas import AskRequest, AskResponse
from graph_app import run_query, mock_llm_enabled

app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-backed support assistant over Zepto's own policy corpus.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": mock_llm_enabled()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        answer_schema = run_query(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return AskResponse(**answer_schema.model_dump())
