"""Pydantic models used across the service.

- AskRequest / AskResponse: the FastAPI request/response contract for POST /ask.
- AnswerSchema: the JSON output schema enforced on the final answer (Task 4).
  In mock mode this is populated deterministically by our own code. In the
  optional MOCK_LLM=0 extension, this is the schema the raw LLM output must
  validate against (with retries on failure).
"""
from typing import List
from pydantic import BaseModel, Field


class AnswerSchema(BaseModel):
    """The enforced structured-output schema for the final answer."""
    answer: str = Field(..., description="The natural-language answer to the user's query.")
    sources: List[str] = Field(
        default_factory=list,
        description="Chunk/document IDs used to ground the answer. Empty for general_question answers.",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1."
    )


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's natural-language question.")


class AskResponse(AnswerSchema):
    """The response returned by POST /ask. Identical shape to AnswerSchema,
    kept as a distinct name so the API contract is explicit and can diverge
    later (e.g. adding request metadata) without touching AnswerSchema."""
    pass
