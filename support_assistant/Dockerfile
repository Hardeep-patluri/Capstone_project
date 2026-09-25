FROM python:3.11-slim

WORKDIR /app

# System deps needed by chromadb / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the ChromaDB index at image build time so the first request
# isn't slow (also downloads/caches the sentence-transformers model layer).
# Safe to skip/fail silently if there is no network at build time; ingest
# will lazily build the collection on first request instead.
RUN python ingest.py || true

# MOCK_LLM defaults to unset -> deterministic offline mock mode (graded
# baseline). Set MOCK_LLM=0 and GROQ_API_KEY at `docker run` time (or as a
# platform "secret", e.g. a Hugging Face Space secret) to opt into the
# optional real-LLM extension. Never hardcode API keys in this file.
ENV MOCK_LLM=1
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
