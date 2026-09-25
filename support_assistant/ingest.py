"""Task 1: Ingestion + embedding.

Loads the 8 Zepto policy documents from docs/, chunks each one (each short
policy doc is used as a single chunk, since every doc is already only a
few sentences describing one topic - a fixed-size sub-split would just cut
a single policy statement in half for no benefit here), embeds each chunk
locally with sentence-transformers' all-MiniLM-L6-v2 (no API key, no
network call to any LLM provider), and stores the resulting vectors in a
persistent ChromaDB collection.

This module is imported by graph_app.py, which calls get_collection() once
at startup and reuses the same in-memory/persisted collection for every
retrieval in the retrieve_and_answer node.
"""
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embedder = None


def get_embedder() -> SentenceTransformer:
    """Lazily load the local embedding model (downloaded once, cached by
    sentence-transformers/huggingface locally; no API key required)."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def load_and_chunk_documents() -> list[dict]:
    """Loads every docs/doc_*.txt file and turns it into one chunk.

    Returns a list of {"id": chunk_id, "doc_id": doc_id, "text": chunk_text}.
    """
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # Each doc is a single short policy paragraph -> one chunk per doc.
        chunk_id = f"{doc_id}_chunk_0"
        chunks.append({"id": chunk_id, "doc_id": doc_id, "text": text})
    return chunks


def build_collection(reset: bool = False) -> chromadb.api.models.Collection.Collection:
    """Embeds all chunks and (re)populates the ChromaDB collection.

    Uses a persistent client so the index survives across process restarts
    (docs/ rarely change, but rebuilding is cheap and idempotent either way).
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # If already populated (e.g. persisted from a previous run), reuse it.
    if collection.count() > 0 and not reset:
        return collection

    chunks = load_and_chunk_documents()
    embedder = get_embedder()
    embeddings = embedder.encode([c["text"] for c in chunks]).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[{"doc_id": c["doc_id"]} for c in chunks],
    )
    return collection


_collection = None


def get_collection():
    """Singleton accessor used by graph_app.py's retrieve_and_answer node."""
    global _collection
    if _collection is None:
        _collection = build_collection()
    return _collection


def retrieve_top_k(query: str, k: int = 3) -> list[dict]:
    """Embeds `query` and retrieves the top-k most similar chunks via
    cosine similarity from the ChromaDB collection. Runs for real in both
    MOCK_LLM modes - embedding and ChromaDB need no API key/network call.
    """
    collection = get_collection()
    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(query_embeddings=query_embedding, n_results=k)

    retrieved = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results.get("distances", [[None] * len(ids)])[0]
    for _id, _text, _dist in zip(ids, docs, distances):
        retrieved.append({"id": _id, "text": _text, "distance": _dist})
    return retrieved


if __name__ == "__main__":
    # Quick manual sanity check: `python ingest.py`
    col = build_collection(reset=True)
    print(f"Collection '{COLLECTION_NAME}' now has {col.count()} chunks.")
    for r in retrieve_top_k("How much is delivery?", k=3):
        print(r["id"], "->", r["text"][:80])
