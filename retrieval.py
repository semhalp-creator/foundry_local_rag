import json
import math
import sqlite3
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

# Anchored to this file's directory, not the current working directory.
# With a bare "rag.db", running the app from anywhere else would silently
# create an empty database there and then fail with "no such table:
# documents" - confusing, and it litters stray files around the filesystem.
DB_PATH = str(Path(__file__).resolve().parent / "rag.db")


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    main.py has its own copy of this. That duplication is deliberate:
    main.py is the Week 1-2 teaching file, meant to be read and tinkered
    with, and it should stay runnable on its own. Importing this five-line
    function from there would have made every experiment in a learning file
    able to break the real app.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class KnowledgeBaseMissing(RuntimeError):
    """Raised when rag.db hasn't been built yet, or holds no documents."""


def ensure_knowledge_base():
    """Fail fast, with a clear message, if ingest.py hasn't been run yet.

    rag.db is generated and gitignored, so a fresh clone starts with no
    database at all. Without this check the first question dies on a raw
    "sqlite3.OperationalError: no such table: documents" - and only after
    the user has already waited ~15s for the models to load. Callers run
    this *before* loading anything so the failure is instant and says what
    to actually do about it.
    """
    hint = (
        f"Knowledge base not found at {DB_PATH}.\n"
        "Build it first with:  python3 ingest.py"
    )
    if not Path(DB_PATH).exists():
        raise KnowledgeBaseMissing(hint)

    conn = sqlite3.connect(DB_PATH)
    try:
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    except sqlite3.OperationalError as exc:
        raise KnowledgeBaseMissing(hint) from exc
    finally:
        conn.close()

    if count == 0:
        raise KnowledgeBaseMissing(
            f"Knowledge base at {DB_PATH} is empty.\n"
            "Rebuild it with:  python3 ingest.py"
        )


def load_documents(conn):
    """Load every (id, content, source, embedding) row from SQLite."""
    cursor = conn.execute("SELECT id, content, source, embedding FROM documents")
    rows = []
    for doc_id, content, source, embedding_json in cursor.fetchall():
        rows.append((doc_id, content, source, json.loads(embedding_json)))
    return rows


def get_top_chunks(query, embedding_client, top_k=2):
    """Return the top_k most relevant (content, source, score) tuples for a query.

    This is the SQLite-backed version of main.py's find_relevant():
    instead of comparing against an in-memory list, it pulls every row
    (and its embedding) back from the database, then does the same
    brute-force cosine similarity comparison in Python. Fine for a
    handful of documents; a real vector DB would be needed at scale.
    """
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    conn = sqlite3.connect(DB_PATH)
    rows = load_documents(conn)
    conn.close()

    scored = [
        (content, source, cosine_similarity(query_embedding, embedding))
        for _, content, source, embedding in rows
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]


def main():
    """Manual test: load the embedding model and try a few sample queries."""
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    test_queries = [
        "What programming languages does the SDK support?",
        "How does Foundry Local run models?",
        "What is retrieval-augmented generation?",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}")
        for content, source, score in get_top_chunks(query, embedding_client, top_k=2):
            print(f"  [{score:.3f}] ({source}) {content}")

    embedding_model.unload()


if __name__ == "__main__":
    main()
