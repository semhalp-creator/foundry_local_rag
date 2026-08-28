import json
import sqlite3
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

# Anchored to this file's directory so ingest.py always writes to the same
# database retrieval.py reads from, no matter where it's run from.
DB_PATH = str(Path(__file__).resolve().parent / "rag.db")

# Week 3: same knowledge base as main.py's in-memory list, but this time
# each entry gets embedded once and persisted to SQLite instead of being
# recomputed in memory every time the program runs.
documents = [
    "Foundry Local runs AI models directly on your device without cloud connectivity.",
    "The Foundry Local SDK supports Python, C#, JavaScript, and Rust.",
    "Embedding models convert text into numerical vectors for similarity search.",
    "Foundry Local uses ONNX Runtime for efficient model inference on CPUs and GPUs.",
    "The model catalog provides pre-optimized models that you can download and run locally.",
    "Retrieval-augmented generation grounds model responses in your own data.",
    "Vector similarity search finds documents that are semantically close to a query.",
    "Chat completions generate natural language responses from a prompt and context.",
]

# A longer, multi-paragraph "real" document, to demonstrate chunking. In an
# actual project this would be a full article, FAQ page, or course notes
# file loaded from disk instead of a hardcoded string.
FOUNDRY_LOCAL_OVERVIEW = """
Foundry Local is Microsoft's end-to-end local AI solution for running large
language models entirely on a user's device. It ships as a lightweight
runtime plus an SDK, so applications can call into on-device models without
needing a cloud account, an internet connection, or a dedicated GPU.

The SDK manages the full model lifecycle: discovering models in a curated
catalog, downloading them on demand, loading them into memory, and unloading
them when they are no longer needed. Foundry Local automatically picks the
best available hardware acceleration, whether that is a CPU, GPU, or NPU.

Because everything runs on-device, applications built on Foundry Local can
operate with zero network calls at inference time. This makes it a natural
fit for privacy-sensitive use cases, offline environments, or any scenario
where sending data to a cloud API isn't acceptable.
"""


def chunk_text(text, max_paragraphs=1):
    """Split a document into chunks of up to `max_paragraphs` paragraphs.

    RAG typically retrieves at passage-level, not whole-document level, so
    long documents are broken into smaller chunks before embedding. Here we
    split on blank lines (paragraph breaks) and group every `max_paragraphs`
    paragraphs into one chunk.
    """
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk = " ".join(paragraphs[i : i + max_paragraphs])
        chunks.append(chunk)
    return chunks


def create_table(conn):
    """(Re)create the documents table with a fresh schema.

    Dropping and recreating keeps ingest.py idempotent even when the
    schema itself changes (e.g. adding the `source` column below) —
    simpler than writing a migration for a learning project like this.
    """
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()


def ingest():
    # Initialize the SDK and load the embedding model (same as main.py)
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

    # Chunk the longer document into passage-sized pieces, then combine
    # with the short, already-atomic sentences from `documents`. Each chunk
    # is paired with a short source name so answers can later cite where
    # the information came from.
    chunked = chunk_text(FOUNDRY_LOCAL_OVERVIEW)
    all_chunks = (
        [(content, "Foundry Local FAQ") for content in documents]
        + [
            (content, f"Foundry Local Overview (part {i + 1})")
            for i, content in enumerate(chunked)
        ]
    )
    contents = [content for content, _source in all_chunks]

    # Embed every chunk in one batch call
    response = embedding_client.generate_embeddings(contents)
    embeddings = [item.embedding for item in response.data]

    # Persist to SQLite
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)
    for (content, source), embedding in zip(all_chunks, embeddings):
        conn.execute(
            "INSERT INTO documents (content, source, embedding) VALUES (?, ?, ?)",
            (content, source, json.dumps(embedding)),
        )
    conn.commit()

    # Test: verify the DB actually has the expected number of entries
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    expected = len(all_chunks)
    conn.close()

    embedding_model.unload()
    if count == expected:
        print(f"Ingested and verified {count} chunks in {DB_PATH}.")
    else:
        print(f"WARNING: expected {expected} rows but found {count} in {DB_PATH}.")


if __name__ == "__main__":
    ingest()
