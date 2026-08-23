import json
import sqlite3

DB_PATH = "practice.db"


def create_table(conn):
    """Create the documents table if it doesn't already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )
    conn.commit()


def insert_sample_rows(conn):
    """Insert a few sample documents with fake (tiny) embeddings."""
    # Real embeddings from Foundry Local would have hundreds of numbers;
    # here we use tiny 3-number vectors just to practice the mechanics.
    samples = [
        ("Foundry Local runs AI models directly on your device.", [0.9, 0.1, 0.0]),
        ("SQLite is a lightweight, serverless local database.", [0.1, 0.9, 0.0]),
        ("Cosine similarity measures the angle between two vectors.", [0.0, 0.2, 0.9]),
    ]

    for content, embedding in samples:
        conn.execute(
            "INSERT INTO documents (content, embedding) VALUES (?, ?)",
            (content, json.dumps(embedding)),  # embedding stored as JSON text
        )
    conn.commit()


def query_by_id(conn, doc_id):
    """Fetch a single row by its id."""
    cursor = conn.execute(
        "SELECT id, content, embedding FROM documents WHERE id = ?", (doc_id,)
    )
    return cursor.fetchone()


def query_by_keyword(conn, keyword):
    """Fetch rows whose content contains the given keyword."""
    cursor = conn.execute(
        "SELECT id, content, embedding FROM documents WHERE content LIKE ?",
        (f"%{keyword}%",),
    )
    return cursor.fetchall()


def main():
    conn = sqlite3.connect(DB_PATH)

    create_table(conn)
    insert_sample_rows(conn)

    print("--- Query by id=2 ---")
    row = query_by_id(conn, 2)
    if row:
        doc_id, content, embedding_json = row
        embedding = json.loads(embedding_json)  # back to a Python list
        print(f"id={doc_id}\ncontent={content}\nembedding={embedding}")

    print("\n--- Query by keyword='SQLite' ---")
    rows = query_by_keyword(conn, "SQLite")
    for doc_id, content, embedding_json in rows:
        embedding = json.loads(embedding_json)
        print(f"id={doc_id} | content={content} | embedding={embedding}")

    conn.close()


if __name__ == "__main__":
    main()
