"""Week 5 stretch goal: Option C from the plan — a static HTML+JS page
served by a small local Flask backend, calling the same answer_query()
pipeline from app.py. No changes to the CLI (app.py) or the RAG logic
itself — this is purely an alternate interface on top of it.
"""

from flask import Flask, jsonify, render_template, request
from foundry_local_sdk import Configuration, FoundryLocalManager

from app import CHAT_MODEL_ID, EMBEDDING_MODEL_ID, answer_query, format_source_line

app = Flask(__name__)

# Populated once by load_models(), before the server starts accepting
# requests — the models are expensive to load, so we do it once, not
# per-request.
embedding_client = None
chat_client = None
embedding_model = None
chat_model = None


def load_models():
    global embedding_client, chat_client, embedding_model, chat_model

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ID)
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model(CHAT_MODEL_ID)
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("Models loaded. Web server ready.")


@app.route("/")
def index():
    """Serve the static HTML+JS page (templates/index.html)."""
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """The endpoint the page's JavaScript calls for each question."""
    data = request.get_json(silent=True) or {}
    query = (data.get("question") or "").strip()
    if not query:
        return jsonify({"error": "Please enter a question."}), 400

    answer, chunks = answer_query(query, embedding_client, chat_client, verbose=False)
    return jsonify(
        {
            "answer": answer,
            # Built by the same helper the CLI uses, so both interfaces show
            # an identically worded citation (or none at all) for the same
            # answer - the page just renders this string as-is.
            "source_line": format_source_line(chunks),
            "chunks": [
                {"content": content, "source": source, "score": round(score, 3)}
                for content, source, score in chunks
            ],
        }
    )


def unload_models():
    if embedding_model is not None:
        embedding_model.unload()
    if chat_model is not None:
        chat_model.unload()


if __name__ == "__main__":
    load_models()
    try:
        app.run(debug=False, port=5000)
    finally:
        unload_models()
