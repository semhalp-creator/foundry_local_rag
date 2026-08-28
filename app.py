from foundry_local_sdk import Configuration, FoundryLocalManager

from retrieval import (  # Week 3's SQLite-backed retrieval
    KnowledgeBaseMissing,
    ensure_knowledge_base,
    get_top_chunks,
)


def format_source_line(results):
    """Build a single deterministic 'Source: ...' line from retrieved chunks.

    We used to have the *model* write its own citation into the answer
    text. That turned out to be unreliable in two ways: it sometimes added
    a citation to an "I don't know" answer despite being told not to, and
    it duplicated the source info we can already derive ourselves from
    `results` - callers ended up showing both a model-written "Source:"
    line and our own metadata-driven one side by side. This function is
    the single source of truth instead: given the same `results` that
    were actually retrieved and sent to the model, both the CLI (below)
    and the web UI (via the `chunks` field in web_app.py's JSON response)
    can build an identical, always-accurate citation from it - no need to
    trust anything the model says about its own sources.
    """
    if not results:
        return None
    sources = list(dict.fromkeys(source for _content, source, _score in results))
    return "Source: " + ", ".join(sources)


# Minimum cosine similarity a retrieved chunk needs before we trust it
# enough to hand to the chat model. Below this, the "best" chunk is really
# just the least-bad match, not relevant content - sending it to the model
# anyway invites hallucination (observed concretely with off-topic/
# non-English questions, where the model sometimes fails to say "I don't
# know" cleanly). Chosen from this project's own test data: answerable
# questions scored 0.65+, unanswerable ones scored under 0.5.
MIN_RELEVANT_SCORE = 0.55

NO_MATCH_ANSWER = "I don't have information about that in my knowledge base."

EMBEDDING_MODEL_ID = "qwen3-embedding-0.6b"

# Week 4: upgraded from qwen2.5-0.5b (0.5B params) to Phi-3.5 Mini (~3.8B),
# matching the plan's "Phi-3.5 Mini or similar 3-5B parameter model"
# recommendation for better answer quality, while still being small/fast
# enough to run comfortably on a laptop.
CHAT_MODEL_ID = "phi-3.5-mini"


def answer_query(query, embedding_client, chat_client, top_k=2, verbose=True):
    """Retrieve context for `query`, then ask the chat model to answer.

    This is the Week 4 exercise: wire Week 3's get_top_chunks() (SQLite
    retrieval) into the chat model, using the same grounded system-prompt
    pattern from main.py.

    Returns (answer_text, retrieved_chunks) so callers (like Week 5's test
    harness, or web_app.py's JSON response) can inspect the result
    programmatically and build their own source citation from
    `retrieved_chunks` via format_source_line() - the model's answer text
    itself never includes source names. `verbose=True` (the CLI's default)
    prints the retrieved context, the answer, and that same deterministic
    source line, buffered rather than streamed token-by-token so the
    printed output always matches the returned value.
    """
    results = get_top_chunks(query, embedding_client, top_k=top_k)

    # Log the retrieved chunks so we can verify retrieval is actually
    # happening and see *why* the model answered the way it did.
    if verbose:
        print("Retrieved context:")
        for content, source, score in results:
            print(f"  [{score:.3f}] ({source}) {content}")

    # If even the best-matching chunk is a weak match, don't hand it to the
    # chat model at all - a low-relevance chunk plus a genuinely off-topic
    # question is exactly when small models are most likely to hallucinate
    # instead of cleanly declining (reproduced with a completely off-topic
    # question whose best retrieved chunk scored below MIN_RELEVANT_SCORE).
    # Skipping the model call here is also strictly faster.
    best_score = results[0][2] if results else 0.0
    if best_score < MIN_RELEVANT_SCORE:
        if verbose:
            print(f"Answer: {NO_MATCH_ANSWER}\n")
        # Return no chunks here, not `results` - those chunks were rejected
        # for being too weak, not used to produce the answer. Returning them
        # anyway would make callers (like web_app.py) display a misleading
        # "Sources: ..." line under an "I don't know" answer.
        return NO_MATCH_ANSWER, []

    context = "\n".join(f"- ({source}) {content}" for content, source, _score in results)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using only the provided context below. "
                "Be polite and concise. Do not mention or name your sources - "
                "that's added separately, outside your answer. "
                "If the context doesn't contain enough information, say so honestly "
                "and stop there — do not add information from outside the context, "
                "even as a suggestion or aside.\n\n"
                f"Context:\n{context}"
            ),
        },
        {"role": "user", "content": query},
    ]

    # Collect the full answer *before* printing anything, rather than
    # streaming each chunk live to the console - we want the printed
    # output to match the returned value exactly, and (now) to be followed
    # by our own deterministic source line, not whatever the model wrote.
    answer_parts = []
    for chunk in chat_client.complete_streaming_chat(messages):
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            answer_parts.append(content)

    full_answer = "".join(answer_parts).strip()

    if verbose:
        source_line = format_source_line(results)
        print(f"Answer: {full_answer}")
        if source_line:
            print(source_line)
        print()

    return full_answer, results


def main():
    # Check this before loading any models - they take ~15s, and there's no
    # point paying that just to fail on the first question.
    try:
        ensure_knowledge_base()
    except KnowledgeBaseMissing as exc:
        print(exc)
        return

    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Load the embedding model (needed by get_top_chunks to embed queries)
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ID)
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Load the (upgraded) chat model
    chat_model = manager.catalog.get_model(CHAT_MODEL_ID)
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nModels loaded. Ready for questions.")
    print('Type "quit" to exit.\n')

    # Option A from the plan: simple CLI loop.
    # Edge case handling (Week 5): only "quit" exits the loop now. A blank
    # Enter press just reprompts instead of silently ending the session.
    while True:
        query = input("Question: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            print("Please enter a question (or type 'quit' to exit).\n")
            continue
        answer_query(query, embedding_client, chat_client)

    embedding_model.unload()
    chat_model.unload()
    print("Models unloaded. Done!")


if __name__ == "__main__":
    main()
