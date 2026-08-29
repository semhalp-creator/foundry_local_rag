# Demo Day — Presentation Notes

This mirrors the final slide deck (5 slides, ~2 minutes) built for the Week 6
milestone. It's a "what I learned building this" talk, first-person and
honest about the starting point, not a feature pitch.

## 1. Cover — a Q&A assistant that works offline

**"An offline question-answering assistant."** RAG, entirely on-device:
an embedding model + SQLite + a local ~3.8B-parameter chat model.
Stack: Python 3.13, `foundry-local-sdk`, `sqlite3`, Flask.

*I didn't know what RAG (Retrieval-Augmented Generation) meant when I
started this six-week project. This talk is a short walk through what I
built and what I learned along the way.*

## 2. I started by learning what RAG actually is

RAG answers from *our* data, not the model's memory:

| Step | What happens | Code |
|---|---|---|
| 01. Chunk | Split on paragraph boundaries | `chunk_text()` |
| 02. Embed | Turn each chunk into a vector | `generate_embeddings()` |
| 03. Retrieve | Cosine similarity finds the closest top-K | `get_top_chunks()` |
| 04. Generate | The model writes using only that context | `answer_query()` |

The problem RAG solves: ask a general model something about a narrow topic
and it either doesn't know, or it hallucinates — makes up something
confident-sounding instead. RAG fixes that in four steps, and the answer
comes from my data, not the model's training set.

## 3. I split the project into four layers

| Layer | What it does | Tech |
|---|---|---|
| Client | CLI + web UI — both call the same core | `app.py` · `web_app.py` |
| Server / pipeline | Embeds the query, builds context, writes the prompt | `answer_query()` |
| Data | `documents(id, content, source, embedding)` — 11 rows | `rag.db` · `sqlite3` |
| AI | `qwen3-embedding-0.6b` + `phi-3.5-mini` (~3.8B) | Foundry Local · ONNX |

No HTTP requests at inference time — models are downloaded once and cached
locally. The client layer is where I first saw how a backend and a frontend
actually talk to each other (Flask serving HTML, JS calling it with
`fetch`). The data layer is where I learned real SQL: creating a table,
inserting rows, querying them.

## 4. I learned cosine similarity has a use in retrieval

```
cos(θ) = a · b / (‖a‖ × ‖b‖)
```

I already knew this formula from math. What I didn't know is that it can
measure how similar two pieces of *text* are: divide the dot product of two
vectors by the product of their lengths, the length cancels out, and what's
left is the angle — texts that mean similar things point in a similar
direction.

```python
# retrieval.py
def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sqrt(sum(x*x for x in a))
    nb = sqrt(sum(x*x for x in b))
    return dot / (na*nb) if na and nb else 0.0

scored.sort(key=lambda x: x[2], reverse=True)
return scored[:top_k]
```

This is brute force — every query pulls all 11 rows into memory and scores
them all, O(n). Fine at this scale; a knowledge base with tens of thousands
of chunks would need something like FAISS instead.

## 5. How I built it, stage by stage

1. **Chunked and embedded the documents** — split on paragraph boundaries,
   turned each chunk into a vector, wrote it to SQLite.
2. **Built retrieval** — embed the query too, find the closest chunks by
   cosine similarity.
3. **Wired up the model** — system prompt: "use only the given context;
   say you don't know if it's not enough."
4. **Added source citations** — prompt instruction: "name the source you
   used, never invent one."
5. **Added a relevance threshold** — weak chunks reaching the model caused
   hallucination; anything below 0.55 now never gets sent at all.
6. **Wrote the test suite** — 8 scenarios: answerable, unanswerable, and
   general questions, all automated.

I didn't know how to do any of these six steps when I started. What I
actually walked away with is two habits: trace a bug to the layer it
really comes from before touching the fix, and pick a number by measuring
it, not by guessing.

## Live demo

Run `python3 app.py` (or `web_app.py` for the browser version) and ask:

1. **An answerable question** — e.g. *"What programming languages does the
   SDK support?"* — a grounded answer with a source citation.
2. **An out-of-scope question** — e.g. *"What is the capital of France?"*
   — the assistant honestly declines instead of guessing.
3. *(Optional)* — a broad question like *"Tell me about Foundry Local"* to
   show it can synthesize across multiple retrieved chunks.
