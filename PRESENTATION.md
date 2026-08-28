# Demo Day — Presentation Notes

Outline for the final presentation, following the plan's four talking
points. Aim for ~5 minutes: problem, features, live demo, lessons learned.

## 1. Problem statement

*What need does this assistant target?*

General-purpose chatbots either don't know about a narrow/private topic, or
they'll confidently make something up rather than admit they don't know.
This assistant answers questions **only** from a fixed local knowledge base
(currently: Foundry Local facts), grounding every answer in retrieved text —
and it runs **entirely offline**, so it works without an internet connection
or sending any data to a cloud service.

## 2. Key features / components

- **Retrieval-Augmented Generation (RAG)**: questions are answered using
  content retrieved from a knowledge base, not the model's own training
  data — reduces hallucination, keeps answers grounded and up to date with
  *our* documents.
- **Fully local**: embedding model (`qwen3-embedding-0.6b`) and chat model
  (`phi-3.5-mini`) both run on-device via Foundry Local. No API keys, no
  network calls at inference time.
- **Persistent knowledge base**: documents are chunked, embedded once, and
  stored in a SQLite database (`rag.db`) — not recomputed on every run.
- **Source citations**: answers name which document they drew on, so a user
  can verify the claim rather than trusting it blindly.
- **Two interfaces**: a CLI (`app.py`) and a minimal web UI (`web_app.py` +
  Flask), both backed by the exact same `answer_query()` logic.
- **Tested**: an automated suite (`test_suite.py`) checks answerable,
  unanswerable, and general questions, plus response time, on every change.

## 3. Live demo script

Run `python3 app.py` (or `web_app.py` for the browser version) and ask, in
order:

1. **An answerable question** — e.g. *"What programming languages does the
   SDK support?"* — shows a correct, grounded answer with a source citation.
2. **An out-of-scope question** — e.g. *"What is the capital of France?"* —
   shows the assistant honestly declining instead of guessing. This is the
   moment that proves it's not just calling a general chatbot API under the
   hood.
3. *(Optional, if time)* — a general/broad question like *"Tell me about
   Foundry Local"* to show it can synthesize across multiple retrieved
   chunks, not just quote one sentence.

## 4. Lessons learned

Fill this in with what actually surprised you while building — a few
honest starting points from this project to adapt or replace:

- **Retrieval quality matters more than prompt wording.** A real off-topic,
  non-English test question ("what's the temperature tonight," asked in
  Turkish) made the model produce garbled, hallucinated text instead of
  declining — because the only chunks retrieved were weak matches, and our
  English "say you don't know" instruction didn't transfer reliably to a
  non-English response. The fix wasn't a better prompt; it was adding a
  similarity-score cutoff so weak matches never reach the model at all.
  Good retrieval is the first line of defense against hallucination, not
  the prompt.
- **Prompt instructions aren't guarantees.** Even after telling the model
  explicitly not to add a citation on "I don't know" answers, it sometimes
  did anyway. Small/local models don't follow every instruction reliably —
  the fix was to double-check the important rules in code, not just trust
  the prompt.
- **The order of operations matters in streaming code.** An early version
  checked `chunk.choices[0]` before checking whether `chunk.choices` was
  empty, which crashed on some stream chunks — a reminder that streaming
  APIs don't guarantee every chunk looks like the "normal" one.
- **Chunking is easy to skip — and worth verifying, not assuming.** It was
  tempting to just embed whole documents. After splitting the longer document
  into paragraph-sized chunks, we checked that a chunk from the *middle* of
  it is genuinely reachable: asking "How does the SDK manage the model
  lifecycle?" returns Overview part 2 as the top hit at 0.785, ahead of every
  short FAQ line. (We never ran a before/after precision comparison, so
  "chunking made retrieval better" stays an untested claim — what we can say
  is that mid-document content is retrievable, which is the thing chunking
  was supposed to buy us.)

*(Swap in your own — these are seeded from what actually happened in this
project's build log, not hypothetical.)*

## Optional: naming / customizing

The plan suggests naming the assistant and/or customizing the interface to
build presentation confidence. The web UI (`templates/index.html`) is
intentionally basic right now — a good place to add a name/logo before demo
day if there's time, without touching any of the RAG logic.
