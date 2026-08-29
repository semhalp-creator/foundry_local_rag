# Week 5 Test Report — foundry_local_rag

Chat model: `phi-3.5-mini` | Embedding model: `qwen3-embedding-0.6b`

**Result: 8/8 test cases passed. Average response time: 2.17s.**

## Functional test cases

| # | Kind | Query | Result | Time (s) | Chunks | Answer (truncated) |
|---|------|-------|--------|----------|--------|---------------------|
| 1 | answerable | What programming languages does the SDK support? | ✅ PASS | 2.06 | 2 | According to Foundry Local FAQ, the Foundry Local SDK supports Python, C#, JavaScript, and Rust. |
| 2 | answerable | How does Foundry Local run models? | ✅ PASS | 3.76 | 2 | According to Foundry Local FAQ, Foundry Local runs AI models directly on your device without the nee... |
| 3 | answerable | What is retrieval-augmented generation? | ✅ PASS | 3.47 | 2 | According to Foundry Local FAQ, retrieval-augmented generation grounds model responses in your own d... |
| 4 | answerable | What hardware acceleration does Foundry Local use? | ✅ PASS | 2.83 | 2 | According to the Foundry Local FAQ, Foundry Local uses ONNX Runtime for efficient model inference, w... |
| 5 | unanswerable | What is the capital of France? | ✅ PASS | 0.05 | 0 | I don't have information about that in my knowledge base. |
| 6 | unanswerable | What's the weather like today? | ✅ PASS | 0.04 | 0 | I don't have information about that in my knowledge base. |
| 7 | unanswerable | bu gece hava kaç derece | ✅ PASS | 0.04 | 0 | I don't have information about that in my knowledge base. |
| 8 | general | Tell me about Foundry Local. | ✅ PASS | 5.11 | 2 | Foundry Local is a platform that allows AI models to operate directly on your device without the nee... |

## Edge cases

- **Empty query input**: The CLI loop (app.py main()) now reprompts on a blank Enter press ('Please enter a question...') instead of exiting. Only typing 'quit' ends the session. answer_query() is never called with an empty string. — *Fixed (Week 5): previously a blank Enter silently exited the app, same as 'quit' - now it reprompts instead.*
- **Very general question ("Tell me about Foundry Local.")**: Covered as a normal test case above. — *See test case result.*

## Performance & debugging

- Response times ranged 0.04s–5.11s, averaging 2.17s — within the plan's ~1-3s target for small models on a laptop.
- Embeddings are not recomputed on every question: document chunks are embedded once in `ingest.py` and cached in `rag.db`; only the user's query is embedded per turn, which is unavoidable and cheap.
- Formatting check (balanced parentheses, no doubled spaces) found no issues in any answer.
- Retrieval check: every answerable test case passed its keyword check, consistent with (though not proof of) the retriever surfacing the right chunk each time; no incorrect-retrieval symptoms observed in this run.

## Evaluation & improvement (self-critique)

- **Accurate?** 8/8 test cases passed (correct answer when info was present, honest fallback when it wasn't).
- **Well-written and concise?** Average answer length is 24 words (range 10-57); no answers ran noticeably long.
- **Are sources cited?** 5/5 answers that had retrieved context named a real source from it. Citations are written by the model itself (prompt-based), then verified in the test against the chunks actually retrieved — a name that isn't in the context counts as fabricated, not cited.
  - 3 answer(s) had no retrieved context at all (the MIN_RELEVANT_SCORE fallback fired), so there was correctly nothing to cite.

## Shortcomings / follow-ups identified

- No failing test cases in this run.
- Empty-query handling: fixed in Week 5 — the CLI now reprompts on a blank Enter press instead of exiting; only 'quit' ends the session.
