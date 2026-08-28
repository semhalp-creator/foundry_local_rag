# Week 5 Test Report — foundry_local_rag

Chat model: `phi-3.5-mini` | Embedding model: `qwen3-embedding-0.6b`

**Result: 8/8 test cases passed. Average response time: 1.70s.**

## Functional test cases

| # | Kind | Query | Result | Time (s) | Chunks | Answer (truncated) |
|---|------|-------|--------|----------|--------|---------------------|
| 1 | answerable | What programming languages does the SDK support? | ✅ PASS | 1.17 | 2 | The Foundry Local SDK supports Python, C#, JavaScript, and Rust programming languages. |
| 2 | answerable | How does Foundry Local run models? | ✅ PASS | 2.44 | 2 | Foundry Local runs AI models directly on your device without the need for cloud connectivity. It ach... |
| 3 | answerable | What is retrieval-augmented generation? | ✅ PASS | 2.60 | 2 | Retrieval-augmented generation is a method used in natural language processing where chat completion... |
| 4 | answerable | What hardware acceleration does Foundry Local use? | ✅ PASS | 1.84 | 2 | Foundry Local uses hardware acceleration through ONNX Runtime for efficient model inference on both ... |
| 5 | unanswerable | What is the capital of France? | ✅ PASS | 0.04 | 0 | I don't have information about that in my knowledge base. |
| 6 | unanswerable | What's the weather like today? | ✅ PASS | 0.03 | 0 | I don't have information about that in my knowledge base. |
| 7 | unanswerable | bu gece hava kaç derece | ✅ PASS | 0.03 | 0 | I don't have information about that in my knowledge base. |
| 8 | general | Tell me about Foundry Local. | ✅ PASS | 5.49 | 2 | Foundry Local is a platform that allows you to run Artificial Intelligence (AI) models directly on y... |

## Edge cases

- **Empty query input**: The CLI loop (app.py main()) now reprompts on a blank Enter press ('Please enter a question...') instead of exiting. Only typing 'quit' ends the session. answer_query() is never called with an empty string. — *Fixed (Week 5): previously a blank Enter silently exited the app, same as 'quit' - now it reprompts instead.*
- **Very general question ("Tell me about Foundry Local.")**: Covered as a normal test case above. — *See test case result.*

## Performance & debugging

- Response times ranged 0.03s–5.49s, averaging 1.70s — within the plan's ~1-3s target for small models on a laptop.
- Embeddings are not recomputed on every question: document chunks are embedded once in `ingest.py` and cached in `rag.db`; only the user's query is embedded per turn, which is unavoidable and cheap.
- Formatting check (balanced parentheses, no doubled spaces) found no issues in any answer.
- Retrieval check: every answerable test case passed its keyword check, consistent with (though not proof of) the retriever surfacing the right chunk each time; no incorrect-retrieval symptoms observed in this run.

## Evaluation & improvement (self-critique)

- **Accurate?** 8/8 test cases passed (correct answer when info was present, honest fallback when it wasn't).
- **Well-written and concise?** Average answer length is 42 words (range 10-147); at least one answer is longer than ideal for a quick Q&A — could tighten the system prompt further (e.g. 'answer in 1-2 sentences').
- **Are sources cited?** 5/8 answers carry a source line, built in code by `format_source_line()` from the chunks actually retrieved — the model is explicitly told not to name sources itself, so this can't drift from what was really used.
  - Refinement idea: some answers (e.g. the 'I don't know' cases) correctly have nothing to cite, so a citation gap there is expected, not a defect — worth confirming case-by-case rather than assuming it's a bug.

## Shortcomings / follow-ups identified

- No failing test cases in this run.
- Empty-query handling: fixed in Week 5 — the CLI now reprompts on a blank Enter press instead of exiting; only 'quit' ends the session.
