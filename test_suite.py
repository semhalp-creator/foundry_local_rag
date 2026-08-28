"""Week 5: Functional testing & evaluation for the RAG assistant.

Runs a fixed set of test queries (answerable, unanswerable, and general)
against app.py's answer_query(), checks whether each response looks
correct, measures response time, and writes a documented report to
test_results.md — the plan's "Milestone by mid Week 5" deliverable.
"""

import time
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

from app import CHAT_MODEL_ID, EMBEDDING_MODEL_ID, answer_query, cited_sources

# Anchored to this file's directory so the report always lands next to the
# code, not wherever the suite happened to be run from.
REPORT_PATH = Path(__file__).resolve().parent / "test_results.md"

# A mix of queries the knowledge base *can* answer, queries it *can't*
# (should trigger the "I don't know" fallback), and one general/broad
# question. This mirrors the plan's "compile a small set of Q&As (some
# answerable, some unanswerable)" testing approach.
TEST_CASES = [
    {
        "query": "What programming languages does the SDK support?",
        "kind": "answerable",
        "expect_keywords": ["python", "c#", "javascript", "rust"],
    },
    {
        "query": "How does Foundry Local run models?",
        "kind": "answerable",
        "expect_keywords": ["device", "local", "cloud"],
    },
    {
        "query": "What is retrieval-augmented generation?",
        "kind": "answerable",
        "expect_keywords": ["retriev", "context", "data"],
    },
    {
        "query": "What hardware acceleration does Foundry Local use?",
        "kind": "answerable",
        "expect_keywords": ["cpu", "gpu", "npu"],
    },
    {
        "query": "What is the capital of France?",
        "kind": "unanswerable",
    },
    {
        "query": "What's the weather like today?",
        "kind": "unanswerable",
    },
    {
        # Regression test: a real user tried this and the model produced
        # garbled, hallucinated Turkish text instead of declining, because
        # the retrieved chunks were weak matches and our "say you don't
        # know" instruction was written in English. Fixed by adding a
        # MIN_RELEVANT_SCORE cutoff in app.py that skips the model call
        # entirely below a similarity threshold - so this should now return
        # the same fixed NO_MATCH_ANSWER regardless of query language.
        "query": "bu gece hava kaç derece",
        "kind": "unanswerable",
    },
    {
        "query": "Tell me about Foundry Local.",
        "kind": "general",
        "expect_keywords": ["device", "local", "model"],
    },
]

# Phrases we'd expect somewhere in a well-behaved "I don't know" answer.
UNKNOWN_PHRASES = [
    "don't have",
    "doesn't contain",
    "does not contain",
    "not enough information",
    "don't know",
    "no information",
    "cannot answer",
    "can't answer",
    "can't provide",
    "cannot provide",
    "unable to provide",
    "not able to provide",
]


def check_answerable(answer, expect_keywords):
    lower = answer.lower()
    return any(keyword in lower for keyword in expect_keywords)


def check_unanswerable(answer):
    lower = answer.lower()
    return any(phrase in lower for phrase in UNKNOWN_PHRASES)


def check_citation(answer, chunks):
    """Did the model cite a source, and is the one it named actually real?

    The model writes its own citation from the source names in its context.
    That's the plan's prompt-based approach, but it means the citation is a
    *claim* - so this checks it against the chunks that were really
    retrieved. Returns (cited_a_real_source, looks_fabricated).
    """
    if not chunks:
        # Nothing was retrieved (the MIN_RELEVANT_SCORE fallback fired), so
        # there is correctly nothing to cite.
        return False, False
    real = cited_sources(answer, chunks)
    if real:
        return True, False
    # No real source name appears. If the answer nonetheless talks about a
    # source, the model likely invented one - worth flagging in the report.
    mentions_source = any(
        word in answer.lower() for word in ("source:", "according to", "per the")
    )
    return False, mentions_source


def run_tests(embedding_client, chat_client):
    results = []
    for case in TEST_CASES:
        query = case["query"]
        start = time.time()
        answer, chunks = answer_query(query, embedding_client, chat_client, verbose=False)
        elapsed = time.time() - start
        answer = answer.strip()

        if case["kind"] == "unanswerable":
            passed = check_unanswerable(answer)
        else:
            # Both "answerable" and "general" need a real, grounded answer.
            # A bare `len(answer) > 0` check would have passed even when the
            # MIN_RELEVANT_SCORE fallback fired and no model call happened at
            # all, so the general case gets keyword-checked too.
            passed = check_answerable(answer, case["expect_keywords"])

        cited_source, fabricated = check_citation(answer, chunks)
        results.append(
            {
                "query": query,
                "kind": case["kind"],
                "answer": answer,
                "elapsed": elapsed,
                "passed": passed,
                "num_chunks": len(chunks),
                "word_count": len(answer.split()),
                "cited_source": cited_source,
                "fabricated_citation": fabricated,
            }
        )
    return results


def check_edge_cases():
    """Edge cases from the plan: empty input, and other boundary behavior.

    Empty input isn't run through answer_query() directly - it's handled
    one level up, in app.py's CLI loop (main()), before answer_query() is
    ever called. Documenting the loop's behavior here, as the plan asks.
    """
    return [
        {
            "case": "Empty query input",
            "observed_behavior": (
                "The CLI loop (app.py main()) now reprompts on a blank Enter "
                "press ('Please enter a question...') instead of exiting. "
                "Only typing 'quit' ends the session. answer_query() is "
                "never called with an empty string."
            ),
            "verdict": "Fixed (Week 5): previously a blank Enter silently "
            "exited the app, same as 'quit' - now it reprompts instead.",
        },
        {
            "case": "Very general question (\"Tell me about Foundry Local.\")",
            "observed_behavior": "Covered as a normal test case above.",
            "verdict": "See test case result.",
        },
    ]


def print_and_save_report(results, edge_cases):
    lines = []
    lines.append("# Week 5 Test Report — foundry_local_rag")
    lines.append("")
    lines.append(f"Chat model: `{CHAT_MODEL_ID}` | Embedding model: `{EMBEDDING_MODEL_ID}`")
    lines.append("")

    passed_count = sum(1 for r in results if r["passed"])
    avg_time = sum(r["elapsed"] for r in results) / len(results)

    lines.append(f"**Result: {passed_count}/{len(results)} test cases passed. "
                  f"Average response time: {avg_time:.2f}s.**")
    lines.append("")
    lines.append("## Functional test cases")
    lines.append("")
    lines.append("| # | Kind | Query | Result | Time (s) | Chunks | Answer (truncated) |")
    lines.append("|---|------|-------|--------|----------|--------|---------------------|")

    for i, r in enumerate(results, start=1):
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        answer_preview = r["answer"].replace("\n", " ")[:100]
        if len(r["answer"]) > 100:
            answer_preview += "..."
        lines.append(
            f"| {i} | {r['kind']} | {r['query']} | {status} | {r['elapsed']:.2f} "
            f"| {r['num_chunks']} | {answer_preview} |"
        )

    lines.append("")
    lines.append("## Edge cases")
    lines.append("")
    for e in edge_cases:
        lines.append(f"- **{e['case']}**: {e['observed_behavior']} — *{e['verdict']}*")

    # --- Performance & Debugging (plan: response times, caching, formatting) ---
    lines.append("")
    lines.append("## Performance & debugging")
    lines.append("")
    slowest = max(results, key=lambda r: r["elapsed"])
    fastest = min(results, key=lambda r: r["elapsed"])
    lines.append(
        f"- Response times ranged {fastest['elapsed']:.2f}s–{slowest['elapsed']:.2f}s, "
        f"averaging {avg_time:.2f}s — "
        + ("within" if avg_time <= 3.0 else "above")
        + " the plan's ~1-3s target for small models on a laptop."
    )
    lines.append(
        "- Embeddings are not recomputed on every question: document chunks "
        "are embedded once in `ingest.py` and cached in `rag.db`; only the "
        "user's query is embedded per turn, which is unavoidable and cheap."
    )
    malformed = [
        r for r in results
        if r["answer"].count("(") != r["answer"].count(")") or "  " in r["answer"]
    ]
    if malformed:
        lines.append(
            f"- Formatting check found possible issues in {len(malformed)} answer(s) "
            "(unbalanced parentheses or doubled spaces) — see: "
            + ", ".join(f'"{r["query"]}"' for r in malformed)
        )
    else:
        lines.append(
            "- Formatting check (balanced parentheses, no doubled spaces) found "
            "no issues in any answer."
        )
    wrong_retrieval = [r for r in results if r["kind"] == "answerable" and not r["passed"]]
    if wrong_retrieval:
        lines.append(
            f"- Retrieval check: {len(wrong_retrieval)} answerable case(s) failed, "
            "which can indicate the wrong chunk was retrieved rather than a model "
            "error — worth inspecting the printed 'Retrieved context' for those queries."
        )
    else:
        lines.append(
            "- Retrieval check: every answerable test case passed its keyword check, "
            "consistent with (though not proof of) the retriever surfacing the right "
            "chunk each time; no incorrect-retrieval symptoms observed in this run."
        )

    # --- Evaluation & Improvement (plan: accuracy, conciseness, citations) ---
    lines.append("")
    lines.append("## Evaluation & improvement (self-critique)")
    lines.append("")
    cited_count = sum(1 for r in results if r["cited_source"])
    avg_words = sum(r["word_count"] for r in results) / len(results)
    lines.append(f"- **Accurate?** {passed_count}/{len(results)} test cases passed "
                  "(correct answer when info was present, honest fallback when it wasn't).")
    lines.append(f"- **Well-written and concise?** Average answer length is "
                  f"{avg_words:.0f} words (range "
                  f"{min(r['word_count'] for r in results)}-"
                  f"{max(r['word_count'] for r in results)}); "
                  + ("no answers ran noticeably long." if max(r["word_count"] for r in results) < 80
                     else "at least one answer is longer than ideal for a quick Q&A — "
                          "could tighten the system prompt further (e.g. 'answer in 1-2 sentences')."))
    grounded = [r for r in results if r["num_chunks"] > 0]
    fabricated = [r for r in results if r["fabricated_citation"]]
    lines.append(
        f"- **Are sources cited?** {cited_count}/{len(grounded)} answers that had "
        "retrieved context named a real source from it. Citations are written by "
        "the model itself (prompt-based), then verified in the test against the "
        "chunks actually retrieved — a name that isn't in the context counts as "
        "fabricated, not cited."
    )
    if fabricated:
        lines.append(
            f"  - ⚠️ {len(fabricated)} answer(s) referred to a source that was not in "
            "the retrieved context: "
            + ", ".join(f'"{r["query"]}"' for r in fabricated)
            + ". The prompt tells the model never to invent a source name; when that "
            "isn't enough, the fallback is post-processing the citation in code."
        )
    elif cited_count < len(grounded):
        lines.append(
            "  - The uncited answers didn't name a source but didn't invent one "
            "either — they simply answered without attribution."
        )
    if len(grounded) < len(results):
        lines.append(
            f"  - {len(results) - len(grounded)} answer(s) had no retrieved context at "
            "all (the MIN_RELEVANT_SCORE fallback fired), so there was correctly "
            "nothing to cite."
        )

    lines.append("")
    lines.append("## Shortcomings / follow-ups identified")
    lines.append("")
    failed = [r for r in results if not r["passed"]]
    if failed:
        for r in failed:
            lines.append(f"- \"{r['query']}\" did not meet the pass criteria — review prompt/retrieval for this case.")
    else:
        lines.append("- No failing test cases in this run.")
    if avg_time > 3.0:
        lines.append(
            f"- Average response time ({avg_time:.2f}s) is above the plan's "
            "~1-3s target for small models — consider a smaller chat model, "
            "fewer retrieved chunks (top_k), or checking hardware acceleration."
        )
    lines.append(
        "- Empty-query handling: fixed in Week 5 — the CLI now reprompts on "
        "a blank Enter press instead of exiting; only 'quit' ends the session."
    )

    report = "\n".join(lines)
    print("\n" + report)

    REPORT_PATH.write_text(report + "\n")
    print(f"\n\nSaved to {REPORT_PATH.name}")


def main():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL_ID)
    embedding_model.download(lambda p: None)
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    chat_model = manager.catalog.get_model(CHAT_MODEL_ID)
    chat_model.download(lambda p: None)
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print(f"Running {len(TEST_CASES)} test cases...")
    results = run_tests(embedding_client, chat_client)
    edge_cases = check_edge_cases()
    print_and_save_report(results, edge_cases)

    embedding_model.unload()
    chat_model.unload()


if __name__ == "__main__":
    main()
