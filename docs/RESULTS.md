# RAG Document QA / Reading the results

[← Project overview](../README.md) · [Saved CSV](../test_results.csv)

The committed artifact contains 100 rows: 93 labeled `CORRECT`, seven labeled `INCORRECT`. This count was checked during the documentation redesign on 2026-09-19. No paid evaluation was rerun.

## Inspect a row

1. Read `question` and `actual_answer`.
2. Inspect `found_context`: is the required fact present, and is unrelated neighboring text present?
3. Compare `generated_answer` with the source and reference.
4. Read `semantic_score`, `llm_grade`, and `llm_reason` as two imperfect evaluation signals.

`test_number` is zero-based. The saved failures are **16, 46, 47, 56, 84, 89, and 98**.

## A useful disagreement

Test **84** has similarity **0.9910669326782227** and an `INCORRECT` grade. It is evidence that an embedding similarity score can coexist with a factual-error judgment. Neither a high similarity nor the grader's explanation alone establishes truth.

## Reproducibility boundary

The source specifies the first 100 rows of the Ko-LongRAG test parquet, `gpt-4.1-mini` for answers and grading, normalized E5 embeddings, 400-character chunks, 80-character overlap, and three retrieved chunks plus neighbors.

The CSV does not bind the dataset, dependency environment, prompts, and remote model behavior with a complete versioned run manifest. Treat the existing source as the current procedure and the CSV as a saved artifact; do not infer bit-for-bit reproducibility or that every current source detail is proven to match the historical run.

## Questions the artifact cannot answer by itself

- Whether the top retrieved passages contain the necessary evidence across the full benchmark.
- How another independent grader or human reviewer would score the answers.
- Whether the chosen settings generalize to another slice or domain.
- How much quality changes when neighbor expansion crosses an article boundary.
- How much time or cost the pipeline uses in a controlled benchmark.

No latency, cost, or retrieval-recall result is claimed. The README's counts are descriptive, not a leaderboard submission.
