# Evaluation Summary

> Fill this in after running `inference/run_all.py` (and optionally
> `inference/compute_perplexity.py`). Keep the qualitative summary at or
> under 300 words total.

## Quantitative signal (optional)
- Metric used: perplexity under GPT-2 (see `compute_perplexity.py`) — used
  instead of BLEU because these are open-ended prompts with no reference
  outputs to score against.
- Mean perplexity, GPT-5 samples: _fill in from perplexity_results.csv_
- Mean perplexity, Gemini 3 samples: _fill in from perplexity_results.csv_
- Caveat: perplexity under an external scorer reflects surface fluency
  only — not factual accuracy, reasoning correctness, or creativity.

## Qualitative summary (≤ 300 words)

Cover, per model, comparing across the five prompt categories:

1. **Coherence** — Did outputs stay on-topic and internally consistent
   across the full response, or drift/contradict themselves partway
   through longer generations (especially reasoning and creative prompts)?
2. **Instruction-following** — Were explicit constraints honored (word
   counts, format requirements like dialogue-only, "explain step by
   step", etc.)?
3. **Failure modes observed** — e.g., reasoning prompts: arithmetic
   slips or skipped steps; creative prompts: generic/cliché phrasing;
   factual prompts: confident but unverified claims (hallucination
   risk) — spot-check these against a reliable source; code prompts:
   syntactically valid but logically wrong edge-case handling; dialogue
   prompts: tone drift from the requested persona.
4. **Head-to-head differences** — Where did the two models diverge
   most? Which handled multi-step reasoning more reliably? Which
   produced more natural creative prose?

_Draft your ≤300-word narrative here once you have the actual outputs —
a genuine evaluation needs to be grounded in what the models actually
produced, not written in advance._
