# Text Generation Evaluation — GPT-5 & Gemini 3

Runs 20 diverse prompts (reasoning, creative writing, factual/knowledge,
code generation, dialogue) through GPT-5 (OpenAI API) and Gemini 3
(Google API), saves the outputs, and provides an evaluation template.

## Models used

| Model | Access | Source / License |
|---|---|---|
| GPT-5 | Hosted API (OpenAI) | https://openai.com/policies/usage-policies — proprietary, closed-weight; outputs licensed to the API caller under OpenAI's Terms of Use. |
| Gemini 3 | Hosted API (Google) | https://ai.google.dev/gemini-api/terms — proprietary, closed-weight; usage governed by Google's Gemini API Terms of Service. |

Confirm the exact model string available to your account (e.g.
`gpt-5` vs. a dated/tier variant, `gemini-3-pro` vs. `gemini-3-flash`)
in each provider's current docs before running — hosted model names
and availability can change.

## Repo structure

```
text-gen-eval/
├── README.md
├── requirements.txt
├── prompts.txt                  # 20 prompts, grouped by category
├── evaluation_summary.md        # template for the ≤300-word write-up
├── inference/
│   ├── gpt5_inference.py        # calls OpenAI API for all prompts
│   ├── gemini3_inference.py     # calls Google Gemini API for all prompts
│   ├── run_all.py               # runs both scripts back to back
│   └── compute_perplexity.py    # optional: perplexity via local GPT-2 scorer
└── generated_samples/           # output .txt files land here
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

## Running

```bash
cd inference
python run_all.py
```

This produces two files per prompt in `generated_samples/`:
`prompt_NN_<slug>__gpt5.txt` and `prompt_NN_<slug>__gemini3.txt`, each
containing the original prompt, model name, and generated output.

To run a single model, or change temperature/model string:

```bash
python gpt5_inference.py --model gpt-5 --temperature 0.7
python gemini3_inference.py --model gemini-3-pro --temperature 0.7
```

## Evaluation

BLEU isn't meaningful here since these are open-ended prompts with no
reference outputs to compare against. Instead:

- **Quantitative (optional):** `compute_perplexity.py` scores each
  output's fluency under an independent GPT-2 model — a rough proxy
  signal only, not a quality judgment.
- **Qualitative (required):** fill in `evaluation_summary.md` — a
  ≤300-word summary of coherence and failure modes across the two
  models, after reading the actual generated samples.

## Status

Scaffold and scripts are complete and ready to run. `generated_samples/`
is currently empty (needs live API keys to populate) — the evaluation
summary should only be written after real outputs exist.
