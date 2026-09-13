"""
Optional quantitative metric: perplexity of each generated sample under an
independent open-source scorer model (GPT-2 by default).

Why perplexity instead of BLEU: BLEU requires reference translations/outputs
to compare against, which doesn't apply to open-ended prompts (there's no
single "correct" story or explanation). Perplexity under a fixed external
model is a standard proxy for fluency that doesn't need references. Treat
it as a rough fluency signal, not a quality score -- it says nothing about
factual correctness, reasoning validity, or creativity.

Setup:
    pip install transformers torch

Usage:
    python compute_perplexity.py --samples ../generated_samples --out ../perplexity_results.csv
"""

import argparse
import csv
import math
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def extract_output(text: str) -> str:
    marker = "OUTPUT:\n"
    idx = text.find(marker)
    return text[idx + len(marker):].strip() if idx != -1 else text.strip()


def compute_perplexity(text: str, model, tokenizer, device) -> float:
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)
    if input_ids.size(1) < 2:
        return float("nan")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("../generated_samples"))
    parser.add_argument("--out", type=Path, default=Path("../perplexity_results.csv"))
    parser.add_argument("--scorer-model", type=str, default="gpt2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(args.scorer_model)
    model = GPT2LMHeadModel.from_pretrained(args.scorer_model).to(device)
    model.eval()

    rows = []
    files = sorted(args.samples.glob("*.txt"))
    print(f"Scoring {len(files)} files with {args.scorer_model} on {device}...")

    for f in files:
        raw = f.read_text(encoding="utf-8")
        output_text = extract_output(raw)
        ppl = compute_perplexity(output_text, model, tokenizer, device)
        rows.append({"file": f.name, "perplexity": round(ppl, 2) if ppl == ppl else "nan"})
        print(f"  {f.name}: {ppl:.2f}" if ppl == ppl else f"  {f.name}: nan")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "perplexity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
