"""
Inference script for GPT-5 (OpenAI API).

Setup:
    pip install openai
    export OPENAI_API_KEY="sk-..."

Usage:
    python gpt5_inference.py --prompts ../prompts.txt --out ../generated_samples --model gpt-5

Notes:
    - Verify the exact model string (e.g. "gpt-5", "gpt-5-turbo", "gpt-5.1")
      against your OpenAI dashboard / API docs at time of running, since
      exact naming and availability can change.
    - Source / license: https://openai.com/policies/usage-policies
      (API outputs are licensed to the caller per OpenAI's terms; OpenAI's
      models themselves are proprietary/closed-weight, not open-licensed).
"""

import argparse
import os
import re
import time
from pathlib import Path

from openai import OpenAI


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def slugify(text: str, maxlen: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:maxlen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("../prompts.txt"))
    parser.add_argument("--out", type=Path, default=Path("../generated_samples"))
    parser.add_argument("--model", type=str, default="gpt-5")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts.")

    for i, prompt in enumerate(prompts, start=1):
        sample_id = f"prompt_{i:02d}_{slugify(prompt)}"
        out_path = args.out / f"{sample_id}__gpt5.txt"
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue

        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            text = response.choices[0].message.content
        except Exception as e:
            text = f"[ERROR generating output: {e}]"

        out_path.write_text(
            f"PROMPT:\n{prompt}\n\nMODEL: {args.model}\n\nOUTPUT:\n{text}\n",
            encoding="utf-8",
        )
        print(f"[{i}/{len(prompts)}] saved -> {out_path.name}")
        time.sleep(1)  # be polite to rate limits


if __name__ == "__main__":
    main()
