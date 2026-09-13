"""
Inference script for Gemini 3 (Google Generative AI API).

Setup:
    pip install google-genai
    export GOOGLE_API_KEY="..."

Usage:
    python gemini3_inference.py --prompts ../prompts.txt --out ../generated_samples --model gemini-3-pro

Notes:
    - Verify the exact model string (e.g. "gemini-3-pro", "gemini-3-flash")
      against Google AI Studio / Vertex AI docs at time of running.
    - Source / license: https://ai.google.dev/gemini-api/terms
      (Gemini models are proprietary/closed-weight; usage governed by
      Google's API Terms of Service, not an open-source license).
"""

import argparse
import os
import re
import time
from pathlib import Path

from google import genai


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
    parser.add_argument("--model", type=str, default="gemini-3-pro")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts.")

    for i, prompt in enumerate(prompts, start=1):
        sample_id = f"prompt_{i:02d}_{slugify(prompt)}"
        out_path = args.out / f"{sample_id}__gemini3.txt"
        if out_path.exists():
            print(f"[skip] {out_path.name} already exists")
            continue

        try:
            response = client.models.generate_content(
                model=args.model,
                contents=prompt,
                config={"temperature": args.temperature},
            )
            text = response.text
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
