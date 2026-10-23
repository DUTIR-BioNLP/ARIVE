"""Verify the Qwen3-VL token ids assumed by the ARIVE rollout implementation."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer


EXPECTED = {
    "<|endoftext|>": 151643,
    "<|im_end|>": 151645,
    "<|box_end|>": 151649,
    "<think>": 151667,
    "</think>": 151668,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qwen3-VL model/checkpoint path or model id.")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=True)
    mismatches = []
    for token, expected_id in EXPECTED.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: expected={expected_id}, actual={actual_id}")
        if actual_id != expected_id:
            mismatches.append((token, expected_id, actual_id))
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    print(f"assistant: expected=[77091], actual={assistant_ids}")
    if assistant_ids != [77091]:
        mismatches.append(("assistant", 77091, assistant_ids))
    if mismatches:
        raise SystemExit(
            "Token ids do not match this implementation. Do not train until the "
            "constants in train_grpo_verl.py are reviewed for this tokenizer."
        )
    print("ARIVE token-id check passed.")


if __name__ == "__main__":
    main()

