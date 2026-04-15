"""Prepare the Anthropic HH-RLHF dataset for Ludwig alignment training.

Downloads `Anthropic/hh-rlhf` from HuggingFace and converts it into the
column format expected by Ludwig's DPO, KTO, and ORPO trainers.

DPO / ORPO output:  train.csv, test.csv
  Columns: prompt, chosen, rejected

KTO output:         train_kto.csv, test_kto.csv
  Columns: prompt, response, label  (label is True for chosen, False for rejected)

The HH-RLHF dataset stores full multi-turn conversations as raw text with the
pattern:
    "\n\nHuman: <turn>\n\nAssistant: <turn>\n\nHuman: ...\n\nAssistant: ..."

We extract the last Human turn as the prompt and the final Assistant turn as
the response.  For DPO we do this for both `chosen` and `rejected` columns.
"""

import argparse
import re

import pandas as pd
from datasets import load_dataset


def extract_last_human_turn(conversation: str) -> str:
    """Return the last Human turn from a raw HH-RLHF conversation string."""
    human_turns = re.findall(r"\n\nHuman: (.*?)(?=\n\nAssistant:|\Z)", conversation, re.DOTALL)
    if human_turns:
        return human_turns[-1].strip()
    return conversation.strip()


def extract_last_assistant_turn(conversation: str) -> str:
    """Return the last Assistant turn from a raw HH-RLHF conversation string."""
    assistant_turns = re.findall(r"\n\nAssistant: (.*?)(?=\n\nHuman:|\Z)", conversation, re.DOTALL)
    if assistant_turns:
        return assistant_turns[-1].strip()
    return ""


def convert_split(split_data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert one HF dataset split into DPO and KTO DataFrames."""
    rows_dpo = []
    rows_kto = []

    for example in split_data:
        chosen_conv = example["chosen"]
        rejected_conv = example["rejected"]

        prompt = extract_last_human_turn(chosen_conv)
        chosen_response = extract_last_assistant_turn(chosen_conv)
        rejected_response = extract_last_assistant_turn(rejected_conv)

        if not prompt or not chosen_response or not rejected_response:
            continue

        rows_dpo.append(
            {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
            }
        )

        # KTO: expand each pair into two rows with a boolean label
        rows_kto.append({"prompt": prompt, "response": chosen_response, "label": True})
        rows_kto.append({"prompt": prompt, "response": rejected_response, "label": False})

    return pd.DataFrame(rows_dpo), pd.DataFrame(rows_kto)


def main():
    parser = argparse.ArgumentParser(description="Prepare Anthropic HH-RLHF for Ludwig alignment training.")
    parser.add_argument("--output_dir", default=".", help="Directory to write CSV files into.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap the number of training examples (useful for quick experiments).",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Cap the number of test examples.",
    )
    args = parser.parse_args()

    print("Downloading Anthropic/hh-rlhf …")
    dataset = load_dataset("Anthropic/hh-rlhf")

    train_split = dataset["train"]
    test_split = dataset["test"]

    if args.max_train_samples:
        train_split = train_split.select(range(min(args.max_train_samples, len(train_split))))
    if args.max_test_samples:
        test_split = test_split.select(range(min(args.max_test_samples, len(test_split))))

    print(f"Converting {len(train_split)} train examples …")
    train_dpo, train_kto = convert_split(train_split)

    print(f"Converting {len(test_split)} test examples …")
    test_dpo, test_kto = convert_split(test_split)

    import os

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    train_kto_path = os.path.join(args.output_dir, "train_kto.csv")
    test_kto_path = os.path.join(args.output_dir, "test_kto.csv")

    train_dpo.to_csv(train_path, index=False)
    test_dpo.to_csv(test_path, index=False)
    train_kto.to_csv(train_kto_path, index=False)
    test_kto.to_csv(test_kto_path, index=False)

    print(f"\nDPO dataset:  {len(train_dpo)} train rows -> {train_path}")
    print(f"              {len(test_dpo)} test rows  -> {test_path}")
    print(f"KTO dataset:  {len(train_kto)} train rows -> {train_kto_path}")
    print(f"              {len(test_kto)} test rows  -> {test_kto_path}")
    print("\nColumns in DPO files:  prompt, chosen, rejected")
    print("Columns in KTO files:  prompt, response, label")
    print("\nDone.")


if __name__ == "__main__":
    main()
