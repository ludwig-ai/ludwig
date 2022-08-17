import json

from torchtext.transforms import BERTTokenizer
from transformers import AutoTokenizer
from transformers.utils.hub import cached_path

hf_name = "bert-base-uncased"
hf_tokenizer = AutoTokenizer.from_pretrained(hf_name)

vocab_file = cached_path(f"https://huggingface.co/{hf_name}/resolve/main/vocab.txt")
hf_config_file = cached_path(f"https://huggingface.co/{hf_name}/resolve/main/tokenizer_config.json")
with open(hf_config_file, "r") as f:
    hf_config = json.load(f)

# Extract kwargs for TorchText tokenizer from tokenizer config
tokenizer_kwargs = {}
if "do_lower_case" in hf_config:
    tokenizer_kwargs["do_lower_case"] = hf_config["do_lower_case"]
if "strip_accents" in hf_config:
    tokenizer_kwargs["strip_accents"] = hf_config["strip_accents"]

# Sample input. HF tokenizer never splits special tokens
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py#L244
sample = " \tHeLLo!how  \n Are yoU? [UNK]"
print(f"Sample: {repr(sample)}\n")

# Prints tokens
tt_tokenizer = BERTTokenizer(
    vocab_path=vocab_file,
    **tokenizer_kwargs,
    return_tokens=True,
)
print("Tokens")
print("\tHF (expected):\t", hf_tokenizer.tokenize(sample))
print("\tTT (actual):\t", tt_tokenizer(sample))

# Prints token IDs
tt_tokenizer = BERTTokenizer(
    vocab_path=vocab_file,
    **tokenizer_kwargs,
)
print("Token IDs")
print("\tHF (expected):\t", hf_tokenizer.encode(sample))
tt_output = tt_tokenizer(sample)
tt_token_ids = [int(idx) for idx in tt_output]
print("\tTT (actual):\t", tt_token_ids)

# Requested interface: never_split exposed at the constructor level
"""
tt_tokenizer = BERTTokenizer(
    vocab_path=vocab_file,
    **tokenizer_kwargs,
    return_tokens=True,
    never_split=[
        hf_tokenizer.unk_token,
        hf_tokenizer.pad_token,
        hf_tokenizer.cls_token,
        hf_tokenizer.sep_token,
        ...
    ]
)
"""
