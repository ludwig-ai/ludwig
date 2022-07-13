import torch
from torchtext.transforms import BERTTokenizer
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import PRETRAINED_VOCAB_FILES_MAP, PRETRAINED_INIT_CONFIGURATION
from transformers.utils.hub import cached_path

model_names = {
    "bert-base-german-cased",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "TurkuNLP/bert-base-finnish-cased-v1",
}

inputs = "Hello, I'm a single sentence!"

for model_name in model_names:
    print(model_name)

    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # When using transformers.utils.hub.cached_path,
    # there is an OBO error if there is an empty line in the vocab file.

    print("Using cached_path function...")

    torch_tokenizer = BERTTokenizer(
        cached_path(PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_name]),
        return_tokens=True,
        **PRETRAINED_INIT_CONFIGURATION[model_name]
    )
    print("\tExpected:\t", hf_tokenizer.tokenize(inputs))
    print("\tActual:\t\t", torch_tokenizer(inputs))

    torch_tokenizer_ids_only = BERTTokenizer(
        cached_path(PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_name]),
        return_tokens=False,
        **PRETRAINED_INIT_CONFIGURATION[model_name]
    )

    hf_tokens = hf_tokenizer.encode(inputs, truncation=True)[1:-1]  # remove start and stop tokens
    hf_tokens = [str(hf_token) for hf_token in hf_tokens]  # convert to string for parity with torchtext
    print("\tExpected:\t", hf_tokens)
    print("\tActual:\t\t", torch_tokenizer_ids_only(inputs))

    # There seems to be some larger difference when loading the vocab files directly.

    print("Loading vocab files with torchtext directly...")

    torch_tokenizer = BERTTokenizer(
        PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_name],
        return_tokens=True,
        **PRETRAINED_INIT_CONFIGURATION[model_name]
    )
    print("\tExpected:\t", hf_tokenizer.tokenize(inputs))
    print("\tActual:\t\t", torch_tokenizer(inputs))

    torch_tokenizer_ids_only = BERTTokenizer(
        PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_name],
        return_tokens=False,
        **PRETRAINED_INIT_CONFIGURATION[model_name]
    )

    hf_tokens = hf_tokenizer.encode(inputs, truncation=True)[1:-1]  # remove start and stop tokens
    hf_tokens = [str(hf_token) for hf_token in hf_tokens]  # convert to string for parity with torchtext
    print("\tExpected:\t", hf_tokens)
    print("\tActual:\t\t", torch_tokenizer_ids_only(inputs))
