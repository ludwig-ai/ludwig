from ludwig.utils.tokenizers import (EnglishLemmatizeFilterTokenizer,
                                     NgramTokenizer, StringSplitTokenizer,
                                     get_tokenizer_from_registry)


def test_ngram_tokenizer():
    inputs = "Hello, I'm a single sentence!"
    tokenizer = NgramTokenizer(n=2)
    tokens_expected = [
        "Hello,",
        "I'm",
        "a",
        "single",
        "sentence!",
        "Hello, I'm",
        "I'm a",
        "a single",
        "single sentence!",
    ]
    tokens = tokenizer(inputs)
    assert tokens == tokens_expected


def test_string_split_tokenizer():
    inputs = "Multiple,Elements,Are here!"
    tokenizer = StringSplitTokenizer(",")
    tokens = tokenizer(inputs)
    assert tokens == ["Multiple", "Elements", "Are here!"]


def test_english_lemmatize_filter_tokenizer():
    inputs = "Hello, I'm a single sentence!"
    tokenizer = EnglishLemmatizeFilterTokenizer()
    tokens = tokenizer(inputs)
    assert len(tokens) > 0


def test_sentence_piece_tokenizer():
    inputs = "This is a sentence. And this is another one."
    tokenizer = get_tokenizer_from_registry("sentencepiece")()
    tokens = tokenizer(inputs)
    assert tokens == ["▁This", "▁is", "▁a", "▁sentence", ".", "▁And", "▁this", "▁is", "▁another", "▁one", "."]


def test_clip_tokenizer():
    inputs = "This is a sentence. And this is another one."
    tokenizer = get_tokenizer_from_registry("clip")()
    tokens = tokenizer(inputs)
    print(tokens)
    assert tokens == [
        "this</w>",
        "is</w>",
        "a</w>",
        "sentence</w>",
        ".</w>",
        "and</w>",
        "this</w>",
        "is</w>",
        "another</w>",
        "one</w>",
        ".</w>",
    ]


def test_gpt2_bpe_tokenizer():
    inputs = "This is a sentence. And this is another one."
    tokenizer = get_tokenizer_from_registry("gpt2bpe")()
    tokens = tokenizer(inputs)
    print(tokens)
    assert tokens == ["This", "Ġis", "Ġa", "Ġsentence", ".", "ĠAnd", "Ġthis", "Ġis", "Ġanother", "Ġone", "."]


def test_bert_tokenizer():
    inputs = "This is a sentence. And this is another one."
    tokenizer = get_tokenizer_from_registry("bert")()
    tokens = tokenizer(inputs)
    print(tokens)
    assert tokens == ["this", "is", "a", "sentence", ".", "and", "this", "is", "another", "one", "."]
