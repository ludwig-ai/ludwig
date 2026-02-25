from ludwig.utils.tokenizers import EnglishLemmatizeFilterTokenizer, NgramTokenizer, StringSplitTokenizer


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
