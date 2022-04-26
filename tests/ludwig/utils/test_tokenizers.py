from ludwig.utils.tokenizers import tokenizer_registry


def test_space_punct_tokenizer_parity():
    """Tests that the torchscript-compatible implementation of space_punct tokenizer is equivalent to the regex one."""

    sentences = ["this is123, a_sent-ence", "...test test?", "hello..world", "", "    ", ".", "ç–12¡™£•"]
    legacy_tokenizer = tokenizer_registry["legacy_space_punct"]()
    tokens_expected = [legacy_tokenizer(s) for s in sentences]

    tokenizer = tokenizer_registry["space_punct"]()
    tokens = [tokenizer(s) for s in sentences]

    assert all(tokens[i] == tokens_expected[i] for i in range(len(sentences)))
