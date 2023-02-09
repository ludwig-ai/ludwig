import torch

from ludwig.explain.captum import get_token_attributions


def test_get_token_attributions():
    feature_name = "text_8D824"
    input_ids = torch.tensor([[1, 5, 6, 4, 4, 4, 6, 0, 2], [1, 4, 5, 6, 4, 4, 6, 5, 0]], dtype=torch.int8)
    model = type("Model", (), {})()
    model.training_set_metadata = {
        feature_name: {
            "idx2str": [
                "<EOS>",
                "<SOS>",
                "<PAD>",
                "<UNK>",
                "oypszb",
                "yscnrkzw",
                "llcgslcvzr",
            ]
        }
    }
    token_attributions = torch.tensor(
        [
            [-0.1289, -0.3222, -0.4931, -0.2914, -0.2891, -0.2871, -0.4118, -0.4647, 0.0000],
            # zero norm should not lead to division by zero
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ],
        dtype=torch.float64,
    )

    toks_and_attrs = get_token_attributions(model, feature_name, input_ids, token_attributions)

    # assert equality up to 4 decimal places
    assert [[(ta[0], round(ta[1], 4)) for ta in tas] for tas in toks_and_attrs] == [
        [
            # normalized attributions
            ("<SOS>", -0.1289),
            ("yscnrkzw", -0.3222),
            ("llcgslcvzr", -0.4931),
            ("oypszb", -0.2914),
            ("oypszb", -0.2891),
            ("oypszb", -0.2871),
            ("llcgslcvzr", -0.4118),
            ("<EOS>", -0.4647),
            ("<PAD>", 0.0),
        ],
        [
            # zero norm should retain original zero attributions
            ("<SOS>", 0.0),
            ("oypszb", 0.0),
            ("yscnrkzw", 0.0),
            ("llcgslcvzr", 0.0),
            ("oypszb", 0.0),
            ("oypszb", 0.0),
            ("llcgslcvzr", 0.0),
            ("yscnrkzw", 0.0),
            ("<EOS>", 0.0),
        ],
    ]
