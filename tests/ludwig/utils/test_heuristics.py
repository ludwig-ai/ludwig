from typing import Any, Dict, Optional

import pytest

from ludwig.constants import DEFAULTS, ENCODER, TEXT, TRAINABLE, TRAINER, TYPE
from ludwig.schema.model_config import ModelConfig
from ludwig.utils.heuristics import get_auto_learning_rate


@pytest.mark.parametrize(
    "text_encoder,expected_lr",
    [
        (None, 0.001),
        ({}, 0.00001),
        ({"type": "parallel_cnn"}, 0.0001),
        ({"type": "bert"}, 0.00002),
        ({"type": "bert", "trainable": True}, 0.00001),
        ({"type": "bert", "trainable": True, "use_pretrained": False}, 0.0001),
    ],
    ids=["no_text", "default_electra", "parallel_cnn", "bert_fixed", "bert_trainable", "bert_untrained"],
)
def test_get_auto_learning_rate(text_encoder: Optional[Dict[str, Any]], expected_lr: float):
    input_features = [{"name": "bin1", "type": "binary"}]
    if text_encoder is not None:
        input_features.append({"name": "text1", "type": "text", "encoder": text_encoder})

    config = {
        "input_features": input_features,
        "output_features": [{"name": "bin2", "type": "binary"}],
        TRAINER: {
            "train_steps": 1,
            "learning_rate": "auto",
        },
        DEFAULTS: {
            TEXT: {
                ENCODER: {
                    # Note that encoder defaults are all or nothing: if the encoder type is overridden, trainable
                    # here is ignored
                    TYPE: "electra",
                    TRAINABLE: True,
                }
            }
        },
    }

    config = ModelConfig.from_dict(config)
    lr = get_auto_learning_rate(config)
    assert lr == expected_lr
