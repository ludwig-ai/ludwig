import os

import pandas as pd
import yaml

from ludwig.api import LudwigModel


def test_eval_steps_determinism():
    # Force CPU to avoid CUBLAS errors with tiny random LLM models on GPU.
    old_val = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        _run_eval_steps_determinism()
    finally:
        if old_val is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_val


def _run_eval_steps_determinism():
    df = pd.DataFrame(
        {
            "in": "a b c d e f g h i j k l m n o p q r s t".split(" "),
            "out": [i for i in range(20)],
            "split": ([0] * 10) + ([2] * 10),
        }
    )
    config = yaml.safe_load(
        """
    model_type: llm
    base_model: hf-internal-testing/tiny-random-GPT2LMHeadModel

    input_features:
      - name: in
        type: text

    output_features:
      - name: out
        type: text

    prompt:
        template: >-
            {in}

    generation:
        temperature: null
        do_sample: False
        max_new_tokens: 64

    preprocessing:
        split:
            type: fixed
            column: split

    trainer:
        type: finetune
        epochs: 1
        batch_size: 1
        eval_batch_size: 2
        learning_rate: 0.00001
        gradient_clipping:
            clipglobalnorm: 1.0

    backend:
        type: local
    """
    )
    model = LudwigModel(config=config)
    model.train(df)
    results1 = model.evaluate(df)

    model.config_obj.trainer.eval_steps = 4
    results2 = model.evaluate(df)
    results3 = model.evaluate(df)

    for k in results1[0]["out"]:
        # The core assertion: repeated evaluations with the same eval_steps
        # setting must produce identical results (determinism).
        assert (
            results2[0]["out"][k] == results3[0]["out"][k]
        ), f"Metric '{k}' differs between repeated evaluations: {results2[0]['out'][k]} vs {results3[0]['out'][k]}"
