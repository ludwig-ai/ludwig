import pandas as pd
import yaml

from ludwig.api import LudwigModel


def test_eval_steps_determinism():
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
    base_model: HuggingFaceH4/tiny-random-LlamaForCausalLM

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
        max_new_tokens: 512

    preprocessing:
        split:
            type: fixed
            column: split

    adapter:
        type: lora

    quantization:
        bits: 4

    trainer:
        type: finetune
        optimizer:
            type: paged_adam
        epochs: 1
        batch_size: 1
        eval_batch_size: 2
        gradient_accumulation_steps: 16
        learning_rate: 0.0004
        learning_rate_scheduler:
            warmup_fraction: 0.03
        enable_gradient_checkpointing: True

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
        if results1[0]["out"][k] != 0:  # Some metrics will be 0 across all runs
            assert results1[0]["out"][k] != results2[0]["out"][k]
        assert results2[0]["out"][k] == results3[0]["out"][k]
