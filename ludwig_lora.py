import logging
import os

import pandas as pd
import yaml

from ludwig.api import LudwigModel

df = pd.read_csv("peft_twitter.csv")

# tokenizer.batch_decode([dataset.to_numpy()[0][0]], skip_special_tokens=True)

config = yaml.safe_load(
    """
        model_type: llm
        model_name: hf-internal-testing/tiny-random-GPTJForCausalLM
        adapter:
            type: lora
            r: 8
            alpha: 16
            dropout: 0
        input_features:
            - name: Tweet text
              type: text
              preprocessing:
                max_sequence_length: 64
                lowercase: false
        output_features:
            - name: text_label
              type: text
              preprocessing:
                lowercase: false
        prompt:
            # Extra space after the Label : is important
            task: >-
                Label :
            template: >-
                Tweet text : {__sample__} {__task__}
        preprocessing:
            split:
                type: fixed
                column: split
        trainer:
            type: finetune
            batch_size: 8
            learning_rate: 0.03
            # steps_per_checkpoint: 1
            epochs: 10
            early_stop: -1
            optimizer:
                type: adamw
            regularization_type: null
            should_shuffle: false
            increase_batch_size_on_plateau: 0
            gradient_clipping:
                clipglobalnorm: 1.0
            learning_rate_scheduler:
                decay: null
        backend:
            type: local
    """
)

model = LudwigModel(config=config, logging_level=logging.INFO)

# initiate model training
(
    train_stats,  # dictionary containing training statistics
    preprocessed_data,  # tuple Ludwig Dataset objects of pre-processed training data
    output_directory,  # location of training results stored on disk
) = model.train(
    dataset=df,
    experiment_name="ludwig_peft",
    model_name="ludwig_peft",
)

# list contents of output directory
print("contents of output directory:", output_directory)
for item in os.listdir(output_directory):
    print("\t", item)
