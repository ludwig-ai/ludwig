import logging

from ludwig.api import LudwigModel
from ludwig.datasets import poker_hand

model = LudwigModel(
    config='config_tabnet_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

poker_hand_df, _, _ = poker_hand.load()
model.train(
    dataset=poker_hand_df,
    experiment_name='poker_hand_tabnet_sanity_laptop',
    model_name='poker_hand_tabnet_sanity_laptop',
    skip_save_processed_input=True
)
