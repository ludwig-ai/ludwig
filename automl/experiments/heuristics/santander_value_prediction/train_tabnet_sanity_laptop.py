import logging

from ludwig.api import LudwigModel
from ludwig.datasets import santander_value_prediction

model = LudwigModel(
    config='config_tabnet_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

santander_value_prediction_df = santander_value_prediction.load()
model.train(
    dataset=santander_value_prediction_df,
    experiment_name='santander_value_prediction_tabnet_sanity_laptop',
    model_name='santander_value_prediction_tabnet_sanity_laptop',
    skip_save_processed_input=True
)
