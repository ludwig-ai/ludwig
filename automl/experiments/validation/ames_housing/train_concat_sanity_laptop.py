import logging

from ludwig.api import LudwigModel
from ludwig.datasets import ames_housing

model = LudwigModel(
    config='config_concat_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

ames_housing_df = ames_housing.load()
model.train(
    dataset=ames_housing_df,
    experiment_name='ames_housing_concat_sanity_laptop',
    model_name='ames_housing_concat_sanity_laptop',
    skip_save_processed_input=True
)
