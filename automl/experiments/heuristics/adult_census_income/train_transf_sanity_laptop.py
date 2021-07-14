import logging

from ludwig.api import LudwigModel
from ludwig.datasets import adult_census_income

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

adult_census_income_df = adult_census_income.load()
model.train(
    dataset=adult_census_income_df,
    experiment_name='adult_census_income_transf_sanity_laptop',
    model_name='adult_census_income_transf_sanity_laptop',
    skip_save_processed_input=True
)
