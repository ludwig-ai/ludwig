import logging

from ludwig.api import LudwigModel
from ludwig.datasets import santander_customer_satisfaction

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

santander_customer_satisfaction_df = santander_customer_satisfaction.load()
model.train(
    dataset=santander_customer_satisfaction_df,
    experiment_name='santander_customer_satisfaction_transf_sanity_laptop',
    model_name='santander_customer_satisfaction_transf_sanity_laptop',
    skip_save_processed_input=True
)
