import logging

from ludwig.api import LudwigModel
from ludwig.datasets import ieee_fraud

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

ieee_fraud_df = ieee_fraud.load()
model.train(
    dataset=ieee_fraud_df,
    experiment_name='ieee_fraud_transf_sanity_laptop',
    model_name='ieee_fraud_transf_sanity_laptop',
    skip_save_processed_input=True
)
