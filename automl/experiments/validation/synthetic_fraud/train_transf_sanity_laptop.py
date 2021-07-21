import logging

from ludwig.api import LudwigModel
from ludwig.datasets import synthetic_fraud

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

synthetic_fraud_df = synthetic_fraud.load()
model.train(
    dataset=synthetic_fraud_df,
    experiment_name='synthetic_fraud_transf_sanity_laptop',
    model_name='synthetic_fraud_transf_sanity_laptop',
    skip_save_processed_input=True
)
