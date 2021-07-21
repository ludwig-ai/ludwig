import logging

from ludwig.api import LudwigModel
from ludwig.datasets import bnp_claims_management

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

bnp_claims_management_df = bnp_claims_management.load()
model.train(
    dataset=bnp_claims_management_df,
    experiment_name='bnp_claims_management_transf_sanity_laptop',
    model_name='bnp_claims_management_transf_sanity_laptop',
    skip_save_processed_input=True
)
