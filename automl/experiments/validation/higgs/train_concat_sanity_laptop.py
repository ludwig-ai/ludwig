import logging

from ludwig.api import LudwigModel
from ludwig.datasets import higgs

model = LudwigModel(
    config='config_concat_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

higgs_df = higgs.load()
model.train(
    dataset=higgs_df,
    experiment_name='higgs_concat_sanity_laptop',
    model_name='higgs_concat_sanity_laptop',
    skip_save_processed_input=True
)
