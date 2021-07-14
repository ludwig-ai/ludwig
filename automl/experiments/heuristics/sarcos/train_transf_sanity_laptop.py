import logging

from ludwig.api import LudwigModel
from ludwig.datasets import sarcos

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

sarcos_df, _, _ = sarcos.load()
model.train(
    dataset=sarcos_df,
    experiment_name='sarcos_transf_sanity_laptop',
    model_name='sarcos_transf_sanity_laptop',
    skip_save_processed_input=True
)
