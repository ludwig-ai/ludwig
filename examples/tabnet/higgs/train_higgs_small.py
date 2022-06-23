import logging

from ludwig.api import LudwigModel
from ludwig.datasets import higgs

model = LudwigModel(
    config="small_config.yaml",
    logging_level=logging.INFO,
)

higgs_df = higgs.load()
model.train(dataset=higgs_df, experiment_name="higgs_small", model_name="higgs_tabnet_small")
