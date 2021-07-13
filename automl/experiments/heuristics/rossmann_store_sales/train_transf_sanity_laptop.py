import logging

from ludwig.api import LudwigModel
from ludwig.datasets import rossmann_store_sales

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
)

rossmann_store_sales_df = rossmann_store_sales.load()
model.train(
    dataset=rossmann_store_sales_df,
    experiment_name='rossmann_store_sales_transf_sanity_laptop',
    model_name='rossmann_store_sales_transf_sanity_laptop',
    skip_save_processed_input=True
)
