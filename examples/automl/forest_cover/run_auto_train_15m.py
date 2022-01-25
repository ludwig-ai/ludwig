import pprint

from ludwig.datasets import forest_cover
from ludwig.automl import auto_train

forest_cover_df = forest_cover.load(use_tabnet_split=True)

auto_train_results = auto_train(
    dataset=forest_cover_df,
    target='Cover_Type',
    time_limit_s=60*15,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
