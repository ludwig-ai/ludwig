from ludwig.automl import automl


# def create_auto_config(
#     dataset: Union[str, pd.DataFrame, dd.core.DataFrame, DatasetInfo],
#     target: str,
#     time_limit_s: Union[int, float],
#     tune_for_memory: bool,
# ) -> dict:

def test_create_auto_config():
    dataset = '/Users/justin/Downloads/twitter_human_bots_dataset.csv'
    target = 'account_type'
    time_limit_s = 100000
    tune_for_memory = False

    config = automl.create_auto_config(
        dataset, target, time_limit_s, tune_for_memory)
    print(config)
