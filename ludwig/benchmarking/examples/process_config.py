"""This function will take in a Ludwig config, strip away all its parameters except input and output featuresand
add some other parameters to run logistic regression hyperopt."""


def process_config(ludwig_config: dict, experiment_dict: dict) -> dict:
    """Modify a Ludwig config by programmatically adding elements to the config dictionary.

    The purpose is to apply changes for all datasets that are the same or are based on the
     attributes of `experiment_dict` (e.g. dataset_name) removing the need to manually apply
     small changes to configs on many datasets.

    :param ludwig_config: a Ludwig config.
    :param experiment_dict: a benchmarking config experiment dictionary.

    Returns: a modified Ludwig config.
    """

    # only keep input_features and output_features
    main_config_keys = list(ludwig_config.keys())
    for key in main_config_keys:
        if key not in ["input_features", "output_features"]:
            del ludwig_config[key]

    temp = {
        "preprocessing": {"split": {"type": "fixed"}},
        "trainer": {"epochs": 1024, "early_stop": 7, "eval_batch_size": 16384, "evaluate_training_set": False},
        "hyperopt": {
            "goal": "maximize",
            "output_feature": None,
            "metric": None,
            "split": "validation",
            "parameters": {
                "defaults.number.preprocessing.normalization": {"space": "choice", "categories": ["zscore", None]},
                "defaults.number.preprocessing.missing_value_strategy": {
                    "space": "choice",
                    "categories": ["fill_with_const", "fill_with_mean"],
                },
                "combiner.type": {"space": "choice", "categories": ["tabnet", "concat"]},
                "trainer.learning_rate_scheduler.decay": {"space": "choice", "categories": [True, False]},
                "trainer.learning_rate": {"space": "loguniform", "lower": 0.0001, "upper": 0.1},
                "trainer.learning_rate_scheduler.decay_rate": {"space": "uniform", "lower": 0.4, "upper": 0.96},
                "trainer.batch_size": {"space": "randint", "lower": 32, "upper": 2048},
            },
            "search_alg": {"type": "variant_generator"},
            "executor": {"type": "ray", "num_samples": 1000},
            "scheduler": {"type": "bohb", "reduction_factor": 2},
        },
    }

    # add config parameters from temp
    for key, value in temp.items():
        ludwig_config[key] = value

    dataset_name_to_metric = {
        "ames_housing": "r2",
        "mercedes_benz_greener": "r2",
        "mushroom_edibility": "accuracy",
        "amazon_employee_access_challenge": "roc_auc",
        "naval": "r2",
        "sarcos": "r2",
        "protein": "r2",
        "adult_census_income": "accuracy",
        "otto_group_product": "accuracy",
        "santander_customer_satisfaction": "accuracy",
        "amazon_employee_access": "roc_auc",
        "numerai28pt6": "accuracy",
        "bnp_claims_management": "accuracy",
        "allstate_claims_severity": "r2",
        "santander_customer_transaction": "accuracy",
        "connect4": "accuracy",
        "forest_cover": "accuracy",
        "ieee_fraud": "accuracy",
        "porto_seguro_safe_driver": "accuracy",
        "walmart_recruiting": "accuracy",
        "poker_hand": "accuracy",
        "higgs": "accuracy",
    }

    # add hyperopt output feature and metric.
    dataset_name = experiment_dict["dataset_name"]
    ludwig_config["hyperopt"]["metric"] = dataset_name_to_metric[dataset_name]
    ludwig_config["hyperopt"]["output_feature"] = ludwig_config["output_features"][0]["name"]

    # use sparse encoder for categorical features to mimic logistic regression.
    for i, feature in enumerate(ludwig_config["input_features"]):
        if feature["type"] == "category":
            ludwig_config["input_features"][i]["encoder"] = "sparse"
    for i, feature in enumerate(ludwig_config["output_features"]):
        if feature["type"] == "category":
            ludwig_config["output_features"][i]["encoder"] = "sparse"

    # make sure to return the ludwig_config
    return ludwig_config
