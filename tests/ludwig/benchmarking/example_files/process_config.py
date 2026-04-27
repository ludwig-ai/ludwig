def process_config(ludwig_config: dict, experiment_dict: dict) -> dict:
    """Modify a Ludwig config.

    :param ludwig_config: a Ludwig config.
    :param experiment_dict: a benchmarking config experiment dictionary.

    returns: a modified Ludwig config.
    """
    # Only keep input_features and output_features for the ames_housing dataset.
    if experiment_dict["dataset_name"] == "ames_housing":
        main_config_keys = list(ludwig_config.keys())
        for key in main_config_keys:
            if key not in ["input_features", "output_features"]:
                del ludwig_config[key]

    # Set the early_stop criteria to stop training after 7 epochs of no score improvement.
    ludwig_config["trainer"] = {"early_stop": 7}

    # use sparse encoder for categorical features to mimic logreg
    ludwig_config["combiner"] = {"type": "concat"}
    for i, feature in enumerate(ludwig_config["input_features"]):
        if feature["type"] == "category":
            ludwig_config["input_features"][i]["encoder"] = "sparse"
    for i, feature in enumerate(ludwig_config["output_features"]):
        if feature["type"] == "category":
            ludwig_config["output_features"][i]["encoder"] = "sparse"

    return ludwig_config
