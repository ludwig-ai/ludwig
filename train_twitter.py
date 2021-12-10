import logging

from ludwig import api
from ludwig.visualize import learning_curves

config = {
    "training": {
        "epochs": 20,
        "learning_rate": 0.0002,
    },
    "input_features": [
        {
            "name": "created_at",
            # 'column': 'created_at', 'type': 'numerical'},
            "column": "created_at",
            "type": "date",
        },
        {"name": "default_profile", "column": "default_profile", "type": "binary"},
        {"name": "default_profile_image", "column": "default_profile_image", "type": "binary"},
        {"name": "favourites_count", "column": "favourites_count", "type": "numerical"},
        {"name": "followers_count", "column": "followers_count", "type": "numerical"},
        {"name": "friends_count", "column": "friends_count", "type": "numerical"},
        {"name": "geo_enabled", "column": "geo_enabled", "type": "binary"},
        {"name": "lang", "column": "lang", "type": "category"},
        {
            "name": "location",
            "column": "location",
            #  'type': 'numerical'},
            "type": "text",
            "encoder": "embed",
        },
        # {'name': 'profile_background_image_url',
        #  'column': 'profile_background_image_url',
        #  'type': 'image'},
        # {'name': 'profile_image_url',
        #  'column': 'profile_image_url',
        #  'type': 'image'},
        {
            "name": "screen_name",
            "column": "screen_name",
            #  'type': 'numerical'},
            "type": "text",
            "encoder": "embed",
        },
        {"name": "statuses_count", "column": "statuses_count", "type": "numerical"},
        {"name": "verified", "column": "verified", "type": "binary"},
        {"name": "average_tweets_per_day", "column": "average_tweets_per_day", "type": "numerical"},
        {"name": "account_age_days", "column": "account_age_days", "type": "numerical"},
    ],
    "output_features": [
        {
            "name": "account_type",
            "column": "account_type",
            # "type": "category",
            "type": "text",
            "cell_type": "lstm",
        }
    ],
    "combiner": {
        # concat, tabnet, transformer
        "type": "transformer",
    },
}

model = api.LudwigModel(config, logging_level=logging.INFO)
(train_stats, preprocessed_data, output_directory) = model.train(
    dataset="/Users/justin/Downloads/twitter_human_bots_dataset.csv",
    experiment_name="simple_experiment",
    model_name="simple_model",
)

# config = {
#     "training": {
#         "epochs": 20,
#         "learning_rate": 0.0002,
#     },
#     "input_features": [
#         {
#             "name": "english",
#             "column": "english",
#             "type": "text",
#             "encoder": "embed",
#         },
#     ],
#     "output_features": [
#         {
#             "name": "french",
#             "column": "french",
#             "type": "text",
#         }
#     ],
# }

# model = api.LudwigModel(config, logging_level=logging.INFO)
# (train_stats, preprocessed_data, output_directory) = model.train(
#     dataset="/Users/justin/Downloads/eng-fra2.tsv",
#     experiment_name="simple_experiment",
#     model_name="simple_model",
# )

learning_curves([train_stats], "accuracy", model_names=["0"], output_directory="./visualizations", file_format="png")
