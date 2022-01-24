""""Hyperopt python example, based on model_hyperopt_example.ipynb"""
import warnings
warnings.simplefilter('ignore')

import datetime
import yaml

from ludwig.datasets import forest_cover
from ludwig.hyperopt.run import hyperopt
from ludwig.visualize import hyperopt_results_to_dataframe, hyperopt_hiplot_cli, hyperopt_report_cli


train_df = forest_cover.load()

with open("hyperopt_example_config.yaml", "r") as config_file:
    try:
        config = yaml.safe_load(config_file)
    except yaml.YAMLError as e:
        print(e)


# -------------------- Defines Hyperparameter Search Space --------------------


ray_hyperopt_configs = {
    "parameters": {
        "training.learning_rate": {
            "type": "float",
            "lower": 0.0001,
            "upper": 0.01,
            "space": "loguniform",
            "steps": 3,
        },
        "training.batch_size": {
            "type": "int",
            "lower": 32,
            "upper": 256,
            "space": "lograndint",
            "steps": 5,
            "base": 2
        },
        "quality.fc_size": {
            "type": "int",
            "lower": 32,
            "upper": 256,
            "space": "randint",
            "steps": 5
        },
        "quality.num_fc_layers": {
            'type': 'int',
            'lower': 1,
            'upper': 5,
            "space": "randint",
            'steps': 4
        }
    },
    "goal": "minimize",
    'output_feature': "Cover_Type",
    'validation_metrics': 'loss'
}

local_hyperopt_configs = {
    "parameters": {
        "training.learning_rate": {
            "type": "float",
            "low": 0.0001,
            "high": 0.01,
            "space": "log",
            "steps": 3,
        },
        "training.batch_size": {
            "type": "int",
            "low": 32,
            "high": 256,
            "space": "log",
            "steps": 5,
            "base" : 2
        },
        "quality.fc_size": {
            "type": "int",
            "low": 32,
            "high": 256,
            "steps": 5
        },
        "quality.num_fc_layers": {
            'type': 'int',
            'low': 1,
            'high': 5,
            'space': 'linear',
            'steps': 4
        }
    },
    "goal": "minimize",
    'output_feature': "Cover_Type",
    'validation_metrics': 'loss'
}


# -------------------- Random Search with ray executor --------------------


print("starting:", datetime.datetime.now())
config['hyperopt'] = ray_hyperopt_configs
config['hyperopt']['executor'] = {'type': 'ray', 'num_workers': 4}
config['hyperopt']['sampler'] = {'type': 'ray'}
random_ray_results = hyperopt(
    config,
    dataset=train_df,
    output_directory='results_random_ray'
)


# -------------------- Random Search with serial executor --------------------


print("starting:", datetime.datetime.now())
config['hyperopt'] = local_hyperopt_configs.copy()
config['hyperopt']['executor'] = {'type': 'serial'}
config['hyperopt']['sampler'] = {'type': 'random', 'num_samples': 10}
random_serial_results = hyperopt(
    config,
    dataset= train_df,
    output_directory='results_random_serial'
)


# -------------------- Grid Search with 4 parallel executors (takes about 35 minutes) --------------------


print("starting:", datetime.datetime.now())
config['hyperopt'] = ray_hyperopt_configs
config['hyperopt']['executor'] = {'type': 'ray', 'num_workers': 4}
config['hyperopt']['sampler'] = {'type': 'ray'}
grid_ray_results = hyperopt(
    config,
    dataset=train_df,
    output_directory='results_grid_ray'  # location to place results
)

# Convert hyperparameter optimization results to pandas dataframes

df1 = hyperopt_results_to_dataframe(
    random_ray_results,
    ray_hyperopt_configs['parameters'],
    ray_hyperopt_configs['validation_metrics']
)
print("df1:")
print(df1.head(5))

df2 = hyperopt_results_to_dataframe(
    random_serial_results,
    local_hyperopt_configs['parameters'],
    local_hyperopt_configs['validation_metrics']
)
print("df2:")
print(df2.head(5))

df3 = hyperopt_results_to_dataframe(
    grid_ray_results,
    ray_hyperopt_configs['parameters'],
    ray_hyperopt_configs['validation_metrics']
)
print("df3:")
print(df3.head(5))

# Generate charts of hyperparameter optimization results.

hyperopt_report_cli(
    'results_random_ray/hyperopt_statistics.json',
    output_directory='./visualizations_random'
)

hyperopt_hiplot_cli(
    'results_random_ray/hyperopt_statistics.json',
    output_directory='./visualizations_random'
)

hyperopt_report_cli(
    'results_grid_ray/hyperopt_statistics.json',
    output_directory='./visualizations_grid'
)

hyperopt_hiplot_cli(
    'results_grid_ray/hyperopt_statistics.json',
    output_directory='./visualizations_grid'
)
