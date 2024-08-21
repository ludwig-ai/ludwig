# Ludwig Benchmarking

### Some use cases

- Regression testing for ML experiments across releases and PRs.
- Model performance testing for experimenting with new features and hyperparameters.
- Resource usage tracking for the full ML pipeline.

## Ludwig benchmarking CLI and API

To run benchmarks, run the following command from the command line

```
ludwig benchmark --benchmarking_config path/to/benchmarking/config.yaml
```

To use the API

```
from ludwig.benchmarking.benchmark import benchmark

benchmarking_config_path = "path/to/benchmarking/config.yaml"
benchmark(benchmarking_config_path)
```

In what follows, we describe what the benchmarking config looks for
multiple use cases.

## The benchmarking config

The benchmarking config is where you can specify

1. The datasets you want to run the benchmarks on and their configs.
1. Whether these experiments are hyperopt or regular train and eval experiments.
1. The name of the experiment.
1. A python script to edit the specified Ludwig configs programmatically/on the fly.
1. The export path of these experiment's artifacts. (remotely or locally)
1. Whether to use `LudwigProfiler` to track resource
   usage for preprocessing, training, and evaluation of the experiment.

You can find an example of a benchmarking config in the `examples/` directory.

## Basic Usage

Say you implemented a new feature and would like to test it on several datasets.
In this case, this is what the benchmarking config could look like

```
experiment_name: SMOTE_test
hyperopt: false
export:
  export_artifacts: true
  export_base_path: s3://benchmarking.us-west-2.ludwig.com/bench/    # include the slash at the end.
experiments:
  - dataset_name: ames_housing
    config_path: /home/ray/configs/ames_housing_SMOTE.yaml
    experiment_name: SMOTE_test_with_hyperopt
    hyperopt: true
  - dataset_name: protein
  - ...
    ...
  - dataset_name: mercedes_benz_greener
    config_path: /home/ray/configs/mercedes_benz_greener_SMOTE.yaml
```

For each experiment:

- `dataset_name`: name of the dataset in `ludwig.datasets` to run the benchmark on.
- `config_path` (optional): path to Ludwig config. If not specified, this will load
  the config corresponding to the dataset only containing `input_features` and
  `output_features`.

This will run `LudwigModel.experiment` on the datasets with their specified configs.
If these configs contain a hyperopt section and you'd like to run hyperopt, change
to `hyperopt: true`.
You can specify the same dataset multiple times with different configs.

**Exporting artifacts**
By specifying `export_artifacts: true`, this will export the experiment artifacts
to the `export_base_path`. Once the model is trained and the artifacts are pushed
to the specified path, you will get a similar message to the following:

```
Uploaded metrics report and experiment config to
	 s3://benchmarking.us-west-2.ludwig.com/bench/ames_housing/SMOTE_test
```

This is the directory structure of the exported artifacts for one of the experiments.

```
s3://benchmarking.us-west-2.ludwig.com/bench/
└── ames_housing
    └── SMOTE_test
        ├── config.yaml
        └── experiment_run
            ├── description.json
            ├── model
            │   ├── logs
            │   │   ├── test
            │   │   │   └── events.out.tfevents.1663320893.macbook-pro.lan.8043.2
            │   │   ├── training
            │   │   │   └── events.out.tfevents.1663320893.macbook-pro.lan.8043.0
            │   │   └── validation
            │   │       └── events.out.tfevents.1663320893.macbook-pro.lan.8043.1
            │   ├── model_hyperparameters.json
            │   ├── training_progress.json
            │   └── training_set_metadata.json
            ├── test_statistics.json
            └── training_statistics.json
```

Note that model checkpoints are not exported. Any other experiments on
the `ames_housing` dataset will also live under
`s3://benchmarking.us-west-2.ludwig.com/bench/ames_housing/`

**Overriding parameters**
The benchmarking config's global parameters `experiment_name` and `hyperopt` can be overridden
if specified within an experiment.

## Programmatically editing Ludwig configs

To apply some changes to multiple Ludwig configs, you can specify a path to a python script
that does this without the need to do manual modifications across many configs. Example:

```
experiment_name: logistic_regression_hyperopt
hyperopt: true
process_config_file_path: /home/ray/process_config.py
export:
  export_artifacts: true
  export_base_path: s3://benchmarking.us-west-2.ludwig.com/bench/    # include the slash at the end.
experiments:
  - dataset_name: ames_housing
    config_path: /home/ray/configs/ames_housing_SMOTE.yaml
  ...
```

In `/home/ray/process_config.py`, define the following function and add custom code to modify
ludwig configs

```
def process_config(ludwig_config: dict, experiment_dict: dict) -> dict:
    """Modify a Ludwig config.

    :param ludwig_config: a Ludwig config.
    :param experiment_dict: a benchmarking config experiment dictionary.

    returns: a modified Ludwig config.
    """

    # code to modify the Ludwig config.

    return ludwig_config
```

View the `examples/` folder for an example `process_config.py`.

## Benchmarking the resource usage with `LudwigProfiler`

To benchmark the resource usage of the preprocessing, training, and evaluation
steps of `LudwigModel.experiment`, you can specify in the benchmarking config
global parameters

```
profiler:
  enable: true
  use_torch_profiler: false
  logging_interval: 0.1
```

- `enable: true` will run benchmarking with `LudwigProfiler`.
- `use_torch_profiler: false` will skip using the torch profiler.
- `logging_interval: 0.1` will instruct `LudwigProfiler` to collect
  resource usage information every 0.1 seconds.

Note that profiling is only enabled in the case where `hyperopt: false`.
`LudwigProfiler` is passed in to `LudwigModel` callbacks. The specific
callbacks that will be called are:

- `on_preprocess_(start/end)`
- `on_train_(start/end)`
- `on_evaluation_(start/end)`

This is an example directory output when using the profiler:

```
full_bench_with_profiler_with_torch
├── config.yaml
├── experiment_run
├── system_resource_usage
│   ├── evaluation
│   │   └── run_0.json
│   ├── preprocessing
│   │   └── run_0.json
│   └── training
│       └── run_0.json
└── torch_ops_resource_usage
    ├── evaluation
    │   └── run_0.json
    ├── preprocessing
    │   └── run_0.json
    └── training
        └── run_0.json
```

The only difference is the `system_resource_usage` and `torch_ops_resource_usage`.
The difference between these two outputs can be found in the `LudwigProfiler` README.

## Parameters and defaults

Each of these parameters can also be specified in the experiments section to override the global value.
If not specified, the value of the global parameter will be propagated to the experiments.

- `experiment_name` (required): name of the benchmarking run.
- `export` (required): dictionary specifying whether to export the experiment artifacts and the export path.
- `hyperopt` (optional): whether this is a hyperopt run or `LudwigModel.experiment`.
- `process_config_file_path` (optional): path to python script that will modify configs.
- `profiler` (optional): dictionary specifying whether to use the profiler and its parameters.

## Comparing experiments

You can summarize the exported artifacts of two experiments on multiple datasets.
For example, if you ran two experiments on the datasets `ames_housing` called
`small_batch_size` and `big_batch_size` where you varied the batch size,
you can create a diff summary of the model performance and resource usage of the two
experiments. This is how:

```
from ludwig.benchmarking.summarize import summarize_metrics

dataset_list, metric_diffs, resource_usage_diffs = summarize_metrics(
    bench_config_path = "path/to/benchmarking_config.yaml",
    base_experiment = "small_batch_size",
    experimental_experiment = "big_batch_size",
    download_base_path = "s3://benchmarking.us-west-2.ludwig.com/bench/")
```

This will print

```
Model performance metrics for *small_batch_size* vs. *big_batch_size* on dataset *ames_housing*
Output Feature Name  Metric Name                       small_batch_size big_batch_size Diff          Diff Percentage
SalePrice            mean_absolute_error               180551.609    180425.109    -126.5        -0.07
SalePrice            mean_squared_error                38668763136.0 38618021888.0 -50741248.0   -0.131
SalePrice            r2                                -5.399        -5.391        0.008         -0.156
SalePrice            root_mean_squared_error           196643.75     196514.688    -129.062      -0.066
SalePrice            root_mean_squared_percentage_error 1.001         1.001         -0.001        -0.07
Exported a CSV report to summarize_output/performance_metrics/ames_housing/small_batch_size-big_batch_size.csv

Resource usage for *small_batch_size* vs. *big_batch_size* on *training* of dataset *ames_housing*
Metric Name                          small_batch_size     big_batch_size       Diff                 Diff Percentage
average_cpu_memory_usage             106.96 Mb            109.43 Mb            2.48 Mb              2.315
average_cpu_utilization              1.2966666666666666   1.345                0.04833333333333334  3.728
average_global_cpu_memory_available  3.46 Gb              3.46 Gb              -1.10 Mb             -0.031
average_global_cpu_utilization       37.43333333333334    40.49                3.056666666666665    8.166
disk_footprint                       372736               413696               40960                10.989
max_cpu_memory_usage                 107.50 Mb            111.93 Mb            4.43 Mb              4.117
max_cpu_utilization                  1.44                 1.67                 0.22999999999999998  15.972
max_global_cpu_utilization           54.1                 60.9                 6.799999999999997    12.569
min_global_cpu_memory_available      3.46 Gb              3.46 Gb              -712.00 Kb           -0.02
num_cpu                              10                   10                   0                    0.0
num_oom_events                       0                    0                    0                    inf
num_runs                             1                    1                    0                    0.0
torch_cpu_average_memory_used        81.44 Kb             381.15 Kb            299.70 Kb            367.992
torch_cpu_max_memory_used            334.26 Kb            2.65 Mb              2.32 Mb              711.877
torch_cpu_time                       57.400ms             130.199ms            72.799ms             126.828
torch_cuda_time                      0.000us              0.000us              0.000us              inf
total_cpu_memory_size                32.00 Gb             32.00 Gb             0 b                  0.0
total_execution_time                 334.502ms            1.114s               779.024ms            232.891
Exported a CSV report to summarize_output/resource_usage_metrics/ames_housing/training-small_batch_size-big_batch_size.csv

Resource usage for *small_batch_size* vs. *big_batch_size* on *evaluation* of dataset *ames_housing*
...
Resource usage for *small_batch_size* vs. *big_batch_size* on *preprocessing* of dataset *ames_housing*
...
```
