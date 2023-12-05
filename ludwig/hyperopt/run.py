import copy
import logging
import os
from pprint import pformat
from typing import List, Optional, Union

import pandas as pd
import yaml
from tabulate import tabulate

from ludwig.api import LudwigModel
from ludwig.backend import Backend, initialize_backend, LocalBackend
from ludwig.callbacks import Callback
from ludwig.constants import (
    AUTO,
    COMBINED,
    EXECUTOR,
    GOAL,
    HYPEROPT,
    LOSS,
    MAX_CONCURRENT_TRIALS,
    METRIC,
    NAME,
    OUTPUT_FEATURES,
    PARAMETERS,
    PREPROCESSING,
    SEARCH_ALG,
    SPLIT,
    TEST,
    TRAINING,
    TYPE,
    VALIDATION,
)
from ludwig.data.split import get_splitter
from ludwig.hyperopt.results import HyperoptResults
from ludwig.hyperopt.utils import (
    log_warning_if_all_grid_type_parameters,
    print_hyperopt_results,
    save_hyperopt_stats,
    should_tune_preprocessing,
    update_hyperopt_params_with_defaults,
)
from ludwig.schema.model_config import ModelConfig
from ludwig.utils.backward_compatibility import upgrade_config_dict_to_latest_version
from ludwig.utils.dataset_utils import generate_dataset_statistics
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import makedirs, open_file

try:
    from ray.tune import Callback as TuneCallback

    from ludwig.backend.ray import RayBackend
except ImportError:
    TuneCallback = object

    class RayBackend:
        pass


logger = logging.getLogger(__name__)


def hyperopt(
    config: Union[str, dict],
    dataset: Union[str, dict, pd.DataFrame] = None,
    training_set: Union[str, dict, pd.DataFrame] = None,
    validation_set: Union[str, dict, pd.DataFrame] = None,
    test_set: Union[str, dict, pd.DataFrame] = None,
    training_set_metadata: Union[str, dict] = None,
    data_format: str = None,
    experiment_name: str = "hyperopt",
    model_name: str = "run",
    resume: Optional[bool] = None,
    skip_save_training_description: bool = False,
    skip_save_training_statistics: bool = False,
    skip_save_model: bool = False,
    skip_save_progress: bool = False,
    skip_save_log: bool = False,
    skip_save_processed_input: bool = True,
    skip_save_unprocessed_output: bool = False,
    skip_save_predictions: bool = False,
    skip_save_eval_stats: bool = False,
    skip_save_hyperopt_statistics: bool = False,
    output_directory: str = "results",
    gpus: Union[str, int, List[int]] = None,
    gpu_memory_limit: Optional[float] = None,
    allow_parallel_threads: bool = True,
    callbacks: List[Callback] = None,
    tune_callbacks: List[TuneCallback] = None,
    backend: Union[Backend, str] = None,
    random_seed: int = default_random_seed,
    hyperopt_log_verbosity: int = 3,
    **kwargs,
) -> HyperoptResults:
    """This method performs an hyperparameter optimization.

    # Inputs

    :param config: (Union[str, dict]) config which defines
        the different parameters of the model, features, preprocessing and
        training.  If `str`, filepath to yaml configuration file.
    :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing the entire dataset to be used in the experiment.
        If it has a split column, it will be used for splitting (0 for train,
        1 for validation, 2 for test), otherwise the dataset will be
        randomly split.
    :param training_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing training data.
    :param validation_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing validation data.
    :param test_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing test data.
    :param training_set_metadata: (Union[str, dict], default: `None`)
        metadata JSON file or loaded metadata.  Intermediate preprocessed
        structure containing the mappings of the input
        dataset created the first time an input file is used in the same
        directory with the same name and a '.meta.json' extension.
    :param data_format: (str, default: `None`) format to interpret data
        sources. Will be inferred automatically if not specified.  Valid
        formats are `'auto'`, `'csv'`, `'df'`, `'dict'`, `'excel'`, `'feather'`,
        `'fwf'`, `'hdf5'` (cache file produced during previous training),
        `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
        `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
        `'stata'`, `'tsv'`.
    :param experiment_name: (str, default: `'experiment'`) name for
        the experiment.
    :param model_name: (str, default: `'run'`) name of the model that is
        being used.
    :param resume: (bool) If true, continue hyperopt from the state of the previous
        run in the output directory with the same experiment name. If false, will create
        new trials, ignoring any previous state, even if they exist in the output_directory.
        By default, will attempt to resume if there is already an existing experiment with
        the same name, and will create new trials if not.
    :param skip_save_training_description: (bool, default: `False`) disables
        saving the description JSON file.
    :param skip_save_training_statistics: (bool, default: `False`) disables
        saving training statistics JSON file.
    :param skip_save_model: (bool, default: `False`) disables
        saving model weights and hyperparameters each time the model
        improves. By default Ludwig saves model weights after each epoch
        the validation metric improves, but if the model is really big
        that can be time consuming. If you do not want to keep
        the weights and just find out what performance a model can get
        with a set of hyperparameters, use this parameter to skip it,
        but the model will not be loadable later on and the returned model
        will have the weights obtained at the end of training, instead of
        the weights of the epoch with the best validation performance.
    :param skip_save_progress: (bool, default: `False`) disables saving
        progress each epoch. By default Ludwig saves weights and stats
        after each epoch for enabling resuming of training, but if
        the model is really big that can be time consuming and will uses
        twice as much space, use this parameter to skip it, but training
        cannot be resumed later on.
    :param skip_save_log: (bool, default: `False`) disables saving
        TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
        but if it is not needed turning it off can slightly increase the
        overall speed.
    :param skip_save_processed_input: (bool, default: `False`) if input
        dataset is provided it is preprocessed and cached by saving an HDF5
        and JSON files to avoid running the preprocessing again. If this
        parameter is `False`, the HDF5 and JSON file are not saved.
    :param skip_save_unprocessed_output: (bool, default: `False`) by default
        predictions and their probabilities are saved in both raw
        unprocessed numpy files containing tensors and as postprocessed
        CSV files (one for each output feature). If this parameter is True,
        only the CSV ones are saved and the numpy ones are skipped.
    :param skip_save_predictions: (bool, default: `False`) skips saving test
        predictions CSV files.
    :param skip_save_eval_stats: (bool, default: `False`) skips saving test
        statistics JSON file.
    :param skip_save_hyperopt_statistics: (bool, default: `False`) skips saving
        hyperopt stats file.
    :param output_directory: (str, default: `'results'`) the directory that
        will contain the training statistics, TensorBoard logs, the saved
        model and the training progress files.
    :param gpus: (list, default: `None`) list of GPUs that are available
        for training.
    :param gpu_memory_limit: (float: default: `None`) maximum memory fraction
        [0, 1] allowed to allocate per GPU device.
    :param allow_parallel_threads: (bool, default: `True`) allow PyTorch
        to use multithreading parallelism to improve performance at
        the cost of determinism.
    :param callbacks: (list, default: `None`) a list of
        `ludwig.callbacks.Callback` objects that provide hooks into the
        Ludwig pipeline.
    :param backend: (Union[Backend, str]) `Backend` or string name
        of backend to use to execute preprocessing / training steps.
    :param random_seed: (int: default: 42) random seed used for weights
        initialization, splits and any other random function.
    :param hyperopt_log_verbosity: (int: default: 3) controls verbosity of
        ray tune log messages.  Valid values: 0 = silent, 1 = only status updates,
        2 = status and brief trial results, 3 = status and detailed trial results.

    # Return

    :return: (List[dict]) List of results for each trial, ordered by
        descending performance on the target metric.
    """
    from ludwig.hyperopt.execution import get_build_hyperopt_executor, RayTuneExecutor

    # check if config is a path or a dict
    if isinstance(config, str):  # assume path
        with open_file(config, "r") as def_file:
            config_dict = yaml.safe_load(def_file)
    else:
        config_dict = config

    if HYPEROPT not in config:
        raise ValueError("Hyperopt Section not present in config")

    # backwards compatibility
    upgraded_config = upgrade_config_dict_to_latest_version(config_dict)

    # Initialize config object
    config_obj = ModelConfig.from_dict(upgraded_config)

    # Retain pre-merged config for hyperopt schema generation
    premerged_config = copy.deepcopy(upgraded_config)

    # Get full config with defaults
    full_config = config_obj.to_dict()  # TODO (Connor): Refactor to use config object

    hyperopt_config = full_config[HYPEROPT]

    # Explicitly default to a local backend to avoid picking up Ray or Horovod
    # backend from the environment.
    backend = backend or config_dict.get("backend") or "local"
    backend = initialize_backend(backend)

    update_hyperopt_params_with_defaults(hyperopt_config)

    # Check if all features are grid type parameters and log UserWarning if needed
    log_warning_if_all_grid_type_parameters(hyperopt_config)

    # Infer max concurrent trials
    if hyperopt_config[EXECUTOR].get(MAX_CONCURRENT_TRIALS) == AUTO:
        hyperopt_config[EXECUTOR][MAX_CONCURRENT_TRIALS] = backend.max_concurrent_trials(hyperopt_config)
        logger.info(f"Setting max_concurrent_trials to {hyperopt_config[EXECUTOR][MAX_CONCURRENT_TRIALS]}")

    # Print hyperopt config
    logger.info("Hyperopt Config")
    logger.info(pformat(hyperopt_config, indent=4))
    logger.info("\n")

    search_alg = hyperopt_config[SEARCH_ALG]
    executor = hyperopt_config[EXECUTOR]
    parameters = hyperopt_config[PARAMETERS]
    split = hyperopt_config[SPLIT]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config[METRIC]
    goal = hyperopt_config[GOAL]

    ######################
    # check validity of output_feature / metric/ split combination
    ######################
    splitter = get_splitter(**full_config[PREPROCESSING]["split"])
    if split == TRAINING:
        if training_set is None and not splitter.has_split(0):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                "was not provided, "
                "or the split amount specified in the preprocessing section "
                "of the config is not greater than 0".format(split)
            )
    elif split == VALIDATION:
        if validation_set is None and not splitter.has_split(1):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                "was not provided, "
                "or the split amount specified in the preprocessing section "
                "of the config is not greater than 0".format(split)
            )
    elif split == TEST:
        if test_set is None and not splitter.has_split(2):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                "was not provided, "
                "or the split amount specified in the preprocessing section "
                "of the config is not greater than 0".format(split)
            )
    else:
        raise ValueError(
            'unrecognized hyperopt split "{}". ' "Please provide one of: {}".format(split, {TRAINING, VALIDATION, TEST})
        )
    if output_feature == COMBINED:
        if metric != LOSS:
            raise ValueError('The only valid metric for "combined" output feature is "loss"')
    else:
        output_feature_names = {of[NAME] for of in full_config[OUTPUT_FEATURES]}
        if output_feature not in output_feature_names:
            raise ValueError(
                'The output feature specified for hyperopt "{}" '
                "cannot be found in the config. "
                'Available ones are: {} and "combined"'.format(output_feature, output_feature_names)
            )

    hyperopt_executor = get_build_hyperopt_executor(executor[TYPE])(
        parameters, output_feature, metric, goal, split, search_alg=search_alg, **executor
    )

    if not (
        isinstance(backend, LocalBackend)
        or (isinstance(hyperopt_executor, RayTuneExecutor) and isinstance(backend, RayBackend))
    ):
        raise ValueError(
            "Hyperopt requires using a `local` backend at this time, or " "`ray` backend with `ray` executor."
        )

    for callback in callbacks or []:
        callback.on_hyperopt_init(experiment_name)

    if not should_tune_preprocessing(full_config):
        # preprocessing is not being tuned, so generate it once before starting trials
        for callback in callbacks or []:
            callback.on_hyperopt_preprocessing_start(experiment_name)

        model = LudwigModel(
            config=full_config,
            backend=backend,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            callbacks=callbacks,
        )

        training_set, validation_set, test_set, training_set_metadata = model.preprocess(
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            training_set_metadata=training_set_metadata,
            data_format=data_format,
            skip_save_processed_input=skip_save_processed_input,
            random_seed=random_seed,
        )
        dataset = None

        dataset_statistics = generate_dataset_statistics(training_set, validation_set, test_set)

        logger.info("\nDataset Statistics")
        logger.info(tabulate(dataset_statistics, headers="firstrow", tablefmt="fancy_grid"))

        for callback in callbacks or []:
            callback.on_hyperopt_preprocessing_end(experiment_name)

    for callback in callbacks or []:
        callback.on_hyperopt_start(experiment_name)

    hyperopt_results = hyperopt_executor.execute(
        premerged_config,
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        experiment_name=experiment_name,
        model_name=model_name,
        resume=resume,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        skip_save_eval_stats=skip_save_eval_stats,
        output_directory=output_directory,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        callbacks=callbacks,
        tune_callbacks=tune_callbacks,
        backend=backend,
        random_seed=random_seed,
        hyperopt_log_verbosity=hyperopt_log_verbosity,
        **kwargs,
    )

    if backend.is_coordinator():
        print_hyperopt_results(hyperopt_results)

        if not skip_save_hyperopt_statistics:
            with backend.storage.artifacts.use_credentials():
                results_directory = os.path.join(output_directory, experiment_name)
                makedirs(results_directory, exist_ok=True)

                hyperopt_stats = {
                    "hyperopt_config": hyperopt_config,
                    "hyperopt_results": [t.to_dict() for t in hyperopt_results.ordered_trials],
                }

                save_hyperopt_stats(hyperopt_stats, results_directory)
                logger.info(f"Hyperopt stats saved to: {results_directory}")

    for callback in callbacks or []:
        callback.on_hyperopt_end(experiment_name)
        callback.on_hyperopt_finish(experiment_name)

    logger.info("Finished hyperopt")

    return hyperopt_results
