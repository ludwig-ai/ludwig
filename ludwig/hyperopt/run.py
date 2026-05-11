import copy
import logging
import os
from pprint import pformat

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
    config: str | dict,
    dataset: str | dict | pd.DataFrame = None,
    training_set: str | dict | pd.DataFrame = None,
    validation_set: str | dict | pd.DataFrame = None,
    test_set: str | dict | pd.DataFrame = None,
    training_set_metadata: str | dict | None = None,
    data_format: str | None = None,
    experiment_name: str = "hyperopt",
    model_name: str = "run",
    resume: bool | None = None,
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
    gpus: str | int | list[int] | None = None,
    gpu_memory_limit: float | None = None,
    allow_parallel_threads: bool = True,
    callbacks: list[Callback] | None = None,
    tune_callbacks: list[TuneCallback] | None = None,
    backend: Backend | str = None,
    random_seed: int = default_random_seed,
    hyperopt_log_verbosity: int = 3,
    **kwargs,
) -> HyperoptResults:
    """Run hyperparameter optimization.

    Args:
        config: Config dict or path to a YAML config file.
        dataset: Source containing the entire dataset. If it has a split
            column, it will be used for splitting (0: train, 1: validation,
            2: test); otherwise the dataset will be randomly split.
        training_set: Source containing training data.
        validation_set: Source containing validation data.
        test_set: Source containing test data.
        training_set_metadata: Metadata JSON file or loaded metadata dict.
            Intermediate preprocessed structure containing feature mappings
            created the first time an input file is used.
        data_format: Format to interpret data sources. Inferred automatically
            if not specified. Valid values: ``'auto'``, ``'csv'``,
            ``'excel'``, ``'feather'``, ``'fwf'``, ``'hdf5'``,
            ``'html'``, ``'json'``, ``'jsonl'``, ``'parquet'``,
            ``'pickle'``, ``'sas'``, ``'spss'``, ``'stata'``, ``'tsv'``.
        experiment_name: Name for the experiment.
        model_name: Name of the model being used.
        resume: If ``True``, resume from the previous run in ``output_directory``
            with the same experiment name. If ``False``, create new trials
            ignoring any prior state. Defaults to resuming when a matching
            experiment exists, creating new trials otherwise.
        skip_save_training_description: Disable saving the description JSON
            file.
        skip_save_training_statistics: Disable saving training statistics
            JSON file.
        skip_save_model: Disable saving model weights after each epoch the
            validation metric improves. The returned model will have weights
            from the final epoch rather than the best epoch.
        skip_save_progress: Disable saving weights and stats after each epoch
            (disables training resumption).
        skip_save_log: Disable saving TensorBoard logs.
        skip_save_processed_input: Disable caching preprocessed input as
            HDF5/JSON files.
        skip_save_unprocessed_output: If ``True``, skip saving raw numpy
            output files; only postprocessed CSV files are saved.
        skip_save_predictions: Disable saving test prediction CSV files.
        skip_save_eval_stats: Disable saving test statistics JSON file.
        skip_save_hyperopt_statistics: Disable saving hyperopt stats file.
        output_directory: Directory that will contain training statistics,
            TensorBoard logs, the saved model, and training progress files.
        gpus: List of GPUs available for training.
        gpu_memory_limit: Maximum memory fraction ``[0, 1]`` allowed to
            allocate per GPU device.
        allow_parallel_threads: Allow PyTorch to use multithreading
            parallelism (improves performance at the cost of determinism).
        callbacks: List of ``Callback`` objects providing hooks into the
            Ludwig pipeline.
        tune_callbacks: Additional Ray Tune callbacks.
        backend: Backend or string name of the backend to use for
            preprocessing and training.
        random_seed: Random seed for weights initialization, splits, and
            shuffling.
        hyperopt_log_verbosity: Verbosity of Ray Tune log messages.
            0 = silent, 1 = status only, 2 = status + brief results,
            3 = status + detailed results.

    Returns:
        Trial results ordered by descending performance on the target metric.
    """
    from ludwig.hyperopt.execution import get_build_hyperopt_executor, RayTuneExecutor

    # check if config is a path or a dict
    if isinstance(config, str):  # assume path
        with open_file(config, "r") as def_file:
            config_dict = yaml.safe_load(def_file)
    else:
        config_dict = config

    if HYPEROPT not in config_dict:
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

    # Explicitly default to a local backend to avoid picking up Ray
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
                f'The data for the specified split for hyperopt "{split}" '
                "was not provided, "
                "or the split amount specified in the preprocessing section "
                "of the config is not greater than 0"
            )
    elif split == VALIDATION:
        if validation_set is None and not splitter.has_split(1):
            raise ValueError(
                f'The data for the specified split for hyperopt "{split}" '
                "was not provided, "
                "or the split amount specified in the preprocessing section "
                "of the config is not greater than 0"
            )
    elif split == TEST:
        if test_set is None and not splitter.has_split(2):
            raise ValueError(
                f'The data for the specified split for hyperopt "{split}" '
                "was not provided, "
                "or the split amount specified in the preprocessing section "
                "of the config is not greater than 0"
            )
    else:
        raise ValueError(
            f'unrecognized hyperopt split "{split}". Please provide one of: { ({TRAINING, VALIDATION, TEST}) }'
        )
    if output_feature == COMBINED:
        if metric != LOSS:
            raise ValueError('The only valid metric for "combined" output feature is "loss"')
    else:
        output_feature_names = {of[NAME] for of in full_config[OUTPUT_FEATURES]}
        if output_feature not in output_feature_names:
            raise ValueError(
                f'The output feature specified for hyperopt "{output_feature}" '
                "cannot be found in the config. "
                f'Available ones are: {output_feature_names} and "combined"'
            )

    hyperopt_executor = get_build_hyperopt_executor(executor[TYPE])(
        parameters, output_feature, metric, goal, split, search_alg=search_alg, **executor
    )

    # Explicitly default to a local backend to avoid picking up Ray
    # backend from the environment.
    backend = backend or config_dict.get("backend") or "local"
    backend = initialize_backend(backend)
    from ludwig.hyperopt.optuna_executor import OptunaExecutor

    if not (
        isinstance(backend, LocalBackend)
        or isinstance(hyperopt_executor, OptunaExecutor)
        or (isinstance(hyperopt_executor, RayTuneExecutor) and isinstance(backend, RayBackend))
    ):
        raise ValueError(
            "Hyperopt requires using a `local` backend at this time, or "
            "`ray` backend with `ray` executor, or `optuna` executor."
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

        preprocessed = model.preprocess(
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            training_set_metadata=training_set_metadata,
            data_format=data_format,
            skip_save_processed_input=skip_save_processed_input,
            random_seed=random_seed,
        )
        training_set = preprocessed.training_set
        validation_set = preprocessed.validation_set
        test_set = preprocessed.test_set
        training_set_metadata = preprocessed.training_set_metadata
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
