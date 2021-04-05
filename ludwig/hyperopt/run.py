import logging
import os
from pprint import pformat
from typing import Union, List

import pandas as pd
import yaml

from ludwig.backend import Backend, initialize_backend
from ludwig.constants import HYPEROPT, TRAINING, VALIDATION, TEST, COMBINED, \
    LOSS, TYPE, RAY
from ludwig.features.feature_registries import output_type_registry
from ludwig.hyperopt.execution import get_build_hyperopt_executor
from ludwig.hyperopt.sampling import get_build_hyperopt_sampler
from ludwig.hyperopt.utils import update_hyperopt_params_with_defaults, \
    print_hyperopt_results, save_hyperopt_stats
from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)


def hyperopt(
        config: Union[str, dict],
        dataset: Union[str, dict, pd.DataFrame] = None,
        training_set: Union[str, dict, pd.DataFrame] = None,
        validation_set: Union[str, dict, pd.DataFrame] = None,
        test_set: Union[str, dict, pd.DataFrame] = None,
        training_set_metadata: Union[str, dict] = None,
        data_format: str = None,
        experiment_name: str = 'hyperopt',
        model_name: str = 'run',
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
        output_directory: str = 'results',
        gpus: Union[str, int, List[int]] = None,
        gpu_memory_limit: int = None,
        allow_parallel_threads: bool = True,
        backend: Union[Backend, str] = None,
        random_seed: int = default_random_seed,
        debug: bool = False,
        **kwargs,
) -> List[dict]:
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
    :param gpu_memory_limit: (int, default: `None`) maximum memory in MB to
        allocate per GPU device.
    :param allow_parallel_threads: (bool, default: `True`) allow TensorFlow
        to use multithreading parallelism to improve performance at
        the cost of determinism.
    :param backend: (Union[Backend, str]) `Backend` or string name
        of backend to use to execute preprocessing / training steps.
    :param random_seed: (int: default: 42) random seed used for weights
        initialization, splits and any other random function.
    :param debug: (bool, default: `False) if `True` turns on `tfdbg` with
        `inf_or_nan` checks.

    # Return

    :return: (List[dict]) The results for the hyperparameter optimization
    """
    backend = initialize_backend(backend)

    # check if config is a path or a dict
    if isinstance(config, str):  # assume path
        with open(config, 'r') as def_file:
            config_dict = yaml.safe_load(def_file)
    else:
        config_dict = config

    # merge config with defaults
    config = merge_with_defaults(config_dict)

    if HYPEROPT not in config:
        raise ValueError(
            "Hyperopt Section not present in config"
        )

    hyperopt_config = config["hyperopt"]

    update_hyperopt_params_with_defaults(hyperopt_config)

    # print hyperopt config
    logger.info(pformat(hyperopt_config, indent=4))
    logger.info('\n')

    sampler = hyperopt_config["sampler"]
    executor = hyperopt_config["executor"]
    parameters = hyperopt_config["parameters"]
    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]

    ######################
    # check validity of output_feature / metric/ split combination
    ######################
    if split == TRAINING:
        if training_set is None and (
                config['preprocessing']['split_probabilities'][0]
                <= 0):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                'was not provided, '
                'or the split amount specified in the preprocessing section '
                'of the config is not greater than 0'.format(split)
            )
    elif split == VALIDATION:
        if validation_set is None and (
                config['preprocessing']['split_probabilities'][1]
                <= 0):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                'was not provided, '
                'or the split amount specified in the preprocessing section '
                'of the config is not greater than 0'.format(split)
            )
    elif split == TEST:
        if test_set is None and (
                config['preprocessing']['split_probabilities'][2]
                <= 0):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                'was not provided, '
                'or the split amount specified in the preprocessing section '
                'of the config is not greater than 0'.format(split)
            )
    else:
        raise ValueError(
            'unrecognized hyperopt split "{}". '
            'Please provide one of: {}'.format(
                split, {TRAINING, VALIDATION, TEST}
            )
        )
    if output_feature == COMBINED:
        if metric != LOSS:
            raise ValueError(
                'The only valid metric for "combined" output feature is "loss"'
            )
    else:
        output_feature_names = set(
            of['name'] for of in config['output_features']
        )
        if output_feature not in output_feature_names:
            raise ValueError(
                'The output feature specified for hyperopt "{}" '
                'cannot be found in the config. '
                'Available ones are: {} and "combined"'.format(
                    output_feature, output_feature_names
                )
            )

        output_feature_type = None
        for of in config['output_features']:
            if of['name'] == output_feature:
                output_feature_type = of[TYPE]
        feature_class = get_from_registry(
            output_feature_type,
            output_type_registry
        )
        if metric not in feature_class.metric_functions:
            # todo v0.4: allow users to specify also metrics from the overall
            #  and per class metrics from the trainign stats and in general
            #  and potprocessed metric
            raise ValueError(
                'The specified metric for hyperopt "{}" is not a valid metric '
                'for the specified output feature "{}" of type "{}". '
                'Available metrics are: {}'.format(
                    metric,
                    output_feature,
                    output_feature_type,
                    feature_class.metric_functions.keys()
                )
            )

    hyperopt_sampler = get_build_hyperopt_sampler(
        sampler[TYPE]
    )(goal, parameters, **sampler)

    hyperopt_executor = get_build_hyperopt_executor(
        executor[TYPE]
    )(hyperopt_sampler, output_feature, metric, split, **executor)

    hyperopt_results = hyperopt_executor.execute(
        config,
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        experiment_name=experiment_name,
        model_name=model_name,
        # model_load_path=None,
        # model_resume_path=None,
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
        backend=backend,
        random_seed=random_seed,
        debug=debug,
        **kwargs
    )

    if backend.is_coordinator():
        print_hyperopt_results(hyperopt_results)

        if not skip_save_hyperopt_statistics:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            hyperopt_stats = {
                'hyperopt_config': hyperopt_config,
                'hyperopt_results': hyperopt_results
            }

            save_hyperopt_stats(hyperopt_stats, output_directory)
            logger.info('Hyperopt stats saved to: {}'.format(output_directory))

    logger.info('Finished hyperopt')

    return hyperopt_results
