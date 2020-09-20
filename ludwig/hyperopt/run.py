import logging
import os
from pprint import pformat

import yaml

from ludwig.constants import HYPEROPT, TRAINING, VALIDATION, TEST, COMBINED, \
    LOSS, TYPE
from ludwig.features.feature_registries import output_type_registry
from ludwig.hyperopt.execution import get_build_hyperopt_executor
from ludwig.hyperopt.sampling import get_build_hyperopt_sampler
from ludwig.hyperopt.utils import update_hyperopt_params_with_defaults, \
    print_hyperopt_results, save_hyperopt_stats
from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.utils.horovod_utils import is_on_master
from ludwig.utils.misc_utils import get_from_registry

logger = logging.getLogger(__name__)


def hyperopt(
        model_definition,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        data_format=None,
        experiment_name="hyperopt",
        model_name="run",
        # model_load_path=None,
        # model_resume_path=None,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=False,  # False because want use model best validation
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
        skip_save_unprocessed_output=True,
        skip_save_predictions=True,
        skip_save_eval_stats=True,
        skip_save_hyperopt_statistics=False,
        output_directory="results",
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        use_horovod=None,
        random_seed=default_random_seed,
        debug=False,
        **kwargs,
) -> dict:
    """This method performs an hyperparameter optimization.

    :param model_definition:
    :param dataset:
    :param training_set:
    :param validation_set:
    :param test_set:
    :param training_set_metadata:
    :param data_format:
    :param experiment_name:
    :param model_name:
    :param skip_save_training_description:
    :param skip_save_training_statistics:
    :param skip_save_model:
    :param skip_save_progress:
    :param skip_save_log:
    :param skip_save_processed_input:
    :param skip_save_unprocessed_output:
    :param skip_save_predictions:
    :param skip_save_eval_stats:
    :param skip_save_hyperopt_statistics:
    :param output_directory:
    :param gpus:
    :param gpu_memory_limit:
    :param allow_parallel_threads:
    :param use_horovod:
    :param random_seed:
    :param debug:
    :param kwargs:
    :return: (dict) The results fo the hyperparameter optimization
    """
    # todo refactoring: complete docstrings
    # check if model definition is a path or a dict
    if isinstance(model_definition, str):  # assume path
        with open(model_definition, 'r') as def_file:
            model_definition_dict = yaml.safe_load(def_file)
    else:
        model_definition_dict = model_definition

    # merge model definition with defaults
    model_definition = merge_with_defaults(model_definition_dict)

    if HYPEROPT not in model_definition:
        raise ValueError(
            "Hyperopt Section not present in Model Definition"
        )

    hyperopt_config = model_definition["hyperopt"]
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
        if not training_set and (
                model_definition['preprocessing']['split_probabilities'][0]
                <= 0):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                'was not provided, '
                'or the split amount specified in the preprocessing section '
                'of the model definition is not greater than 0'.format(split)
            )
    elif split == VALIDATION:
        if not validation_set and (
                model_definition['preprocessing']['split_probabilities'][1]
                <= 0):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                'was not provided, '
                'or the split amount specified in the preprocessing section '
                'of the model definition is not greater than 0'.format(split)
            )
    elif split == TEST:
        if not test_set and (
                model_definition['preprocessing']['split_probabilities'][2]
                <= 0):
            raise ValueError(
                'The data for the specified split for hyperopt "{}" '
                'was not provided, '
                'or the split amount specified in the preprocessing section '
                'of the model definition is not greater than 0'.format(split)
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
            of['name'] for of in model_definition['output_features']
        )
        if output_feature not in output_feature_names:
            raise ValueError(
                'The output feature specified for hyperopt "{}" '
                'cannot be found in the model definition. '
                'Available ones are: {} and "combined"'.format(
                    output_feature, output_feature_names
                )
            )

        output_feature_type = None
        for of in model_definition['output_features']:
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
        model_definition,
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
        use_horovod=use_horovod,
        random_seed=random_seed,
        debug=debug,
        **kwargs
    )

    if is_on_master():
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
