import copy
import logging
import os

import tensorflow as tf
import yaml

from ludwig.data.preprocessing import load_metadata
from ludwig.globals import set_disable_progressbar, \
    MODEL_HYPERPARAMETERS_FILE_NAME, MODEL_WEIGHTS_FILE_NAME, \
    TRAIN_SET_METADATA_FILE_NAME
from ludwig.models.ecd import ECD
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.utils.horovod_utils import should_use_horovod
from ludwig.utils.tf_utils import initialize_tensorflow


class NewLudwigModel:

    def __init__(self,
                 model_definition=None,
                 model_definition_file=None,
                 logging_level=logging.ERROR,
                 use_horovod=False,
                 gpus=None,
                 gpu_memory_limit=None,
                 allow_parallel_threads=True,
                 random_seed=default_random_seed):
        # check for model_definition and model_definition_file
        if model_definition is None and model_definition_file is None:
            raise ValueError(
                'Either model_definition of model_definition_file have to be'
                'not None to initialize a LudwigModel'
            )
        if model_definition is not None and model_definition_file is not None:
            raise ValueError(
                'Only one between model_definition and '
                'model_definition_file can be provided'
            )

        # merge model definition with defaults
        if model_definition_file is not None:
            with open(model_definition_file, 'r') as def_file:
                self.model_definition = merge_with_defaults(
                    yaml.safe_load(def_file)
                )
        else:
            model_definition_copy = copy.deepcopy(model_definition)
            self.model_definition = merge_with_defaults(model_definition_copy)

        # setup horovod
        self._horovod = None
        if should_use_horovod(use_horovod):
            import horovod.tensorflow
            self._horovod = horovod.tensorflow
            self._horovod.init()

        # setup logging
        self.set_logging_level(logging_level)

        # setup TensorFlow
        initialize_tensorflow(gpus, gpu_memory_limit, allow_parallel_threads,
                              self._horovod)
        tf.random.set_seed(random_seed)

        # setup model
        self.model = ECD(
            input_features_def=model_definition['input_features'],
            combiner_def=model_definition['combiner'],
            output_features_def=model_definition['output_features'],
        )

        self.train_set_metadata = None
        self.exp_dir_name = ''

    def train(self, data, training_params):
        preproc_data = preprocess_data(data)
        trainer = Trainer(training_aprams)
        training_stats = trainer.train(self.model, preproc_data)
        return training_stats

    def predict(self, data):
        preproc_data = preprocess_data(data)
        preds = self.model.batch_predict(preproc_data)
        postproc_preds = postprocess_data(preds)
        return postproc_preds

    def evaluate(self, data, return_preds=False):
        preproc_data = preprocess_data(data)
        if return_preds:
            eval_stats, preds = self.model.batch_evaluate(
                preproc_data, return_preds=return_preds
            )
            postproc_preds = postprocess_data(preds)
            return eval_stats, postproc_preds
        else:
            eval_stats = self.model.batch_evaluate(
                preproc_data, return_preds=return_preds
            )
            return eval_stats

    @staticmethod
    def load(model_dir,
             logging_level=logging.ERROR,
             use_horovod=False,
             gpus=None,
             gpu_memory_limit=None,
             allow_parallel_threads=True):
        """This function allows for loading pretrained models


        # Inputs

        :param model_dir: (string) path to the directory containing the model.
               If the model was trained by the `train` or `experiment` command,
               the model is in `results_dir/experiment_dir/model`.
        :param gpus: (string, default: `None`) list of GPUs to use (it uses the
               same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_memory_limit: (int: default: `None`) maximum memory in MB to allocate
              per GPU device.
        :param allow_parallel_threads: (bool, default: `True`) allow TensorFlow to use
               multithreading parallelism to improve performance at the cost of
               determinism.

        # Return

        :return: (LudwigModel) a LudwigModel object


        # Example usage

        ```python
        ludwig_model = LudwigModel.load(model_dir)
        ```

        """
        # load model definition
        model_definition = load_json(
            os.path.join(
                model_dir,
                MODEL_HYPERPARAMETERS_FILE_NAME
            )
        )

        # initialize model
        ludwig_model = NewLudwigModel(
            model_definition,
            logging_level=logging_level,
            use_horovod=use_horovod,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
        )

        # load model weights
        weights_save_path = os.path.join(
            model_dir,
            MODEL_WEIGHTS_FILE_NAME
        )
        ludwig_model.model.load_weights(weights_save_path)

        # load train set metadata
        ludwig_model.train_set_metadata = load_metadata(
            os.path.join(
                model_dir,
                TRAIN_SET_METADATA_FILE_NAME
            )
        )

        return ludwig_model

    def save(self, save_path):
        """This function allows to save models on disk

        # Inputs

        :param  save_path: (string) path to the directory where the model is
                going to be saved. Both a JSON file containing the model
                architecture hyperparameters and checkpoints files containing
                model weights will be saved.


        # Example usage

        ```python
        ludwig_model.save(save_path)
        ```

        """
        if (self.model is None
                or self.model_definition is None
                or self.train_set_metadata is None):
            raise ValueError('Model has not been initialized or loaded')

        # save model definition
        model_hyperparameters_path = os.path.join(
            save_path,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )
        self.model.save_definition(
            model_hyperparameters_path
        )

        # save model weights
        model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        self.model.model.save_weights(model_weights_path)

        # save training set metadata
        train_set_metadata_path = os.path.join(
            save_path,
            TRAIN_SET_METADATA_FILE_NAME
        )
        save_json(train_set_metadata_path, self.train_set_metadata)

    def save_for_serving(self, save_path):
        """This function allows to save models on disk

        # Inputs

        :param  save_path: (string) path to the directory where the SavedModel
                is going to be saved.


        # Example usage

        ```python
        ludwig_model.save_for_serving(save_path)
        ```

        """
        if (self.model is None or self.model._session is None or
                self.model_definition is None or self.train_set_metadata is None):
            raise ValueError('Model has not been initialized or loaded')

        self.model.save_savedmodel(save_path)

    @staticmethod
    def set_logging_level(logging_level):
        """
        :param logging_level: Set/Update the logging level. Use logging
        constants like `logging.DEBUG` , `logging.INFO` and `logging.ERROR`.

        :return: None
        """
        logging.getLogger('ludwig').setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)
        else:
            set_disable_progressbar(False)
