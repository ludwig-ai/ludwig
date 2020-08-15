#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module contains the class and auxiliary methods of a model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import os
import os.path
import signal
import sys
import threading
import time
from collections import OrderedDict

import tensorflow as tf
from tabulate import tabulate
from tqdm import tqdm

from ludwig.constants import LOSS, COMBINED, TRAINING, VALIDATION, TEST
from ludwig.contrib import contrib_command
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.globals import TRAINING_CHECKPOINTS_DIR_PATH
from ludwig.globals import TRAINING_PROGRESS_TRACKER_FILE_NAME
from ludwig.globals import is_on_master
from ludwig.globals import is_progressbar_disabled
from ludwig.modules.metric_modules import get_improved_fun
from ludwig.modules.metric_modules import get_initial_validation_value
from ludwig.modules.optimization_modules import ClippedOptimizer
from ludwig.utils import time_utils
from ludwig.utils.batcher import initialize_batcher
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.horovod_utils import allgather_object
from ludwig.utils.math_utils import learning_rate_warmup, \
    learning_rate_warmup_distributed
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.misc_utils import sum_dicts

logger = logging.getLogger(__name__)

tf.config.experimental_run_functions_eagerly(True)


class Trainer:
    """
    Model is a class that builds the model that Ludwig uses
    """

    def __init__(
            self,
            optimizer={TYPE: 'Adam'},
            epochs=100,
            regularization_lambda=0.0,
            learning_rate=0.001,
            batch_size=128,
            eval_batch_size=0,
            bucketing_field=None,
            validation_field='combined',
            validation_metric='loss',
            early_stop=20,
            reduce_learning_rate_on_plateau=0,
            reduce_learning_rate_on_plateau_patience=5,
            reduce_learning_rate_on_plateau_rate=0.5,
            increase_batch_size_on_plateau=0,
            increase_batch_size_on_plateau_patience=5,
            increase_batch_size_on_plateau_rate=2,
            increase_batch_size_on_plateau_max=512,
            learning_rate_warmup_epochs=1,
            resume=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            random_seed=default_random_seed,
            horovod=None,
            debug=False,
            **kwargs
    ):
        """Trains a model with a set of hyperparameters listed below. Customizable
                :param training_set: The training set
        :param validation_set: The validation dataset
        :param test_set: The test dataset
        :param validation_field: The first output feature, by default it is set
               as the same field of the first output feature.
        :param validation_metric: metric used on the validation field, it is
               accuracy by default
        :type validation_metric:
        :param save_path: The path to save the file
        :type save_path: filepath (str)
        :param regularization_lambda: Strength of the $L2$ regularization
        :type regularization_lambda: Integer
        :param epochs: Number of epochs the algorithm is intended to be run over
        :type epochs: Integer
        :param learning_rate: Learning rate for the algorithm, represents how
               much to scale the gradients by
        :type learning_rate: Integer
        :param batch_size: Size of batch to pass to the model for training.
        :type batch_size: Integer
        :param batch_size: Size of batch to pass to the model for evaluation.
        :type batch_size: Integer
        :param bucketing_field: when batching, buckets datapoints based the
               length of a field together. Bucketing on text length speeds up
               training of RNNs consistently, 30% in some cases
        :type bucketing_field:
        :param validation_field: The first output feature, by default it is set
               as the same field of the first output feature.
        :param validation_metric: metric used on the validation field, it is
               accuracy by default
        :type validation_metric:
        :param dropout_rate: dropout_rate probability (probability of dropping
               a neuron in a given layer)
        :type dropout_rate: Float
        :param early_stop: How many epochs without any improvement in the
               validation_metric triggers the algorithm to stop
        :type early_stop: Integer
        :param reduce_learning_rate_on_plateau: Reduces the learning rate when
               the algorithm hits a plateau (i.e. the performance on the
               validation does not improve)
        :type reduce_learning_rate_on_plateau: Float
        :param reduce_learning_rate_on_plateau_patience: How many epochs have
               to pass before the learning rate reduces
        :type reduce_learning_rate_on_plateau_patience: Float
        :param reduce_learning_rate_on_plateau_rate: Rate at which we reduce
               the learning rate
        :type reduce_learning_rate_on_plateau_rate: Float
        :param increase_batch_size_on_plateau: Increase the batch size on a
               plateau
        :type increase_batch_size_on_plateau: Integer
        :param increase_batch_size_on_plateau_patience: How many epochs to wait
               for before increasing the batch size
        :type increase_batch_size_on_plateau_patience: Integer
        :param increase_batch_size_on_plateau_rate: The rate at which the batch
               size increases.
        :type increase_batch_size_on_plateau_rate: Float
        :param increase_batch_size_on_plateau_max: The maximum size of the batch
        :type increase_batch_size_on_plateau_max: Integer
        :param learning_rate_warmup_epochs: The number of epochs to warmup the
               learning rate for.
        :type learning_rate_warmup_epochs: Integer
        :param resume: Resume training a model that was being trained.
        :type resume: Boolean
        :param skip_save_model: disables
               saving model weights and hyperparameters each time the model
               improves. By default Ludwig saves model weights after each epoch
               the validation metric imrpvoes, but if the model is really big
               that can be time consuming if you do not want to keep
               the weights and just find out what performance can a model get
               with a set of hyperparameters, use this parameter to skip it,
               but the model will not be loadable later on.
        :type skip_save_model: Boolean
        :param skip_save_progress: disables saving progress each epoch.
               By default Ludwig saves weights and stats  after each epoch
               for enabling resuming of training, but if the model is
               really big that can be time consuming and will uses twice
               as much space, use this parameter to skip it, but training
               cannot be resumed later on
        :type skip_save_progress: Boolean
        :param skip_save_log: Disables saving TensorBoard
               logs. By default Ludwig saves logs for the TensorBoard, but if it
               is not needed turning it off can slightly increase the
               overall speed..
        :type skip_save_log: Boolean
        :param random_seed: Default initialization for the random seeds
        :type: Float
        """
        self._epochs = epochs
        self._regularization_lambda = regularization_lambda
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._eval_batch_size = batch_size if eval_batch_size < 1 else eval_batch_size
        self._bucketing_field = bucketing_field
        self._validation_field = validation_field
        self._validation_metric = validation_metric
        self._early_stop = early_stop
        self._reduce_learning_rate_on_plateau = reduce_learning_rate_on_plateau
        self._reduce_learning_rate_on_plateau_patience = reduce_learning_rate_on_plateau_patience
        self._reduce_learning_rate_on_plateau_rate = reduce_learning_rate_on_plateau_rate
        self._increase_batch_size_on_plateau = increase_batch_size_on_plateau
        self._increase_batch_size_on_plateau_patience = increase_batch_size_on_plateau_patience
        self._increase_batch_size_on_plateau_rate = increase_batch_size_on_plateau_rate
        self._increase_batch_size_on_plateau_max = increase_batch_size_on_plateau_max
        self._learning_rate_warmup_epochs = learning_rate_warmup_epochs
        self._resume = resume
        self._skip_save_model = skip_save_model
        self._skip_save_progress = skip_save_progress
        self._skip_save_log = skip_save_log
        self._random_seed = random_seed
        self._horovod = horovod
        self._debug = debug
        self._received_sigint = False

        if self._horovod:
            self._learning_rate *= self._horovod.size()

        # ================ Optimizer ================
        self._optimizer = ClippedOptimizer(
            horovod=horovod,
            **optimizer
        )

    @classmethod
    def write_epoch_summary(
            cls,
            summary_writer,
            metrics,
            step
    ):
        if not summary_writer:
            return

        with summary_writer.as_default():
            for feature_name, output_feature in metrics.items():
                for metric in output_feature:
                    metric_tag = "{}/epoch_{}".format(
                        feature_name, metric
                    )
                    metric_val = output_feature[metric][-1]
                    tf.summary.scalar(metric_tag, metric_val, step=step)
        summary_writer.flush()

    @classmethod
    def write_step_summary(
            cls,
            train_summary_writer,
            combined_loss,
            all_losses,
            step
    ):
        if not train_summary_writer:
            return

        with train_summary_writer.as_default():
            # combined loss
            loss_tag = "{}/step_training_loss".format("combined")
            tf.summary.scalar(loss_tag, combined_loss, step=step)

            # all other losses
            for feature_name, loss in all_losses.items():
                loss_tag = "{}/step_training_loss".format(feature_name)
                tf.summary.scalar(loss_tag, loss, step=step)

        train_summary_writer.flush()

    def train(
            self,
            model,
            training_set,
            validation_set=None,
            test_set=None,
            save_path='model',
            **kwargs
    ):
        """Trains a model with a set of hyperparameters listed below. Customizable
        :param training_set: The training set
        :param validation_set: The validation dataset
        :param test_set: The test dataset
        """
        # ====== General setup =======
        tf.random.set_seed(self._random_seed)

        output_features = model.output_features
        digits_per_epochs = len(str(self._epochs))
        # Only use signals when on the main thread to avoid issues with CherryPy: https://github.com/uber/ludwig/issues/286
        if threading.current_thread() == threading.main_thread():
            signal.signal(signal.SIGINT, self.set_epochs_to_1_or_quit)
        should_validate = validation_set is not None and validation_set.size > 0

        metrics_names = self.get_metrics_names(output_features)

        # check if validation_field is valid
        valid_validation_field = False
        validation_output_feature_name = None
        if self._validation_field == 'combined':
            valid_validation_field = True
            validation_output_feature_name = 'combined'
            if self._validation_metric is not LOSS and len(
                    output_features) == 1:
                only_of = output_features[0]
                if self._validation_metric in metrics_names[only_of]:
                    validation_output_feature_name = only_of
                    logger.warning(
                        "Replacing 'combined' validation field "
                        "with '{}' as the specified validation "
                        "metric {} is invalid for 'combined' "
                        "but is valid for '{}'.".format(
                            only_of, self._validation_metric, only_of
                        ))
        else:
            for output_feature in output_features:
                if self._validation_field == output_feature:
                    valid_validation_field = True
                    validation_output_feature_name = self._validation_field
        if not valid_validation_field:
            raise ValueError(
                'The specificed validation_field {} is not valid.'
                'Available ones are: {}'.format(
                    self._validation_field,
                    [of['name'] for of in output_features] + ['combined']
                )
            )

        # check if validation_metric is valid
        valid_validation_metric = self._validation_metric in metrics_names[
            validation_output_feature_name
        ]
        if not valid_validation_metric:
            raise ValueError(
                'The specificed metric {} is not valid. '
                'Available metrics for {} output feature are: {}'.format(
                    validation_metric,
                    validation_output_feature_name,
                    metrics_names[validation_output_feature_name]
                )
            )

        # ====== Setup file names =======
        model_weights_path = model_hyperparameters_path = None
        training_checkpoints_path = training_checkpoints_prefix_path = training_progress_tracker_path = None
        tensorboard_log_dir = None
        if is_on_master():
            os.makedirs(save_path, exist_ok=True)
            model_weights_path = os.path.join(save_path,
                                              MODEL_WEIGHTS_FILE_NAME)
            model_hyperparameters_path = os.path.join(
                save_path, MODEL_HYPERPARAMETERS_FILE_NAME
            )
            training_checkpoints_path = os.path.join(
                save_path, TRAINING_CHECKPOINTS_DIR_PATH
            )
            # training_checkpoints_prefix_path = os.path.join(
            #    training_checkpoints_path, "ckpt"
            # )
            training_progress_tracker_path = os.path.join(
                save_path, TRAINING_PROGRESS_TRACKER_FILE_NAME
            )
            tensorboard_log_dir = os.path.join(
                save_path, 'logs'
            )

        # ====== Setup session =======
        checkpoint = checkpoint_manager = None
        if is_on_master():
            checkpoint = tf.train.Checkpoint(
                optimizer=self._optimizer,
                model=model
            )
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, training_checkpoints_path, max_to_keep=1
            )

        train_summary_writer = None
        validation_summary_writer = None
        test_summary_writer = None
        if is_on_master() and not self._skip_save_log and tensorboard_log_dir:
            train_summary_writer = tf.summary.create_file_writer(
                os.path.join(
                    tensorboard_log_dir, TRAINING
                )
            )
            if validation_set is not None and validation_set.size > 0:
                validation_summary_writer = tf.summary.create_file_writer(
                    os.path.join(
                        tensorboard_log_dir, VALIDATION
                    )
                )
            if test_set is not None and test_set.size > 0:
                test_summary_writer = tf.summary.create_file_writer(
                    os.path.join(
                        tensorboard_log_dir, TEST
                    )
                )

        if self._debug and is_on_master():
            # See https://www.tensorflow.org/tensorboard/debugger_v2 for usage.
            debug_path = os.path.join(
                save_path, 'debug'
            )
            tf.debugging.experimental.enable_dump_debug_info(
                debug_path,
                tensor_debug_mode='FULL_HEALTH',
                circular_buffer_size=-1,
            )

        # ================ Resume logic ================
        if self._resume:
            progress_tracker = self.resume_training_progress_tracker(
                training_progress_tracker_path
            )
            if is_on_master():
                self.resume_weights_and_optimzier(
                    training_checkpoints_path, checkpoint
                )
        else:
            (
                train_metrics,
                vali_metrics,
                test_metrics
            ) = self.initialize_training_metrics(output_features)

            progress_tracker = ProgressTracker(
                batch_size=self._batch_size,
                epoch=0,
                steps=0,
                last_improvement_epoch=0,
                learning_rate=self._learning_rate,
                best_valid_metric=get_initial_validation_value(
                    self._validation_metric
                ),
                num_reductions_lr=0,
                num_increases_bs=0,
                train_metrics=train_metrics,
                vali_metrics=vali_metrics,
                test_metrics=test_metrics
            )

        set_random_seed(self._random_seed)
        batcher = initialize_batcher(
            training_set, self._batch_size, self._bucketing_field,
            horovod=self._horovod
        )

        # ================ Training Loop ================
        first_batch = True
        while progress_tracker.epoch < self._epochs:
            # epoch init
            start_time = time.time()
            if is_on_master():
                logger.info(
                    '\nEpoch {epoch:{digits}d}'.format(
                        epoch=progress_tracker.epoch + 1,
                        digits=digits_per_epochs
                    )
                )
            # needed because batch size may change
            batcher.batch_size = progress_tracker.batch_size

            # Reset the metrics at the start of the next epoch
            model.reset_metrics()

            # ================ Train ================
            if is_on_master():
                progress_bar = tqdm(
                    desc='Training',
                    total=batcher.steps_per_epoch,
                    file=sys.stdout,
                    disable=is_progressbar_disabled()
                )

            # training step loop
            while not batcher.last_batch():
                batch = batcher.next_batch()
                inputs = {
                    i_feat.feature_name: batch[i_feat.feature_name]
                    for i_feat in model.input_features
                }
                targets = {
                    o_feat.feature_name: batch[o_feat.feature_name]
                    for o_feat in model.output_features
                }

                # Reintroduce for tensorboard graph
                # if first_batch and is_on_master() and not skip_save_log:
                #    tf.summary.trace_on(graph=True, profiler=True)

                loss, all_losses = model.train_step(
                    self._optimizer,
                    inputs,
                    targets,
                    self._regularization_lambda
                )

                # Reintroduce for tensorboard graph
                # if first_batch and is_on_master() and not skip_save_log:
                #     with train_summary_writer.as_default():
                #         tf.summary.trace_export(
                #             name="Model",
                #             step=0,
                #             profiler_outdir=tensorboard_log_dir
                #         )

                if is_on_master() and not self._skip_save_log:
                    self.write_step_summary(
                        train_summary_writer=train_summary_writer,
                        combined_loss=loss,
                        all_losses=all_losses,
                        step=progress_tracker.steps,
                    )

                if self._horovod and first_batch:
                    # Horovod: broadcast initial variable states from rank 0 to all other processes.
                    # This is necessary to ensure consistent initialization of all workers when
                    # training is started with random weights or restored from a checkpoint.
                    #
                    # Note: broadcast should be done after the first gradient step to ensure
                    # optimizer initialization.
                    self._horovod.broadcast_variables(model.variables,
                                                      root_rank=0)
                    self._horovod.broadcast_variables(
                        self._optimizer.variables(), root_rank=0)

                if self._horovod:
                    current_learning_rate = learning_rate_warmup_distributed(
                        progress_tracker.learning_rate,
                        progress_tracker.epoch,
                        self._learning_rate_warmup_epochs,
                        self._horovod.size(),
                        batcher.step,
                        batcher.steps_per_epoch
                    ) * self._horovod.size()
                else:
                    current_learning_rate = learning_rate_warmup(
                        progress_tracker.learning_rate,
                        progress_tracker.epoch,
                        self._learning_rate_warmup_epochs,
                        batcher.step,
                        batcher.steps_per_epoch
                    )
                self._optimizer.set_learning_rate(current_learning_rate)

                progress_tracker.steps += 1
                if is_on_master():
                    progress_bar.update(1)
                first_batch = False

            # ================ Post Training Epoch ================
            if is_on_master():
                progress_bar.close()

            progress_tracker.epoch += 1
            batcher.reset()  # todo this may be useless, doublecheck

            # ================ Eval ================
            # init tables
            tables = OrderedDict()
            for output_feature_name, output_feature in output_features.items():
                tables[output_feature_name] = [
                    [output_feature_name] + metrics_names[output_feature_name]
                ]
            tables[COMBINED] = [[COMBINED, LOSS]]

            # eval metrics on train
            self.evaluation(
                model,
                training_set,
                'train',
                progress_tracker.train_metrics,
                tables,
                self._eval_batch_size,
            )

            self.write_epoch_summary(
                summary_writer=train_summary_writer,
                metrics=progress_tracker.train_metrics,
                step=progress_tracker.epoch,
            )

            if validation_set is not None and validation_set.size > 0:
                # eval metrics on validation set
                self.evaluation(
                    model,
                    validation_set,
                    'vali',
                    progress_tracker.vali_metrics,
                    tables,
                    self._eval_batch_size,
                )

                self.write_epoch_summary(
                    summary_writer=validation_summary_writer,
                    metrics=progress_tracker.vali_metrics,
                    step=progress_tracker.epoch,
                )

            if test_set is not None and test_set.size > 0:
                # eval metrics on test set
                self.evaluation(
                    model,
                    test_set,
                    TEST,
                    progress_tracker.test_metrics,
                    tables,
                    self._eval_batch_size,
                )

                self.write_epoch_summary(
                    summary_writer=test_summary_writer,
                    metrics=progress_tracker.test_metrics,
                    step=progress_tracker.epoch,
                )

            elapsed_time = (time.time() - start_time) * 1000.0

            if is_on_master():
                logger.info('Took {time}'.format(
                    time=time_utils.strdelta(elapsed_time)))

            # metric prints
            if is_on_master():
                for output_feature, table in tables.items():
                    logger.info(
                        tabulate(
                            table,
                            headers='firstrow',
                            tablefmt='fancy_grid',
                            floatfmt='.4f'
                        )
                    )

            # ================ Validation Logic ================
            if should_validate:
                should_break = self.check_progress_on_validation(
                    model,
                    progress_tracker,
                    validation_output_feature_name,
                    self._validation_metric,
                    model_weights_path,
                    model_hyperparameters_path,
                    self._reduce_learning_rate_on_plateau,
                    self._reduce_learning_rate_on_plateau_patience,
                    self._reduce_learning_rate_on_plateau_rate,
                    self._increase_batch_size_on_plateau_patience,
                    self._increase_batch_size_on_plateau,
                    self._increase_batch_size_on_plateau_max,
                    self._increase_batch_size_on_plateau_rate,
                    self._early_stop,
                    self._skip_save_model
                )
                if should_break:
                    break
            else:
                # there's no validation, so we save the model at each iteration
                if is_on_master():
                    if not self._skip_save_model:
                        model.save_weights(model_weights_path)
                        # todo refactoring: this should become save trainer
                        self.save_hyperparameters(
                            self._hyperparameters,
                            model_hyperparameters_path
                        )

            # ========== Save training progress ==========
            if is_on_master():
                if not self._skip_save_progress:
                    checkpoint_manager.save()
                    progress_tracker.save(
                        os.path.join(
                            save_path,
                            TRAINING_PROGRESS_TRACKER_FILE_NAME
                        )
                    )
                    if self._skip_save_model:
                        # todo refactoring: this should become save trainer
                        self.save_hyperparameters(
                            self._hyperparameters,
                            model_hyperparameters_path
                        )

            if is_on_master():
                contrib_command("train_epoch_end", progress_tracker)
                logger.info('')

        if train_summary_writer is not None:
            train_summary_writer.close()
        if validation_summary_writer is not None:
            validation_summary_writer.close()
        if test_summary_writer is not None:
            test_summary_writer.close()

        return (
            progress_tracker.train_metrics,
            progress_tracker.vali_metrics,
            progress_tracker.test_metrics
        )

    def train_online(
            self,
            model,
            dataset,
            batch_size=128,
            regularization_lambda=0.0,
            bucketing_field=None
    ):
        batcher = initialize_batcher(
            dataset, batch_size, bucketing_field,
            horovod=self._horovod
        )

        # training step loop
        progress_bar = tqdm(
            desc='Trainining online',
            total=batcher.steps_per_epoch,
            file=sys.stdout,
            disable=is_progressbar_disabled()
        )

        while not batcher.last_batch():
            batch = batcher.next_batch()
            inputs = {
                i_feat.feature_name: batch[i_feat.feature_name]
                for i_feat in model.input_features
            }
            targets = {
                o_feat.feature_name: batch[o_feat.feature_name]
                for o_feat in model.output_features
            }

            model.train_step(
                self._optimizer,
                inputs,
                targets,
                regularization_lambda
            )

            progress_bar.update(1)

        progress_bar.close()

    def append_metrics(self, model, dataset_name, results, metrics_log,
                       tables):
        for output_feature in model.output_features:
            scores = [dataset_name]

            # collect metric names based on output features metrics to
            # ensure consistent order of reporting metrics
            metric_names = model.output_features[output_feature] \
                .metric_functions.keys()

            for metric in metric_names:
                score = results[output_feature][metric]
                metrics_log[output_feature][metric].append(score)
                scores.append(score)

            tables[output_feature].append(scores)

        metrics_log[COMBINED][LOSS].append(results[COMBINED][LOSS])
        tables[COMBINED].append([dataset_name, results[COMBINED][LOSS]])

        return metrics_log, tables

    def evaluation(
            self,
            model,
            dataset,
            dataset_name,
            metrics_log,
            tables,
            batch_size=128,
    ):
        results = model.batch_evaluation(
            dataset,
            batch_size,
            collect_predictions=False,
            dataset_name=dataset_name
        )

        self.append_metrics(model, dataset_name, results, metrics_log, tables)

        return metrics_log, tables

    def merge_workers_metrics(self, metrics):
        # gather outputs from all workers
        all_workers_output_metrics = allgather_object(metrics)

        # merge them into a single one
        merged_output_metrics = sum_dicts(
            all_workers_output_metrics,
            dict_type=OrderedDict
        )

        return merged_output_metrics

    def check_progress_on_validation(
            self,
            model,
            progress_tracker,
            validation_output_feature_name,
            validation_metric,
            model_weights_path,
            model_hyperparameters_path,
            reduce_learning_rate_on_plateau,
            reduce_learning_rate_on_plateau_patience,
            reduce_learning_rate_on_plateau_rate,
            increase_batch_size_on_plateau_patience,
            increase_batch_size_on_plateau,
            increase_batch_size_on_plateau_max,
            increase_batch_size_on_plateau_rate,
            early_stop,
            skip_save_model
    ):
        should_break = False
        # record how long its been since an improvement
        improved = get_improved_fun(validation_metric)
        if improved(
                progress_tracker.vali_metrics[validation_output_feature_name][
                    validation_metric][-1],
                progress_tracker.best_valid_metric
        ):
            progress_tracker.last_improvement_epoch = progress_tracker.epoch
            progress_tracker.best_valid_metric = progress_tracker.vali_metrics[
                validation_output_feature_name][validation_metric][-1]
            if is_on_master():
                if not skip_save_model:
                    model.save_weights(model_weights_path)
                    # todo refactoring: this should become save trainer
                    self.save_hyperparameters(
                        self._hyperparameters,
                        model_hyperparameters_path
                    )
                    logger.info(
                        'Validation {} on {} improved, model saved'.format(
                            validation_metric,
                            validation_output_feature_name
                        )
                    )

        progress_tracker.last_improvement = (
                progress_tracker.epoch - progress_tracker.last_improvement_epoch
        )
        if progress_tracker.last_improvement != 0:
            if is_on_master():
                logger.info(
                    'Last improvement of {} on {} happened '
                    '{} epoch{} ago'.format(
                        validation_metric,
                        validation_output_feature_name,
                        progress_tracker.last_improvement,
                        '' if progress_tracker.last_improvement == 1 else 's'
                    )
                )

        # ========== Reduce Learning Rate Plateau logic ========
        if reduce_learning_rate_on_plateau > 0:
            self.reduce_learning_rate(
                progress_tracker,
                reduce_learning_rate_on_plateau,
                reduce_learning_rate_on_plateau_patience,
                reduce_learning_rate_on_plateau_rate
            )

        # ========== Increase Batch Size Plateau logic =========
        if increase_batch_size_on_plateau > 0:
            self.increase_batch_size(
                progress_tracker,
                increase_batch_size_on_plateau_patience,
                increase_batch_size_on_plateau,
                increase_batch_size_on_plateau_max,
                increase_batch_size_on_plateau_rate
            )

        # ========== Early Stop logic ==========
        if early_stop > 0:
            if progress_tracker.last_improvement >= early_stop:
                if is_on_master():
                    logger.info(
                        "\nEARLY STOPPING due to lack of validation improvement"
                        ", it has been {0} epochs since last validation "
                        "accuracy improvement\n".format(
                            progress_tracker.epoch -
                            progress_tracker.last_improvement_epoch
                        )
                    )
                should_break = True
        return should_break

    # todo refactoring: should use ECD.batch_predict
        # def predict(
        #         self,
        #         dataset,
        #         batch_size,
        #         **kwargs
        # ):
        #     # predict
        #     return self.model.batch_predict(dataset, batch_size,
        #                                     horovod=self._horovod)

        return collected_tensors

    def collect_weights(
            self,
            tensor_names=None,
            **kwargs
    ):
        def recurse_weights(model, prefix=None):
            results = []
            for layer in model.layers:
                layer_prefix = f'{prefix}/{layer.name}' if prefix else layer.name
                if isinstance(layer, tf.keras.Model):
                    results += recurse_weights(layer, layer_prefix)
                else:
                    results += [(f'{layer_prefix}/{w.name}', w) for w in
                                layer.weights]
            return results

        weights = recurse_weights(self.model)
        if tensor_names:
            # Check for bad tensor names
            weight_set = set(name for name, w in weights)
            for name in tensor_names:
                if name not in weight_set:
                    raise ValueError(
                        f'Tensor {name} not present in the model graph')

            # Filter the weights
            tensor_set = set(tensor_names)
            weights = [(name, w) for name, w in weights if name in tensor_set]
        return weights

    def save_weights(self, save_path):
        # save model
        self.model.save_weights(save_path)

    # todo refactoring: should change
    def save_hyperparameters(self, hyperparameters, save_path):
        # removing pretrained embeddings paths from hyperparameters
        # because the weights are already saved in the model, no need to reload
        # from their path when loading the model next time

        local_hyperparamters = copy.deepcopy(hyperparameters)
        for feature in (local_hyperparamters['input_features'] +
                        local_hyperparamters['output_features']):
            if 'pretrained_embeddings' in feature:
                feature['pretrained_embeddings'] = None
        save_json(save_path, hyperparameters, sort_keys=True, indent=4)

    # todo refactoring: should change
    def restore(self, weights_path):
        self.model.load_weights(weights_path)

    # todo refactoring: should change
    @staticmethod
    def load(load_path, use_horovod=None, gpus=None, gpu_memory_limit=None,
             allow_parallel_threads=True):
        hyperparameter_file = os.path.join(
            load_path,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )
        hyperparameters = load_json(hyperparameter_file)
        trainer = Trainer(use_horovod=use_horovod,
                          gpus=gpus,
                          gpu_memory_limit=gpu_memory_limit,
                          allow_parallel_threads=allow_parallel_threads,
                          **hyperparameters)
        weights_save_path = os.path.join(
            load_path,
            MODEL_WEIGHTS_FILE_NAME
        )
        trainer.restore(weights_save_path)
        return trainer

    def set_epochs_to_1_or_quit(self, signum, frame):
        if not self._received_sigint:
            self._epochs = 1
            self._received_sigint = True
            logger.critical(
                '\nReceived SIGINT, will finish this epoch and then conclude '
                'the training'
            )
            logger.critical(
                'Send another SIGINT to immediately interrupt the process'
            )
        else:
            logger.critical('\nReceived a second SIGINT, will now quit')
            sys.exit(1)

    def quit_training(self, signum, frame):
        logger.critical('Received SIGQUIT, will kill training')
        sys.exit(1)

    def resume_training_progress_tracker(self, training_progress_tracker_path):
        if is_on_master():
            logger.info('Resuming training of model: {0}'.format(
                training_progress_tracker_path
            ))
        progress_tracker = ProgressTracker.load(training_progress_tracker_path)
        return progress_tracker

    def initialize_training_metrics(self, output_features):
        train_metrics = OrderedDict()
        vali_metrics = OrderedDict()
        test_metrics = OrderedDict()

        for output_feature_name, output_feature in output_features.items():
            train_metrics[output_feature_name] = OrderedDict()
            vali_metrics[output_feature_name] = OrderedDict()
            test_metrics[output_feature_name] = OrderedDict()
            for metric in output_feature.metric_functions:
                train_metrics[output_feature_name][metric] = []
                vali_metrics[output_feature_name][metric] = []
                test_metrics[output_feature_name][metric] = []

        for metrics in [train_metrics, vali_metrics, test_metrics]:
            metrics[COMBINED] = {LOSS: []}

        return train_metrics, vali_metrics, test_metrics

    def get_metrics_names(self, output_features):
        metrics_names = {}
        for output_feature_name, output_feature in output_features.items():
            for metric in output_feature.metric_functions:
                metrics = metrics_names.get(output_feature_name, [])
                metrics.append(metric)
                metrics_names[output_feature_name] = metrics
        metrics_names[COMBINED] = [LOSS]
        return metrics_names

    def resume_weights_and_optimzier(
            self,
            model_weights_progress_path,
            checkpoint
    ):
        checkpoint.restore(
            tf.train.latest_checkpoint(model_weights_progress_path)
        )

    def reduce_learning_rate(
            self,
            progress_tracker,
            reduce_learning_rate_on_plateau,
            reduce_learning_rate_on_plateau_patience,
            reduce_learning_rate_on_plateau_rate
    ):
        if (progress_tracker.last_improvement >=
                reduce_learning_rate_on_plateau_patience):
            if (progress_tracker.num_reductions_lr >=
                    reduce_learning_rate_on_plateau):
                if is_on_master():
                    logger.info(
                        'It has been ' +
                        str(progress_tracker.last_improvement) +
                        ' epochs since last validation accuracy improvement '
                        'and the learning rate was already reduced ' +
                        str(progress_tracker.num_reductions_lr) +
                        ' times, not reducing it anymore'
                    )
            else:
                if is_on_master():
                    logger.info(
                        'PLATEAU REACHED, reducing learning rate '
                        'due to lack of validation improvement, it has been ' +
                        str(progress_tracker.last_improvement) +
                        ' epochs since last validation accuracy improvement '
                        'or since the learning rate was reduced'
                    )

                progress_tracker.learning_rate *= (
                    reduce_learning_rate_on_plateau_rate
                )
                progress_tracker.last_improvement_epoch = (
                    progress_tracker.epoch
                )
                progress_tracker.last_improvement = 0
                progress_tracker.num_reductions_lr += 1

    def increase_batch_size(
            self,
            progress_tracker,
            increase_batch_size_on_plateau_patience,
            increase_batch_size_on_plateau,
            increase_batch_size_on_plateau_max,
            increase_batch_size_on_plateau_rate
    ):
        if (progress_tracker.last_improvement >=
                increase_batch_size_on_plateau_patience):
            if (progress_tracker.num_increases_bs >=
                    increase_batch_size_on_plateau):
                if is_on_master():
                    logger.info(
                        'It has been ' +
                        str(progress_tracker.last_improvement) +
                        ' epochs since last validation accuracy improvement '
                        'and the learning rate was already reduced ' +
                        str(progress_tracker.num_increases_bs) +
                        ' times, not reducing it anymore'
                    )

            elif (progress_tracker.batch_size ==
                  increase_batch_size_on_plateau_max):
                if is_on_master():
                    logger.info(
                        'It has been' +
                        str(progress_tracker.last_improvement) +
                        ' epochs since last validation accuracy improvement '
                        'and the batch size was already increased ' +
                        str(progress_tracker.num_increases_bs) +
                        ' times and currently is ' +
                        str(progress_tracker.batch_size) +
                        ', the maximum allowed'
                    )
            else:
                if is_on_master():
                    logger.info(
                        'PLATEAU REACHED '
                        'increasing batch size due to lack of '
                        'validation improvement, it has been ' +
                        str(progress_tracker.last_improvement) +
                        ' epochs since last validation accuracy improvement '
                        'or since the batch size was increased'
                    )

                progress_tracker.batch_size = min(
                    (increase_batch_size_on_plateau_rate *
                     progress_tracker.batch_size),
                    increase_batch_size_on_plateau_max
                )
                progress_tracker.last_improvement_epoch = progress_tracker.epoch
                progress_tracker.last_improvement = 0
                progress_tracker.num_increases_bs += 1


class ProgressTracker:
    def __init__(
            self,
            epoch,
            batch_size,
            steps,
            last_improvement_epoch,
            best_valid_metric,
            learning_rate,
            num_reductions_lr,
            num_increases_bs,
            train_metrics,
            vali_metrics,
            test_metrics,
            last_improvement=0
    ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps = steps
        self.last_improvement_epoch = last_improvement_epoch
        self.last_improvement = last_improvement
        self.learning_rate = learning_rate
        self.best_valid_metric = best_valid_metric
        self.num_reductions_lr = num_reductions_lr
        self.num_increases_bs = num_increases_bs
        self.train_metrics = train_metrics
        self.vali_metrics = vali_metrics
        self.test_metrics = test_metrics

    def save(self, filepath):
        save_json(filepath, self.__dict__)

    @staticmethod
    def load(filepath):
        loaded = load_json(filepath)
        return ProgressTracker(**loaded)


# todo refactoring: should change
def load_model_and_definition(model_dir,
                              use_horovod=None,
                              gpus=None,
                              gpu_memory_limit=None,
                              allow_parallel_threads=True):
    # Load model definition and weights
    model_definition = load_json(
        os.path.join(
            model_dir,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )
    )
    trainer = Trainer.load(model_dir,
                           use_horovod=use_horovod,
                           gpus=gpus,
                           gpu_memory_limit=gpu_memory_limit,
                           allow_parallel_threads=allow_parallel_threads)
    return trainer, model_definition
