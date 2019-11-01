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
import re
import signal
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tabulate import tabulate
from tensorflow.python import debug as tf_debug
from tensorflow.python.saved_model import builder as saved_model_builder
from tqdm import tqdm

from ludwig.constants import *
from ludwig.contrib import contrib_command
from ludwig.features.feature_registries import output_type_registry
from ludwig.features.feature_utils import SEQUENCE_TYPES
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.globals import MODEL_WEIGHTS_PROGRESS_FILE_NAME
from ludwig.globals import TRAINING_PROGRESS_FILE_NAME
from ludwig.globals import is_on_master
from ludwig.globals import is_progressbar_disabled
from ludwig.models.combiners import get_build_combiner
from ludwig.models.inputs import build_inputs, dynamic_length_encoders
from ludwig.models.modules.loss_modules import regularizer_registry
from ludwig.models.modules.measure_modules import get_improved_fun
from ludwig.models.modules.measure_modules import get_initial_validation_value
from ludwig.models.modules.optimization_modules import optimize
from ludwig.models.outputs import build_outputs
from ludwig.utils import time_utils
from ludwig.utils.batcher import Batcher
from ludwig.utils.batcher import BucketedBatcher
from ludwig.utils.batcher import DistributedBatcher
from ludwig.utils.data_utils import load_json, save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.defaults import default_training_params
from ludwig.utils.math_utils import learning_rate_warmup_distributed, \
    learning_rate_warmup
from ludwig.utils.misc import set_random_seed
from ludwig.utils.misc import sum_dicts
from ludwig.utils.tf_utils import get_tf_config

logger = logging.getLogger(__name__)


class Model:
    """
    Model is a class that builds the model that Ludwig uses
    """

    def __init__(
            self,
            input_features,
            output_features,
            combiner,
            training,
            preprocessing,
            use_horovod=False,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        self.horovod = None
        if use_horovod:
            import horovod.tensorflow
            self.horovod = horovod.tensorflow
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD

        self.debug = debug
        self.weights_save_path = None
        self.hyperparameters = {}
        self.session = None

        self.epochs = None
        self.received_sigint = False

        self.__build(
            input_features,
            output_features,
            combiner,
            training,
            preprocessing,
            random_seed,
            **kwargs
        )

    def __build(
            self,
            input_features,
            output_features,
            combiner,
            training,
            preprocessing,
            random_seed,
            **kwargs
    ):
        self.hyperparameters['input_features'] = input_features
        self.hyperparameters['output_features'] = output_features
        self.hyperparameters['combiner'] = combiner
        self.hyperparameters['training'] = training
        self.hyperparameters['preprocessing'] = preprocessing
        self.hyperparameters['random_seed'] = random_seed
        self.hyperparameters.update(kwargs)

        if self.horovod:
            self.horovod.init()

        tf.compat.v1.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            # ================ Setup ================
            tf.compat.v1.set_random_seed(random_seed)

            self.global_step = tf.Variable(0, trainable=False)
            self.regularization_lambda = tf.compat.v1.placeholder(
                tf.float32,
                name='regularization_lambda'
            )
            regularizer = regularizer_registry[training['regularizer']]
            self.regularizer = regularizer(self.regularization_lambda)

            self.learning_rate = tf.compat.v1.placeholder(
                tf.float32,
                name='learning_rate'
            )
            self.dropout_rate = tf.compat.v1.placeholder(tf.float32, name='dropout_rate')
            self.is_training = tf.compat.v1.placeholder(tf.bool, [], name='is_training')

            # ================ Inputs ================
            feature_encodings = build_inputs(
                input_features,
                self.regularizer,
                self.dropout_rate,
                is_training=self.is_training
            )

            for fe_name, fe_properties in feature_encodings.items():
                setattr(self, fe_name, fe_properties['placeholder'])

            # ================ Model ================
            logger.debug('- Combiner {}'.format(combiner['type']))
            build_combiner = get_build_combiner(combiner['type'])(**combiner)
            hidden, hidden_size = build_combiner(
                feature_encodings,
                self.regularizer,
                self.dropout_rate,
                is_training=self.is_training,
                **kwargs
            )

            # ================ Outputs ================
            outs = build_outputs(
                output_features,
                hidden,
                hidden_size,
                regularizer=self.regularizer,
                dropout_rate=self.dropout_rate,
                is_training=self.is_training
            )

            (
                self.train_reg_mean_loss,
                self.eval_combined_loss,
                self.regularization_loss,
                output_tensors
            ) = outs

            for ot_name, ot in output_tensors.items():
                setattr(self, ot_name, ot)

            # ================ Optimizer ================
            self.optimize, self.learning_rate = optimize(
                self.train_reg_mean_loss,
                training,
                self.learning_rate,
                self.global_step,
                self.horovod
            )

            tf.compat.v1.summary.scalar('train_reg_mean_loss', self.train_reg_mean_loss)

            self.merged_summary = tf.compat.v1.summary.merge_all()
            self.graph = graph
            self.graph_initialize = tf.compat.v1.global_variables_initializer()
            if self.horovod:
                self.broadcast_op = self.horovod.broadcast_global_variables(0)
            self.saver = tf.compat.v1.train.Saver()

    def initialize_session(self, gpus=None, gpu_fraction=1):
        if self.session is None:

            self.session = tf.compat.v1.Session(
                config=get_tf_config(gpus, gpu_fraction, self.horovod),
                graph=self.graph
            )
            self.session.run(self.graph_initialize)

            if self.debug:
                session = tf_debug.LocalCLIDebugWrapperSession(self.session)
                session.add_tensor_filter(
                    'has_inf_or_nan',
                    tf_debug.has_inf_or_nan
                )

        return self.session

    def close_session(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def feed_dict(
            self,
            batch,
            regularization_lambda=default_training_params[
                'regularization_lambda'],
            learning_rate=default_training_params['learning_rate'],
            dropout_rate=default_training_params['dropout_rate'],
            is_training=True
    ):
        input_features = self.hyperparameters['input_features']
        output_features = self.hyperparameters['output_features']
        feed_dict = {
            self.is_training: is_training,
            self.regularization_lambda: regularization_lambda,
            self.learning_rate: learning_rate,
            self.dropout_rate: dropout_rate
        }
        for input_feature in input_features:
            feed_dict[getattr(self, input_feature['name'])] = batch[
                input_feature['name']]
        for output_feature in output_features:
            if output_feature['name'] in batch:
                feed_dict[getattr(self, output_feature['name'])] = batch[
                    output_feature['name']]
        return feed_dict

    def train(
            self,
            training_set,
            validation_set=None,
            test_set=None,
            validation_field=None,
            validation_measure=None,
            save_path='model',
            regularization_lambda=0.0,
            epochs=100,
            learning_rate=0.001,
            batch_size=128,
            eval_batch_size=0,
            bucketing_field=None,
            dropout_rate=0.0,
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
            gpus=None,
            gpu_fraction=1,
            random_seed=default_random_seed,
            **kwargs
    ):
        """Trains a model with a set of hyperparameters listed below. Customizable
        :param training_set: The training set
        :param validation_set: The validation dataset
        :param test_set: The test dataset
        :param validation_field: The first output feature, by default it is set
               as the same field of the first output feature.
        :param validation_measure: Measure used on the validation field, it is
               accuracy by default
        :type validation_measure:
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
        :param dropout_rate: dropout_rate probability (probability of dropping
               a neuron in a given layer)
        :type dropout_rate: Float
        :param early_stop: How many epochs without any improvement in the
               validation_measure triggers the algorithm to stop
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
               the validation measure imrpvoes, but if the model is really big
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
        :param gpus: List of gpus to use
        :type gpus: List
        :param gpu_fraction: Percentage of the GPU that is intended to be used
        :type gpu_fraction: Float
        :param random_seed: Default initialization for the random seeds
        :type: Float
        """
        # ====== General setup =======
        output_features = self.hyperparameters['output_features']
        self.epochs = epochs
        digits_per_epochs = len(str(self.epochs))
        self.received_sigint = False
        # Only use signals when on the main thread to avoid issues with CherryPy: https://github.com/uber/ludwig/issues/286
        if threading.current_thread() == threading.main_thread():
            signal.signal(signal.SIGINT, self.set_epochs_to_1_or_quit)
        should_validate = validation_set is not None and validation_set.size > 0
        if eval_batch_size < 1:
            eval_batch_size = batch_size
        stat_names = self.get_stat_names(output_features)
        if self.horovod:
            learning_rate *= self.horovod.size()

        # ====== Setup file names =======
        if is_on_master():
            os.makedirs(save_path, exist_ok=True)
        model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        model_weights_progress_path = os.path.join(
            save_path,
            MODEL_WEIGHTS_PROGRESS_FILE_NAME
        )
        model_hyperparameters_path = os.path.join(
            save_path,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )

        # ====== Setup session =======
        session = self.initialize_session(gpus, gpu_fraction)

        if self.weights_save_path:
            self.restore(session, self.weights_save_path)

        train_writer = None
        if is_on_master():
            if not skip_save_log:
                train_writer = tf.compat.v1.summary.FileWriter(
                    os.path.join(save_path, 'log', 'train'),
                    session.graph
                )

        if self.debug:
            session = tf_debug.LocalCLIDebugWrapperSession(session)
            session.add_tensor_filter(
                'has_inf_or_nan',
                tf_debug.has_inf_or_nan
            )

        # ================ Resume logic ================
        if resume:
            progress_tracker = self.resume_training(
                save_path,
                model_weights_path
            )
            if is_on_master():
                self.resume_session(
                    session,
                    save_path,
                    model_weights_path,
                    model_weights_progress_path
                )
        else:
            (
                train_stats,
                vali_stats,
                test_stats
            ) = self.initialize_training_stats(output_features)

            progress_tracker = ProgressTracker(
                batch_size=batch_size,
                epoch=0,
                steps=0,
                last_improvement_epoch=0,
                learning_rate=learning_rate,
                best_valid_measure=get_initial_validation_value(
                    validation_measure
                ),
                num_reductions_lr=0,
                num_increases_bs=0,
                train_stats=train_stats,
                vali_stats=vali_stats,
                test_stats=test_stats
            )

        # horovod broadcasting after init or restore
        if self.horovod:
            session.run(self.broadcast_op)

        set_random_seed(random_seed)
        batcher = self.initialize_batcher(
            training_set,
            batch_size,
            bucketing_field
        )

        # ================ Training Loop ================
        while progress_tracker.epoch < self.epochs:
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

                if self.horovod:
                    current_learning_rate = learning_rate_warmup_distributed(
                        progress_tracker.learning_rate,
                        progress_tracker.epoch,
                        learning_rate_warmup_epochs,
                        self.horovod.size(),
                        batcher.step,
                        batcher.steps_per_epoch
                    ) * self.horovod.size()
                else:
                    current_learning_rate = learning_rate_warmup(
                        progress_tracker.learning_rate,
                        progress_tracker.epoch,
                        learning_rate_warmup_epochs,
                        batcher.step,
                        batcher.steps_per_epoch
                    )

                readout_nodes = {'optimize': self.optimize}
                if not skip_save_log:
                    readout_nodes['summary'] = self.merged_summary

                output_values = session.run(
                    readout_nodes,
                    feed_dict=self.feed_dict(
                        batch,
                        regularization_lambda=regularization_lambda,
                        learning_rate=current_learning_rate,
                        dropout_rate=dropout_rate,
                        is_training=True
                    )
                )

                if is_on_master():
                    if not skip_save_log:
                        # it is initialized only on master
                        train_writer.add_summary(output_values['summary'],
                                                 progress_tracker.steps)

                progress_tracker.steps += 1
                if is_on_master():
                    progress_bar.update(1)

            # post training
            if is_on_master():
                progress_bar.close()

            progress_tracker.epoch += 1
            batcher.reset()  # todo this may be useless, doublecheck

            # ================ Eval ================
            # init tables
            tables = OrderedDict()
            for output_feature in output_features:
                field_name = output_feature['name']
                tables[field_name] = [
                    [field_name] + stat_names[field_name]]
            tables['combined'] = [['combined', LOSS, ACCURACY]]

            # eval measures on train set
            self.evaluation(
                session,
                training_set,
                'train',
                regularization_lambda,
                progress_tracker.train_stats,
                tables,
                eval_batch_size,
                bucketing_field
            )

            if validation_set is not None and validation_set.size > 0:
                # eval measures on validation set
                self.evaluation(
                    session,
                    validation_set,
                    'vali',
                    regularization_lambda,
                    progress_tracker.vali_stats,
                    tables,
                    eval_batch_size,
                    bucketing_field
                )

            if test_set is not None and test_set.size > 0:
                # eval measures on test set
                self.evaluation(
                    session,
                    test_set,
                    'test',
                    regularization_lambda,
                    progress_tracker.test_stats,
                    tables,
                    eval_batch_size,
                    bucketing_field
                )

            # mbiu and end of epoch prints
            elapsed_time = (time.time() - start_time) * 1000.0

            if is_on_master():
                logger.info('Took {time}'.format(
                    time=time_utils.strdelta(elapsed_time)))

            # stat prints
            for output_feature, table in tables.items():
                if (
                        output_feature != 'combined' or
                        (output_feature == 'combined' and
                         len(output_features) > 1)
                ):
                    if is_on_master():
                        logger.info(
                            tabulate(
                                table,
                                headers='firstrow',
                                tablefmt='fancy_grid',
                                floatfmt='.4f'
                            )
                        )

            if should_validate:
                should_break = self.check_progress_on_validation(
                    progress_tracker,
                    validation_field,
                    validation_measure,
                    session,
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
                )
                if should_break:
                    break
            else:
                # there's no validation, so we save the model at each iteration
                if is_on_master():
                    if not skip_save_model:
                        self.save_weights(session, model_weights_path)
                        self.save_hyperparameters(
                            self.hyperparameters,
                            model_hyperparameters_path
                        )

            # ========== Save training progress ==========
            if is_on_master():
                if not skip_save_progress:
                    self.save_weights(session, model_weights_progress_path)
                    progress_tracker.save(
                        os.path.join(
                            save_path,
                            TRAINING_PROGRESS_FILE_NAME
                        )
                    )
                    if skip_save_model:
                        self.save_hyperparameters(
                            self.hyperparameters,
                            model_hyperparameters_path
                        )

            if is_on_master():
                contrib_command("train_epoch_end", progress_tracker)
                logger.info('')

        if train_writer is not None:
            train_writer.close()

        return (
            progress_tracker.train_stats,
            progress_tracker.vali_stats,
            progress_tracker.test_stats
        )

    def train_online(
            self,
            dataset,
            batch_size=128,
            learning_rate=0.01,
            regularization_lambda=0,
            dropout_rate=0,
            bucketing_field=None,
            gpus=None,
            gpu_fraction=1
    ):
        session = self.initialize_session(gpus, gpu_fraction)
        batcher = self.initialize_batcher(dataset, batch_size, bucketing_field)

        # training step loop
        progress_bar = tqdm(
            desc='Trainining online',
            total=batcher.steps_per_epoch,
            file=sys.stdout,
            disable=is_progressbar_disabled()
        )

        while not batcher.last_batch():
            batch = batcher.next_batch()

            _ = session.run(
                [self.optimize],
                feed_dict=self.feed_dict(
                    batch,
                    regularization_lambda=regularization_lambda,
                    learning_rate=learning_rate,
                    dropout_rate=dropout_rate,
                    is_training=True
                )
            )
            progress_bar.update(1)

        progress_bar.close()

    def evaluation(
            self,
            session,
            dataset,
            dataset_name,
            regularization_lambda,
            stats,
            tables,
            batch_size=128,
            bucketing_field=None
    ):
        results = self.batch_evaluation(
            session,
            dataset,
            batch_size,
            bucketing_field=bucketing_field,
            regularization_lambda=regularization_lambda,
            is_training=False,
            name=dataset_name
        )

        for output_feature in self.hyperparameters['output_features']:
            field_name = output_feature['name']
            scores = [dataset_name]

            for stat in stats[field_name]:
                stats[field_name][stat].append(results[field_name][stat])
                scores.append(results[field_name][stat])

            tables[field_name].append(scores)

        stats['combined'][LOSS].append(results['combined'][LOSS])
        stats['combined'][ACCURACY].append(results['combined'][ACCURACY])
        tables['combined'].append(
            [
                dataset_name,
                results['combined'][LOSS],
                results['combined'][ACCURACY]
            ]
        )
        return stats, tables

    def batch_evaluation(
            self,
            session,
            dataset,
            batch_size,
            bucketing_field=None,
            regularization_lambda=0.0,
            is_training=False,
            collect_predictions=False,
            only_predictions=False,
            name=None
    ):
        output_nodes = self.get_output_nodes(
            collect_predictions,
            only_predictions
        )
        output_stats = self.get_outputs_stats()

        set_size = dataset.size
        if set_size == 0:
            if is_on_master():
                logger.warning('No datapoints to evaluate on.')
            return output_stats
        seq_set_size = {output_feature['name']: {} for output_feature in
                        self.hyperparameters['output_features'] if
                        output_feature['type'] in SEQUENCE_TYPES}

        batcher = self.initialize_batcher(
            dataset,
            batch_size,
            bucketing_field,
            should_shuffle=False
        )

        if is_on_master():
            progress_bar = tqdm(
                desc='Evaluation' if name is None
                else 'Evaluation {0: <5.5}'.format(name),
                total=batcher.steps_per_epoch,
                file=sys.stdout,
                disable=is_progressbar_disabled()
            )

        while not batcher.last_batch():
            batch = batcher.next_batch()
            result = session.run(
                output_nodes,
                feed_dict=self.feed_dict(
                    batch,
                    regularization_lambda=regularization_lambda,
                    dropout_rate=0.0,
                    is_training=is_training
                )
            )

            output_stats, seq_set_size = self.update_output_stats_batch(
                output_stats,
                seq_set_size,
                collect_predictions,
                only_predictions,
                result
            )
            if is_on_master():
                progress_bar.update(1)

        if is_on_master():
            progress_bar.close()

        if self.horovod:
            output_stats, seq_set_size = self.merge_workers_outputs(
                output_stats,
                seq_set_size
            )

        output_stats = self.update_output_stats(
            output_stats,
            set_size,
            seq_set_size,
            collect_predictions,
            only_predictions
        )

        if 'combined' in output_stats and LOSS in output_stats['combined']:
            regularization = session.run(
                [self.regularization_loss],
                feed_dict={self.regularization_lambda: regularization_lambda}
            )[0]
            output_stats['combined'][LOSS] += regularization

        return output_stats

    def merge_workers_outputs(self, output_stats, seq_set_size):
        # gather outputs from all workers
        all_workers_output_stats = self.comm.allgather(output_stats)
        all_workers_seq_set_size = self.comm.allgather(seq_set_size)

        # merge them into a single one
        merged_output_stats = sum_dicts(
            all_workers_output_stats,
            dict_type=OrderedDict
        )
        merged_seq_set_size = sum_dicts(all_workers_seq_set_size)

        return merged_output_stats, merged_seq_set_size

    def batch_collect_activations(
            self,
            session,
            dataset,
            batch_size,
            tensor_names,
            bucketing_field=None
    ):
        output_nodes = {tensor_name: self.graph.get_tensor_by_name(tensor_name)
                        for tensor_name in tensor_names}
        collected_tensors = {tensor_name: [] for tensor_name in tensor_names}

        batcher = self.initialize_batcher(
            dataset,
            batch_size,
            bucketing_field,
            should_shuffle=False
        )

        progress_bar = tqdm(
            desc='Collecting Tensors',
            total=batcher.steps_per_epoch,
            file=sys.stdout,
            disable=is_progressbar_disabled()
        )

        while not batcher.last_batch():
            batch = batcher.next_batch()
            result = session.run(
                output_nodes,
                feed_dict=self.feed_dict(
                    batch,
                    is_training=False
                )
            )

            for tensor_name in result:
                for row in result[tensor_name]:
                    collected_tensors[tensor_name].append(row)

            progress_bar.update(1)

        progress_bar.close()

        return collected_tensors

    def get_output_nodes(self, collect_predictions, only_predictions=False):
        output_features = self.hyperparameters['output_features']
        output_nodes = {}

        for output_feature in output_features:
            field_name = output_feature['name']
            feature_type = output_feature['type']
            output_nodes[field_name] = {}
            output_config = output_type_registry[feature_type].output_config

            for stat in output_config:
                output_name = output_config[stat]['output']
                output_type = output_config[stat]['type']
                if ((output_type == PREDICTION and
                     (collect_predictions or only_predictions)) or
                        (output_type == MEASURE and not only_predictions)):
                    output_nodes[field_name][output_name] = getattr(
                        self,
                        output_name + '_' + field_name
                    )

        if not only_predictions:
            output_nodes['eval_combined_loss'] = getattr(
                self,
                'eval_combined_loss'
            )

        return output_nodes

    def get_outputs_stats(self):
        output_features = self.hyperparameters['output_features']
        output_stats = OrderedDict()

        for output_feature in output_features:
            field_name = output_feature['name']
            feature_type = output_feature['type']
            output_stats[field_name] = {}
            output_config = output_type_registry[feature_type].output_config

            for stat in output_config:
                output_value = output_config[stat]['value']
                if isinstance(output_value, list):
                    output_stats[field_name][stat] = []
                else:
                    output_stats[field_name][stat] = output_value

        output_stats['combined'] = {LOSS: 0, ACCURACY: 0}
        return output_stats

    def update_output_stats_batch(
            self,
            output_stats,
            seq_set_size,
            collect_predictions,
            only_predictions,
            result
    ):
        output_features = self.hyperparameters['output_features']
        combined_correct_predictions = None

        for i, output_feature in enumerate(output_features):
            field_name = output_feature['name']
            feature_type = output_feature['type']
            output_config = output_type_registry[feature_type].output_config

            for stat in output_config:
                stat_config = output_config[stat]
                output_type = output_config[stat]['type']
                if ((output_type == PREDICTION and
                     (collect_predictions or only_predictions)) or
                        (output_type == MEASURE and not only_predictions)):
                    aggregation_method = stat_config['aggregation']
                    if aggregation_method == SUM:
                        output_stats[field_name][stat] += (
                            result[field_name][stat_config['output']].sum()
                        )
                    elif aggregation_method == SEQ_SUM:
                        output_stats[field_name][stat] += (
                            result[field_name][stat_config['output']].sum()
                        )
                        seq_set_size[field_name][stat] = (
                                seq_set_size[field_name].get(stat, 0) +
                                len(result[field_name][stat_config['output']])
                        )
                    elif aggregation_method == AVG_EXP:
                        output_stats[field_name][stat] += (
                            result[field_name][stat_config['output']].sum()
                        )
                    elif aggregation_method == APPEND:
                        output_stats[field_name][stat].append(
                            result[field_name][stat_config['output']]
                        )

            if not only_predictions:
                if feature_type in [CATEGORY, BINARY]:
                    correct_predictions = \
                        result[field_name][CORRECT_PREDICTIONS]
                elif feature_type == SEQUENCE:
                    correct_predictions = \
                        result[field_name][CORRECT_ROWWISE_PREDICTIONS]
                else:
                    correct_predictions = None

                if correct_predictions is not None:
                    if combined_correct_predictions is None:
                        combined_correct_predictions = correct_predictions
                    else:
                        combined_correct_predictions = np.logical_and(
                            combined_correct_predictions,
                            correct_predictions
                        )

        if not only_predictions:
            output_stats['combined'][LOSS] += result['eval_combined_loss'].sum()
            output_stats['combined'][ACCURACY] += (
                combined_correct_predictions.sum()
                if combined_correct_predictions is not None else 0
            )

        return output_stats, seq_set_size

    def update_output_stats(
            self,
            output_stats,
            set_size,
            seq_set_size,
            collect_predictions,
            only_predictions
    ):
        output_features = self.hyperparameters['output_features']

        for i, output_feature in enumerate(output_features):
            feature_type = output_feature['type']
            field_name = output_feature['name']
            output_config = output_type_registry[feature_type].output_config

            for stat in output_config:
                output_type = output_config[stat]['type']
                if ((output_type == PREDICTION and
                     (collect_predictions or only_predictions)) or
                        (output_type == MEASURE and not only_predictions)):

                    if output_config[stat]['aggregation'] == SUM:
                        output_stats[field_name][stat] /= set_size

                    elif output_config[stat]['aggregation'] == SEQ_SUM:
                        output_stats[field_name][stat] /= (
                            seq_set_size[field_name][stat]
                        )

                    elif output_config[stat]['aggregation'] == AVG_EXP:
                        output_stats[field_name][stat] = np.exp(
                            output_stats[field_name][stat] / set_size
                        )

                    elif output_config[stat]['aggregation'] == APPEND:
                        if len(output_stats[field_name][stat]) > 0 and len(
                                output_stats[field_name][stat][0].shape) > 1:
                            max_shape = None
                            for result in output_stats[field_name][stat]:
                                if max_shape is None:
                                    max_shape = result.shape
                                else:
                                    max_shape = np.maximum(
                                        max_shape,
                                        result.shape
                                    )

                            results = []
                            for result in output_stats[field_name][stat]:
                                diff_shape = max_shape - np.array(result.shape)
                                diff_shape[0] = 0
                                pad_width = [(0, k) for k in diff_shape]
                                paded_result = np.pad(
                                    result,
                                    pad_width,
                                    'constant',
                                    constant_values=0
                                )
                                results.append(paded_result)
                        else:
                            results = output_stats[field_name][stat]

                        output_stats[field_name][stat] = np.concatenate(
                            results
                        )

            if feature_type == SEQUENCE:
                # trim output sequences
                if LENGTHS in output_stats[field_name]:
                    lengths = output_stats[field_name][LENGTHS]
                    if PREDICTIONS in output_stats[field_name]:
                        output_stats[field_name][PREDICTIONS] = np.array(
                            [list(output_stats[field_name][PREDICTIONS][i,
                                  0:lengths[i]])
                             for i in range(len(lengths))]
                        )
                    if PROBABILITIES in output_stats[field_name]:
                        output_stats[field_name][PROBABILITIES] = np.array(
                            [list(output_stats[field_name][PROBABILITIES][i,
                                  0:lengths[i]]) for i in
                             range(len(lengths))]
                        )

        if not only_predictions:
            output_stats['combined'][LOSS] /= set_size
            output_stats['combined'][ACCURACY] /= set_size

        return output_stats

    def check_progress_on_validation(
            self,
            progress_tracker,
            validation_field,
            validation_measure,
            session, model_weights_path,
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
        improved = get_improved_fun(validation_measure)
        if improved(
                progress_tracker.vali_stats[validation_field][
                    validation_measure][-1],
                progress_tracker.best_valid_measure
        ):
            progress_tracker.last_improvement_epoch = progress_tracker.epoch
            progress_tracker.best_valid_measure = progress_tracker.vali_stats[
                validation_field][validation_measure][-1]
            if is_on_master():
                if not skip_save_model:
                    self.save_weights(session, model_weights_path)
                    self.save_hyperparameters(
                        self.hyperparameters,
                        model_hyperparameters_path
                    )
                    logger.info(
                        'Validation {} on {} improved, model saved'.format(
                            validation_measure,
                            validation_field
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
                        validation_measure,
                        validation_field,
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

    def predict(
            self,
            dataset,
            batch_size,
            evaluate_performance=True,
            gpus=None,
            gpu_fraction=1,
            **kwargs
    ):
        if self.session is None:
            session = self.initialize_session(gpus, gpu_fraction)

            # load parameters
            if self.weights_save_path:
                self.restore(session, self.weights_save_path)
        else:
            session = self.session

        # predict
        predict_stats = self.batch_evaluation(
            session,
            dataset,
            batch_size,
            is_training=False,
            collect_predictions=True,
            only_predictions=not evaluate_performance
        )

        return predict_stats

    def collect_activations(
            self,
            dataset,
            tensor_names,
            batch_size,
            gpus=None,
            gpu_fraction=1,
            **kwargs
    ):
        if self.session is None:
            session = self.initialize_session(gpus, gpu_fraction)

            # load parameters
            if self.weights_save_path:
                self.restore(session, self.weights_save_path)
        else:
            session = self.session

        # get operation names
        operation_names = set(
            [t.name for op in self.graph.get_operations() for t in op.values()]
        )
        for tensor_name in tensor_names:
            if tensor_name not in operation_names:
                raise ValueError(
                    'Tensor / operation {} not present in the '
                    'model graph'.format(tensor_name)
                )

        # collect tensors
        collected_tensors = self.batch_collect_activations(
            session,
            dataset,
            batch_size,
            tensor_names
        )

        return collected_tensors

    def collect_weights(
            self,
            tensor_names,
            gpus=None,
            gpu_fraction=1,
            **kwargs
    ):
        if self.session is None:
            session = self.initialize_session(gpus, gpu_fraction)

            # load parameters
            if self.weights_save_path:
                self.restore(session, self.weights_save_path)
        else:
            session = self.session

        operation_names = set(
            [t.name for op in self.graph.get_operations() for t in op.values()]
        )
        for tensor_name in tensor_names:
            if tensor_name not in operation_names:
                raise ValueError(
                    'Tensor / operation {} not present in the '
                    'model graph'.format(tensor_name)
                )

        # collect tensors
        collected_tensors = {
            tensor_name: session.run(self.graph.get_tensor_by_name(tensor_name))
            for tensor_name in tensor_names
        }

        return collected_tensors

    def save_weights(self, session, save_path):
        self.weights_save_path = self.saver.save(session, save_path)

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

    def save_savedmodel(self, save_path):

        input_tensors = {}
        for input_feature in self.hyperparameters['input_features']:
            input_tensors[input_feature['name']] = getattr(
                self, input_feature['name']
            )

        output_tensors = {}
        for output_feature in self.hyperparameters['output_features']:
            output_tensors[output_feature['name']] = getattr(
                self,
                output_feature['name']
            )

        session = self.initialize_session()

        builder = saved_model_builder.SavedModelBuilder(save_path)
        builder.add_meta_graph_and_variables(
            session,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict': tf.saved_model.predict_signature_def(
                    input_tensors, output_tensors)
            },
            strip_default_attrs=True,
            saver=self.saver,
        )
        builder.save()

    def restore(self, session, weights_path):
        self.saver.restore(session, weights_path)

    @staticmethod
    def load(load_path, use_horovod=False):
        hyperparameter_file = os.path.join(
            load_path,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )
        hyperparameters = load_json(hyperparameter_file)
        model = Model(use_horovod=use_horovod, **hyperparameters)
        model.weights_save_path = os.path.join(
            load_path,
            MODEL_WEIGHTS_FILE_NAME
        )
        return model

    def set_epochs_to_1_or_quit(self, signum, frame):
        if not self.received_sigint:
            self.epochs = 1
            self.received_sigint = True
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

    def resume_training(self, save_path, model_weights_path):
        if is_on_master():
            logger.info('Resuming training of model: {0}'.format(save_path))
        self.weights_save_path = model_weights_path
        progress_tracker = ProgressTracker.load(
            os.path.join(
                save_path,
                TRAINING_PROGRESS_FILE_NAME
            )
        )
        return progress_tracker

    def initialize_training_stats(self, output_features):
        train_stats = OrderedDict()
        vali_stats = OrderedDict()
        test_stats = OrderedDict()

        for output_feature in output_features:
            field_name = output_feature['name']

            train_stats[field_name] = OrderedDict()
            vali_stats[field_name] = OrderedDict()
            test_stats[field_name] = OrderedDict()
            output_config = output_type_registry[
                output_feature['type']].output_config

            for stat, config in output_config.items():
                if config['type'] == MEASURE:
                    train_stats[field_name][stat] = []
                    vali_stats[field_name][stat] = []
                    test_stats[field_name][stat] = []

        for stats in [train_stats, vali_stats, test_stats]:
            stats['combined'] = {
                LOSS: [],
                ACCURACY: []
            }

        return train_stats, vali_stats, test_stats

    def get_stat_names(self, output_features):
        stat_names = {}
        for output_feature in output_features:
            field_name = output_feature['name']
            output_config = output_type_registry[
                output_feature['type']].output_config

            for stat, config in output_config.items():
                if config['type'] == MEASURE:
                    stats = stat_names.get(field_name, [])
                    stats.append(stat)
                    stat_names[field_name] = stats
        stat_names['combined'] = [LOSS, ACCURACY]
        return stat_names

    def initialize_batcher(
            self,
            dataset,
            batch_size=128,
            bucketing_field=None,
            should_shuffle=True,
            ignore_last=False
    ):
        if self.horovod:
            batcher = DistributedBatcher(
                dataset,
                self.horovod.rank(),
                self.horovod,
                batch_size,
                should_shuffle=should_shuffle,
                ignore_last=ignore_last
            )
        elif bucketing_field is not None:
            input_features = self.hyperparameters['input_features']
            bucketing_feature = [
                feature for feature in input_features if
                feature['name'] == bucketing_field
            ]
            if not bucketing_feature:
                raise ValueError(
                    'Bucketing field {} not present in input features'.format(
                        bucketing_field
                    )
                )
            else:
                bucketing_feature = bucketing_feature[0]
            should_trim = bucketing_feature[
                              'encoder'] in dynamic_length_encoders
            if 'preprocessing' in bucketing_feature:
                trim_side = bucketing_feature['preprocessing']['padding']
            else:
                trim_side = self.hyperparameters['preprocessing'][
                    bucketing_feature['type']]['padding']

            batcher = BucketedBatcher(
                dataset,
                bucketing_field=bucketing_field,
                batch_size=batch_size,
                buckets=10,
                ignore_last=ignore_last,
                should_shuffle=should_shuffle,
                should_trim=should_trim,
                trim_side=trim_side
            )
        else:
            batcher = Batcher(
                dataset,
                batch_size,
                should_shuffle=should_shuffle,
                ignore_last=ignore_last
            )
        return batcher

    def resume_session(
            self,
            session,
            save_path,
            model_weights_path,
            model_weights_progress_path
    ):
        num_matching_files = 0
        pattern = re.compile(MODEL_WEIGHTS_PROGRESS_FILE_NAME)
        for file_path in os.listdir(save_path):
            if pattern.match(file_path):
                num_matching_files += 1
        if num_matching_files == 3:
            self.restore(session, model_weights_progress_path)
        else:
            self.restore(session, model_weights_path)

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
            best_valid_measure,
            learning_rate,
            num_reductions_lr,
            num_increases_bs,
            train_stats,
            vali_stats,
            test_stats,
            last_improvement=0
    ):
        self.batch_size = batch_size
        self.epoch = epoch
        self.steps = steps
        self.last_improvement_epoch = last_improvement_epoch
        self.last_improvement = last_improvement
        self.learning_rate = learning_rate
        self.best_valid_measure = best_valid_measure
        self.num_reductions_lr = num_reductions_lr
        self.num_increases_bs = num_increases_bs
        self.train_stats = train_stats
        self.vali_stats = vali_stats
        self.test_stats = test_stats

    def save(self, filepath):
        save_json(filepath, self.__dict__)

    @staticmethod
    def load(filepath):
        loaded = load_json(filepath)
        return ProgressTracker(**loaded)


def load_model_and_definition(model_dir, use_horovod=False):
    # Load model definition and weights
    model_definition = load_json(
        os.path.join(
            model_dir,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )
    )
    model = Model.load(model_dir, use_horovod=use_horovod)
    return model, model_definition
