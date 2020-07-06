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
import tensorflow as tf


# # todo tf2 need a tf2 version of this function
# def optimize(
#         loss,
#         training_parameters,
#         learning_rate,
#         global_step,
#         horovod=None
# ):
#     if training_parameters is not None and training_parameters[
#         'decay'] is True:
#         learning_rate = tf.train.exponential_decay(
#             learning_rate, global_step,
#             training_parameters['decay_steps'],
#             training_parameters['decay_rate'],
#             staircase=training_parameters['staircase'])
#
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#
#     with tf.variable_scope('optimizer'):
#         if training_parameters is not None:
#             optimizer_args = dict(training_parameters['optimizer'])
#             optimizer_type = optimizer_args.pop('type')
#             optimizer_fun = get_optimizer_fun(optimizer_type)
#             if optimizer_type == 'adagradda':
#                 optimizer = optimizer_fun(learning_rate, global_step,
#                                           **optimizer_args)
#             else:
#                 optimizer = optimizer_fun(learning_rate, **optimizer_args)
#
#             if horovod:
#                 optimizer = horovod.DistributedOptimizer(optimizer)
#
#             optimize = optimizer.minimize(loss,
#                                           global_step=global_step)
#             if 'gradient_clipping' in training_parameters and \
#                     training_parameters['gradient_clipping'] is not None:
#                 grad_clip_norm = training_parameters['gradient_clipping']
#                 gradients, variables = zip(*optimizer.compute_gradients(loss))
#                 gradients, _ = tf.clip_by_global_norm(gradients,
#                                                       grad_clip_norm)
#                 apply_grads = optimizer.apply_gradients(
#                     zip(gradients, variables))
#                 increment_global_step = tf.assign(global_step,
#                                                   global_step + 1)
#                 optimize = tf.group(apply_grads,
#                                     increment_global_step)
#         else:
#             optimizer = tf.train.AdamOptimizer(learning_rate)
#
#             if horovod:
#                 optimizer = horovod.DistributedOptimizer(optimizer)
#
#             optimize = optimizer.minimize(loss,
#                                           global_step=global_step)
#
#     optimize = tf.group([optimize, update_ops])
#
#     return optimize, learning_rate


# def get_optimizer_fun(optimizer_type):
#     optimizer_type = optimizer_type.lower()
#     if (
#             optimizer_type == 'sgd' or
#             optimizer_type == 'stochastic_gradient_descent' or
#             optimizer_type == 'gd' or
#             optimizer_type == 'gradient_descent'
#     ):
#         return tf.train.GradientDescentOptimizer
#     elif optimizer_type == 'adam':
#         return tf.train.AdamOptimizer
#     elif optimizer_type == 'adadelta':
#         return tf.train.AdadeltaOptimizer
#     elif optimizer_type == 'adagrad':
#         return tf.train.AdagradOptimizer
#     elif optimizer_type == 'adagradda':
#         return tf.train.AdagradDAOptimizer
#     elif optimizer_type == 'momentum':
#         return tf.train.MomentumOptimizer
#     elif optimizer_type == 'ftrl':
#         return tf.train.FtrlOptimizer
#     elif optimizer_type == 'proximalgd':
#         return tf.train.ProximalGradientDescentOptimizer
#     elif optimizer_type == 'proximaladagrad':
#         return tf.train.ProximalAdagradOptimizer
#     elif optimizer_type == 'rmsprop':
#         return tf.train.RMSPropOptimizer
#     else:
#         raise ValueError('Invalid optimizer_type: ' + optimizer_type)


# todo tf2: improve this class with better names and parameters
def ClippedOptimizer(type='sgd',
                     clipglobalnorm=5.0,
                     clipnorm=None,
                     clipvalue=None,
                     **kwargs):
    optimizer_type = type.lower()
    if (
            optimizer_type == 'sgd' or
            optimizer_type == 'stochastic_gradient_descent' or
            optimizer_type == 'gd' or
            optimizer_type == 'gradient_descent'
    ):
        optimizer = tf.keras.optimizers.SGD(**kwargs)
    elif optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(**kwargs)
    elif optimizer_type == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(**kwargs)
    elif optimizer_type == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(**kwargs)
    elif optimizer_type == 'adagradda':
        optimizer = None  # tf.train.AdagradDAOptimizer  todo appears tf.keras.optimizers does not support
    elif optimizer_type == 'momentum':
        optimizer = None  # tf.train.MomentumOptimizer  todo appears tf.keras.optimizers does not support
    elif optimizer_type == 'ftrl':
        optimizer = tf.keras.optimizers.Ftrl(**kwargs)
    elif optimizer_type == 'proximalgd':
        optimizer = None  # tf.train.ProximalGradientDescentOptimizer todo appears tf.keras.optimizers does not support
    elif optimizer_type == 'proximaladagrad':
        optimizer = None  # tf.train.ProximalAdagradOptimizer todo appears tf.keras.optimizers does not support
    elif optimizer_type == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(**kwargs)
    else:
        raise ValueError('Invalid optimizer_type: ' + type)

    return clip_optimizer(optimizer, clipglobalnorm, clipnorm, clipvalue)


def clip_optimizer(optimizer, clipglobalnorm, clipnorm, clipvalue):
    class _ClippedOptimizer(tf.keras.optimizers.Optimizer):
        def __init__(self, **kwargs):
            self.clipglobalnorm = clipglobalnorm
            self.clipnorm = clipnorm
            self.clipvalue = clipvalue
            super(self.__class__, self).__init__(**kwargs)

        def minimize_with_tape(self, tape, loss, variables):
            gradients = tape.gradient(loss, variables)
            if self.clipglobalnorm:
                gradients, _ = tf.clip_by_global_norm(gradients,
                                                      self.clipglobalnorm)
            if self.clipnorm:
                gradients = map(
                    lambda x: tf.clip_by_norm(x, self.clipnorm),
                    gradients
                )
            if self.clipvalue:
                gradients = map(
                    lambda x: tf.clip_by_value(
                        x,
                        clip_value_min=self.clipvalue[0],
                        clip_value_max=self.clipvalue[1]
                    ),
                    gradients
                )
            self.apply_gradients(zip(gradients, variables))

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_ClippedOptimizer.__dict__))
    return cls.from_config(optimizer.get_config())
