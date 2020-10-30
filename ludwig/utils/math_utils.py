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
import math

import numpy as np


def softmax(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


def int_type(number):
    if number <= np.iinfo(np.int8).max:
        return np.int8
    elif number <= np.iinfo(np.int16).max:
        return np.int16
    elif number <= np.iinfo(np.int32).max:
        return np.int32
    else:  # if number <= np.iinfo(np.int64).max:
        return np.int64


def convert_size(size_bytes):
    if size_bytes == 0:
        return '0B'
    size_name = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return '{} {}'.format(s, size_name[i])


def exponential_decay(initial_learning_rate, decay_rate, decay_steps, step,
                      staircase=False):
    decay_rate = float(decay_rate)
    decay_steps = float(decay_steps)
    step = float(step)
    exponent = step / decay_steps
    if staircase:
        exponent = math.ceil(exponent)
    return initial_learning_rate * math.pow(decay_rate, exponent)


def learning_rate_warmup_distributed(
        learning_rate,
        epoch,
        warmup_epochs,
        num_workers,
        curr_step,
        steps_per_epoch
):
    """Implements gradual learning rate warmup:
    `lr = initial_lr / hvd.size()` ---> `lr = initial_lr`
     `initial_lr` is the learning rate of the model optimizer at the start
     of the training. This technique was described in the paper
     "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".
     See https://arxiv.org/pdf/1706.02677.pdf for details.

     Inspired by Horovod's implementation:
     https://horovod.readthedocs.io/en/stable/api.html#horovod.tensorflow.keras.callbacks.LearningRateWarmupCallback
     Math recap:
                                                   curr_step
            epoch               = full_epochs + ---------------
                                                steps_per_epoch
                                   lr     size - 1
            lr'(epoch)          = ---- * (-------- * epoch + 1)
                                  size     warmup
                                   lr
            lr'(epoch = 0)      = ----
                                  size
            lr'(epoch = warmup) = lr
    """
    if epoch >= warmup_epochs:
        return learning_rate
    else:
        epoch_adjusted = float(epoch) + (curr_step / steps_per_epoch)
        return learning_rate / num_workers * (
                epoch_adjusted * (num_workers - 1) / warmup_epochs + 1)


def learning_rate_warmup(
        learning_rate,
        epoch,
        warmup_epochs,
        curr_step,
        steps_per_epoch
):
    if epoch >= warmup_epochs:
        return learning_rate
    else:
        global_curr_step = 1 + curr_step + epoch * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        warmup_percent_done = global_curr_step / warmup_steps
        warmup_learning_rate = learning_rate * warmup_percent_done

        is_warmup = int(global_curr_step < warmup_steps)
        interpolated_learning_rate = (
                (1.0 - is_warmup) * learning_rate +
                is_warmup * warmup_learning_rate
        )

        return interpolated_learning_rate


def round2precision(val, precision: int = 0, which: str = ''):
    assert precision >= 0
    val *= 10 ** precision
    round_callback = round
    if which.lower() == 'up':
        round_callback = math.ceil
    if which.lower() == 'down':
        round_callback = math.floor
    return '{1:.{0}f}'.format(precision, round_callback(val) / 10 ** precision)
