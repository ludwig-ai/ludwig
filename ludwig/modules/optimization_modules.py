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
import torch

from ludwig.utils.misc_utils import get_from_registry

'''
optimizers_registry = {
    'sgd': tf.keras.optimizers.SGD,
    'stochastic_gradient_descent': tf.keras.optimizers.SGD,
    'gd': tf.keras.optimizers.SGD,
    'gradient_descent': tf.keras.optimizers.SGD,
    'adam': tf.keras.optimizers.Adam,
    'adadelta': tf.keras.optimizers.Adadelta,
    'adagrad': tf.keras.optimizers.Adagrad,
    'adamax': tf.keras.optimizers.Adamax,
    'ftrl': tf.keras.optimizers.Ftrl,
    'nadam': tf.keras.optimizers.Nadam,
    'rmsprop': tf.keras.optimizers.RMSprop,
}
'''
optimizers_registry = {
    'sgd': torch.optim.SGD,
    'stochastic_gradient_descent': torch.optim.SGD,
    'gd': torch.optim.SGD,
    'gradient_descent': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adamax': torch.optim.Adamax,
    #'ftrl': tf.keras.optimizers.Ftrl,
    #'nadam': tf.keras.optimizers.Nadam,
    'rmsprop': torch.optim.RMSprop,
}


def ClippedOptimizer(type='sgd',
                     clipglobalnorm=5.0,
                     clipnorm=None,
                     clipvalue=None,
                     horovod=None,
                     **kwargs):
    #optimizer = get_from_registry(type.lower(), optimizers_registry)(**kwargs)
    optimizer = get_from_registry(type.lower(), optimizers_registry)
    return clip_optimizer(optimizer, clipglobalnorm, clipnorm, clipvalue,
                          horovod=horovod, **kwargs)


def clip_optimizer(optimizer, clipglobalnorm, clipnorm, clipvalue,
                   horovod=None, **kwargs):
    #class _ClippedOptimizer(tf.keras.optimizers.Optimizer):
    class _ClippedOptimizer(torch.optim.Optimizer):
        def __init__(self, **kwargs):
            self.clipglobalnorm = clipglobalnorm
            self.clipnorm = clipnorm
            self.clipvalue = clipvalue
            self.horovod = horovod
            super(self.__class__, self).__init__(**kwargs)

        '''
        def minimize_with_tape(self, tape, loss, variables):
            if self.horovod:
                tape = self.horovod.DistributedGradientTape(tape)

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
        '''
        def minimize(self, loss, variables):
            if self.horovod:
                tape = self.horovod.DistributedGradientTape(tape)

            loss.backward()
            if self.clipglobalnorm:
                torch.nn.utils.clip_grad_norm_(variables, self.clipglobalnorm)
            if self.clipnorm:
                for x in variables:
                    torch.nn.utils.clip_grad_norm_(x, self.clipglobalnorm)
            if self.clipvalue:
                pass

            self.step()

        def set_learning_rate(self, learning_rate):
            #self.lr.assign(learning_rate)
            for g in self.param_groups:
                g['lr'] = learning_rate
    '''
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_ClippedOptimizer.__dict__))
    '''
    cls = type(optimizer.__name__, (optimizer,),
               dict(_ClippedOptimizer.__dict__))
    #return cls.from_config(optimizer.get_config())
    return cls(**kwargs)
