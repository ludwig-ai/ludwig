import h5py
import numpy as np
import pandas as pd
import json
import yaml
from ludwig import LudwigModel
import copy
import ray

input_features = [{'name': 'Pclass', 'type': 'category'},
  {'name': 'Sex', 'type': 'category'},
  {'name': 'Age',
   'type': 'numerical',
   'missing_value_strategy': 'fill_with_mean'},
  {'name': 'SibSp', 'type': 'numerical'},
  {'name': 'Parch', 'type': 'numerical'},
  {'name': 'Fare',
   'type': 'numerical',
   'missing_value_strategy': 'fill_with_mean'},
  {'name': 'Embarked', 'type': 'category', 'representation': 'sparse'}]


combiner = {'type': 'concat', 'num_fc_layers': 1, 'fc_size': 48}

output_features = [{'name': 'Survived', 'type': 'binary'}]

training = {'optimizer': {'type': 'adam', 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-08},
 'epochs': 100,
 'regularizer': 'l2',
 'regularization_lambda': 0,
 'learning_rate': 0.001,
 'batch_size': 128,
 'dropout_rate': 0.0,
 'early_stop': 5,
 'reduce_learning_rate_on_plateau': 0,
 'reduce_learning_rate_on_plateau_patience': 5,
 'reduce_learning_rate_on_plateau_rate': 0.5,
 'increase_batch_size_on_plateau': 0,
 'increase_batch_size_on_plateau_patience': 5,
 'increase_batch_size_on_plateau_rate': 2,
 'increase_batch_size_on_plateau_max': 512,
 'decay': False,
 'decay_steps': 10000,
 'decay_rate': 0.96,
 'staircase': False,
 'gradient_clipping': None,
 'validation_field': 'combined',
 'validation_measure': 'loss',
 'bucketing_field': None,
 'learning_rate_warmup_epochs': 5}


base_model = {'input_features': input_features, 'output_features': output_features, 'combiner': combiner, 'training': training}

def merge_dict(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    dct = copy.deepcopy(dct)
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], Mapping)):
            dct[k] = merge_dict(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


# simple example....try different sizes of fc after combiner, and try different batch_size
BASE = '/Users/benmackenzie/projects/Teradata/ludwig/examples/hyperparameters/'


def train(model_def, config, reporter):
    combiner = model_def['combiner']
    combiner = merge_dict(combiner, {'num_fc_layers': config['num_fc_layers']})
    training = model_def['training']
    training = merge_dict(training, {'batch_size': config['batch_size']})
    new_model_def = {'input_features': model_def['input_features'],
                     'output_features': model_def['output_features'],
                     'combiner': combiner,
                     'training': training}
    model = LudwigModel(new_model_def)

    train_stats = model.train(data_hdf5=BASE + 'titanic.hdf5', train_set_metadata_json=BASE + 'titanic.json')
    return reporter(mean_accuracy=train_stats['validation']['Survived']['accuracy'][-1], done=True)







from ray.tune import register_trainable, grid_search, run_experiments

#train(base_model, {}, None)

ray.init(local_mode=False)

grid_search_space = {
    'num_fc_layers': grid_search([1,2,3,4]),
    'batch_size': grid_search([4,16,32,64,128])
}

register_trainable('train', lambda cfg, rptr: train(base_model, cfg, rptr))
run_experiments({'my_experiment': {
    'run': 'train',
    'stop': {'mean_accuracy': 0.9},
    'config': grid_search_space}
    })

