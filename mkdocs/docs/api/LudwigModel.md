<span style="float:right;">[[source]](https://github.com/uber/ludwig/blob/master/ludwig/api.py#L69)</span>
# LudwigModel class

```python
ludwig.api.LudwigModel(
  model_definition=None,
  model_definition_file=None,
  logging_level=40
)
```

Class that allows access to high level Ludwig functionalities.

__Inputs__


- __model_definition__ (dict): a dictionary containing information needed
   to build a model. Refer to the [User Guide]
   (http://ludwig.ai/user_guide/#model-definition) for details.
- __model_definition_file__ (string, optional, default: `None`): path to
   a YAML file containing the model definition. If available it will be
   used instead of the model_definition dict.
- __logging_level__ (int, default: `logging.ERROR`): logging level to use
   for logging. Use logging constants like `logging.DEBUG`,
   `logging.INFO` and `logging.ERROR`. By default only errors will be
   printed. It is possible to change the logging_level later by using
   the set_logging_level method.

__Example usage:__


```python
from ludwig.api import LudwigModel
```

Train a model:

```python
model_definition = {...}
ludwig_model = LudwigModel(model_definition)
train_stats = ludwig_model.train(data_csv=csv_file_path)
```

or

```python
train_stats = ludwig_model.train(data_df=dataframe)
```

If you have already trained a model you can load it and use it to predict

```python
ludwig_model = LudwigModel.load(model_dir)
```

Predict:

```python
predictions = ludwig_model.predict(data_csv=csv_file_path)
```

or

```python
predictions = ludwig_model.predict(data_df=dataframe)
```

Test:

```python
predictions, test_stats = ludwig_model.test(data_csv=csv_file_path)
```

or

```python
predictions, test_stats = ludwig_model.predict(data_df=dataframe)
```

Finally in order to release resources:

```python
model.close()
```


---
# LudwigModel methods

## close


```python
close(
)
```


Closes an open LudwigModel (closing the session running it).
It should be called once done with the model to release resources.

---
## initialize_model


```python
initialize_model(
  train_set_metadata=None,
  train_set_metadata_json=None,
  gpus=None,
  gpu_fraction=1,
  random_seed=42,
  debug=False
)
```


This function initializes a model. It is need for performing online
learning, so it has to be called before `train_online`.
`train` initialize the model under the hood, so there is no need to call
this function if you don't use `train_online`.

__Inputs__


- __train_set_metadata__ (dict): it contains metadata information for
   the input and output features the model is going to be trained
   on. It's the same content of the metadata json file that is
   created while training.
- __train_set_metadata_json__ (string):  path to the JSON metadata file
   created while training. it contains metadata information for the
   input and output features the model is going to be trained on
- __gpus__ (string, default: `None`): list of GPUs to use (it uses the
   same syntax of CUDA_VISIBLE_DEVICES)
- __gpu_fraction__ (float, default `1.0`): fraction of GPU memory to
   initialize the process with
- __random_seed__ (int, default`42`): a random seed that is going to be
   used anywhere there is a call to a random number generator: data
   splitting, parameter initialization and training set shuffling
- __debug__ (bool, default: `False`): enables debugging mode
 
---
## load


```python
load(
  model_dir
)
```


This function allows for loading pretrained models


__Inputs__


- __model_dir__ (string): path to the directory containing the model.
   If the model was trained by the `train` or `experiment` command,
   the model is in `results_dir/experiment_dir/model`.


__Return__


- __return__ (LudwigModel): a LudwigModel object


__Example usage__


```python
ludwig_model = LudwigModel.load(model_dir)
```


---
## predict


```python
ludwig.predict(
  data_df=None,
  data_csv=None,
  data_dict=None,
  return_type=<class 'pandas.core.frame.DataFrame'>,
  batch_size=128,
  gpus=None,
  gpu_fraction=1,
  skip_save_unprocessed_output=True
)
```


This function is used to predict the output variables given the input
variables using the trained model.

__Inputs__


- __data_df__ (DataFrame): dataframe containing data. Only the input
   features defined in the model definition need to be present in
   the dataframe.
- __data_csv__ (string): input data CSV file. Only the input features
   defined in the model definition need to be present in the CSV.
- __data_dict__ (dict): input data dictionary. It is expected to
   contain one key for each field and the values have to be lists
   of the same length. Each index in the lists corresponds to one
   datapoint. Only the input features defined in the model
   definition need to be present in the dataframe. For example a
   data set consisting of two datapoints with a input text may be
   provided as the following dict ``{'text_field_name}: ['text of
   the first datapoint', text of the second datapoint']}`.
- __return_type__ (strng or type, default: `DataFrame`):
   string describing the type of the returned prediction object.
   `'dataframe'`, `'df'` and `DataFrame` will return a pandas
   DataFrame , while `'dict'`, ''dictionary'` and `dict` will
   return a dictionary.
- __batch_size__ (int, default: `128`): batch size
- __skip_save_unprocessed_output__ (skip_save_unprocessed_output: If this parameter is False):skip_save_unprocessed_output: If this parameter is False,
   predictions and their probabilities are saved in both raw
   unprocessed numpy files contaning tensors and as postprocessed
   CSV files (one for each output feature). If this parameter is
   True, only the CSV ones are saved and the numpy ones are skipped.
- __gpus__ (string, default: `None`): list of GPUs to use (it uses the
   same syntax of CUDA_VISIBLE_DEVICES)
- __gpu_fraction__ (float, default `1.0`): fraction of gpu memory to
   initialize the process with

__Return__


- __return__ (DataFrame or dict): a dataframe containing the predictions for
     each output feature and their probabilities (for types that
     return them) will be returned. For instance in a 3 way
     multiclass classification problem with a category field names
     `class` as output feature with possible values `one`, `two`
     and `three`, the dataframe will have as many rows as input
     datapoints and five columns: `class_predictions`,
     `class_UNK_probability`, `class_one_probability`,
     `class_two_probability`, `class_three_probability`. (The UNK
     class is always present in categorical features).
     If the `return_type` is a dictionary, the returned object be
     a dictionary contaning one entry for each output feature.
     Each entry is itself a dictionary containing aligned
     arrays of predictions and probabilities / scores.
 
---
## save


```python
save(
  save_path
)
```


This function allows to save models on disk

__Inputs__


- __ save_path__ (string): path to the directory where the model is
    going to be saved. Both a JSON file containing the model
    architecture hyperparameters and checkpoints files containing
    model weights will be saved.


__Example usage__


```python
ludwig_model.save(save_path)
```


---
## save_for_serving


```python
save_for_serving(
  save_path
)
```


This function allows to save models on disk

__Inputs__


- __ save_path__ (string): path to the directory where the SavedModel
    is going to be saved.


__Example usage__


```python
ludwig_model.save_for_serving(save_path)
```


---
## set_logging_level


```python
set_logging_level(
  logging_level
)
```



:param logging_level: Set/Update the logging level. Use logging
constants like `logging.DEBUG` , `logging.INFO` and `logging.ERROR`.

:return: None

---
## test


```python
ludwig.test(
  data_df=None,
  data_csv=None,
  data_dict=None,
  return_type=<class 'pandas.core.frame.DataFrame'>,
  batch_size=128,
  skip_save_unprocessed_output=False,
  gpus=None,
  gpu_fraction=1
)
```


This function is used to predict the output variables given the input
variables using the trained model and compute test statistics like
performance measures, confusion matrices and the like.


__Inputs__


- __data_df__ (DataFrame): dataframe containing data. Both input and
   output features defined in the model definition need to be
   present in the dataframe.
- __data_csv__ (string): input data CSV file. Both input and output
   features defined in the model definition need to be present in
   the CSV.
- __data_dict__ (dict): input data dictionary. It is expected to
   contain one key for each field and the values have to be lists
   of the same length. Each index in the lists corresponds to one
   datapoint. Both input and output features defined in the model
   definition need to be present in the dataframe. For example a
   data set consisting of two datapoints with a input text may be
   provided as the following dict ``{'text_field_name}: ['text of
   the first datapoint', text of the second datapoint']}`.
- __return_type__ (strng or type, default: `DataFrame`):
   string describing the type of the returned prediction object.
   `'dataframe'`, `'df'` and `DataFrame` will return a pandas
   DataFrame , while `'dict'`, ''dictionary'` and `dict` will
   return a dictionary.
- __batch_size__ (int, default: `128`): batch size
- __skip_save_unprocessed_output__ (skip_save_unprocessed_output: If this parameter is False):skip_save_unprocessed_output: If this parameter is False,
   predictions and their probabilities are saved in both raw
   unprocessed numpy files contaning tensors and as postprocessed
   CSV files (one for each output feature). If this parameter is
   True, only the CSV ones are saved and the numpy ones are skipped.
- __gpus__ (string, default: `None`): list of GPUs to use (it uses the
   same syntax of CUDA_VISIBLE_DEVICES)
- __gpu_fraction__ (float, default `1.0`): fraction of GPU memory to
   initialize the process with

__Return__


- __return__ (tuple((DataFrame or dict), dict)): a tuple of a dataframe and a
     dictionary. The dataframe contains the predictions for each
     output feature and their probabilities (for types that return
     them) will be returned. For instance in a 3 way multiclass
     classification problem with a category field names `class` as
     output feature with possible values `one`, `two` and `three`,
     the dataframe will have as many rows as input datapoints and
     five columns: `class_predictions`, `class_UNK_probability`,
     `class_one_probability`, `class_two_probability`,
     `class_three_probability`. (The UNK class is always present in
     categorical features).
     If the `return_type` is a dictionary, the first object
     of the tuple will be a dictionary contaning one entry
     for each output feature.
     Each entry is itself a dictionary containing aligned
     arrays of predictions and probabilities / scores.
     The second object of the tuple is a dictionary that contains
     the test statistics, with each key being the name of an output
     feature and the values being dictionaries containing measures
     names and their values.
 
---
## train


```python
train(
  data_df=None,
  data_train_df=None,
  data_validation_df=None,
  data_test_df=None,
  data_csv=None,
  data_train_csv=None,
  data_validation_csv=None,
  data_test_csv=None,
  data_hdf5=None,
  data_train_hdf5=None,
  data_validation_hdf5=None,
  data_test_hdf5=None,
  data_dict=None,
  data_train_dict=None,
  data_validation_dict=None,
  data_test_dict=None,
  train_set_metadata_json=None,
  experiment_name='api_experiment',
  model_name='run',
  model_load_path=None,
  model_resume_path=None,
  skip_save_training_description=False,
  skip_save_training_statistics=False,
  skip_save_model=False,
  skip_save_progress=False,
  skip_save_log=False,
  skip_save_processed_input=False,
  output_directory='results',
  gpus=None,
  gpu_fraction=1.0,
  use_horovod=False,
  random_seed=42,
  debug=False
)
```


This function is used to perform a full training of the model on the
specified dataset.

__Inputs__


- __data_df__ (DataFrame): dataframe containing data. If it has a split
   column, it will be used for splitting (0: train, 1: validation,
   2: test), otherwise the dataset will be randomly split
- __data_train_df__ (DataFrame): dataframe containing training data
- __data_validation_df__ (DataFrame): dataframe containing validation
   data
- __data_test_df__ (DataFrame dataframe containing test dat):data_test_df: (DataFrame dataframe containing test data
- __data_csv__ (string): input data CSV file. If it has a split column,
   it will be used for splitting (0: train, 1: validation, 2: test),
   otherwise the dataset will be randomly split
- __data_train_csv__ (string): input train data CSV file
- __data_validation_csv__ (string): input validation data CSV file
- __data_test_csv__ (string): input test data CSV file
- __data_hdf5__ (string): input data HDF5 file. It is an intermediate
   preprocess  version of the input CSV created the first time a CSV
   file is used in the same directory with the same name and a hdf5
   extension
- __data_train_hdf5__ (string): input train data HDF5 file. It is an
   intermediate preprocess  version of the input CSV created the
   first time a CSV file is used in the same directory with the same
   name and a hdf5 extension
- __data_validation_hdf5__ (string): input validation data HDF5 file.
   It is an intermediate preprocess version of the input CSV created
   the first time a CSV file is used in the same directory with the
   same name and a hdf5 extension
- __data_test_hdf5__ (string): input test data HDF5 file. It is an
   intermediate preprocess  version of the input CSV created the
   first time a CSV file is used in the same directory with the same
   name and a hdf5 extension
- __data_dict__ (dict): input data dictionary. It is expected to
   contain one key for each field and the values have to be lists of
   the same length. Each index in the lists corresponds to one
   datapoint. For example a data set consisting of two datapoints
   with a text and a class may be provided as the following dict
   `{'text_field_name': ['text of the first datapoint', text of the
   second datapoint'], 'class_filed_name': ['class_datapoints_1',
   'class_datapoints_2']}`.
- __data_train_dict__ (dict): input training data dictionary. It is
   expected to contain one key for each field and the values have
   to be lists of the same length. Each index in the lists
   corresponds to one datapoint. For example a data set consisting
   of two datapoints with a text and a class may be provided as the
   following dict:
   `{'text_field_name': ['text of the first datapoint', 'text of the
   second datapoint'], 'class_field_name': ['class_datapoint_1',
   'class_datapoint_2']}`.
- __data_validation_dict__ (dict): input validation data dictionary. It
   is expected to contain one key for each field and the values have
   to be lists of the same length. Each index in the lists
   corresponds to one datapoint. For example a data set consisting
   of two datapoints with a text and a class may be provided as the
   following dict:
   `{'text_field_name': ['text of the first datapoint', 'text of the
   second datapoint'], 'class_field_name': ['class_datapoint_1',
   'class_datapoint_2']}`.
- __data_test_dict__ (dict): input test data dictionary. It is
   expected to contain one key for each field and the values have
   to be lists of the same length. Each index in the lists
   corresponds to one datapoint. For example a data set consisting
   of two datapoints with a text and a class may be provided as the
   following dict:
   `{'text_field_name': ['text of the first datapoint', 'text of the
   second datapoint'], 'class_field_name': ['class_datapoint_1',
   'class_datapoint_2']}`.
- __train_set_metadata_json__ (string): input metadata JSON file. It is an
   intermediate preprocess file containing the mappings of the input
   CSV created the first time a CSV file is used in the same
   directory with the same name and a json extension
- __experiment_name__ (string): a name for the experiment, used for the save
   directory
- __model_name__ (string): a name for the model, used for the save
   directory
- __model_load_path__ (string): path of a pretrained model to load as
   initialization
- __model_resume_path__ (string): path of a the model directory to
   resume training of
- __skip_save_training_description__ (bool, default: `False`): disables
   saving the description JSON file.
- __skip_save_training_statistics__ (bool, default: `False`): disables
   saving training statistics JSON file.
- __skip_save_model__ (bool, default: `False`): disables
   saving model weights and hyperparameters each time the model
   improves. By default Ludwig saves model weights after each epoch
   the validation measure imrpvoes, but if the model is really big
   that can be time consuming if you do not want to keep
   the weights and just find out what performance can a model get
   with a set of hyperparameters, use this parameter to skip it,
   but the model will not be loadable later on.
- __skip_save_progress__ (bool, default: `False`): disables saving
   progress each epoch. By default Ludwig saves weights and stats
   after each epoch for enabling resuming of training, but if
   the model is really big that can be time consuming and will uses
   twice as much space, use this parameter to skip it, but training
   cannot be resumed later on.
- __skip_save_log__ (bool, default: `False`): disables saving TensorBoard
   logs. By default Ludwig saves logs for the TensorBoard, but if it
   is not needed turning it off can slightly increase the
   overall speed.
- __skip_save_processed_input__ (bool, default: `False`): skips saving
   intermediate HDF5 and JSON files
- __output_directory__ (string, default: `'results'`): directory that
   contains the results
- __gpus__ (string, default: `None`): list of GPUs to use (it uses the
   same syntax of CUDA_VISIBLE_DEVICES)
- __gpu_fraction__ (float, default `1.0`): fraction of gpu memory to
   initialize the process with
- __random_seed__ (int, default`42`): a random seed that is going to be
   used anywhere there is a call to a random number generator: data
   splitting, parameter initialization and training set shuffling
- __debug__ (bool, default: `False`): enables debugging mode

There are three ways to provide data: by dataframes using the `_df`
parameters, by CSV using the `_csv` parameters and by HDF5 and JSON,
using `_hdf5` and `_json` parameters.
The DataFrame approach uses data previously obtained and put in a
dataframe, the CSV approach loads data from a CSV file, while HDF5 and
JSON load previously preprocessed HDF5 and JSON files (they are saved in
the same directory of the CSV they are obtained from).
For all three approaches either a full dataset can be provided (which
will be split randomly according to the split probabilities defined in
the model definition, by default 70% training, 10% validation and 20%
test) or, if it contanins a plit column, it will be plit according to
that column (interpreting 0 as training, 1 as validation and 2 as test).
Alternatively separated dataframes / CSV / HDF5 files can beprovided
for each split.

During training the model and statistics will be saved in a directory
`[output_dir]/[experiment_name]_[model_name]_n` where all variables are
resolved to user spiecified ones and `n` is an increasing number
starting from 0 used to differentiate different runs.


__Return__


- __return__ (dict): a dictionary containing training statistics for each
output feature containing loss and measures values for each epoch.
 
---
## train_online


```python
train_online(
  data_df=None,
  data_csv=None,
  data_dict=None,
  batch_size=None,
  learning_rate=None,
  regularization_lambda=None,
  dropout_rate=None,
  bucketing_field=None,
  gpus=None,
  gpu_fraction=1
)
```


This function is used to perform one epoch of training of the model
on the specified dataset.

__Inputs__


- __data_df__ (DataFrame): dataframe containing data.
- __data_csv__ (string): input data CSV file.
- __data_dict__ (dict): input data dictionary. It is expected to
   contain one key for each field and the values have to be lists of
   the same length. Each index in the lists corresponds to one
   datapoint. For example a data set consisting of two datapoints
   with a text and a class may be provided as the following dict
   ``{'text_field_name': ['text of the first datapoint', text of the
   second datapoint'], 'class_filed_name': ['class_datapoints_1',
   'class_datapoints_2']}`.
- __batch_size__ (int): the batch size to use for training. By default
   it's the one specified in the model definition.
- __learning_rate__ (float): the learning rate to use for training. By
   default the values is the one specified in the model definition.
- __regularization_lambda__ (float): the regularization lambda
   parameter to use for training. By default the values is the one
   specified in the model definition.
- __dropout_rate__ (float): the dropout rate to use for training. By
   default the values is the one specified in the model definition.
- __bucketing_field__ (string): the bucketing field to use for
   bucketing the data. By default the values is one specified in the
   model definition.
- __gpus__ (string, default: `None`): list of GPUs to use (it uses the
   same syntax of CUDA_VISIBLE_DEVICES)
- __gpu_fraction__ (float, default `1.0`): fraction of GPU memory to
   initialize the process with

There are three ways to provide data: by dataframes using the `data_df`
parameter, by CSV using the `data_csv` parameter and by dictionary,
using the `data_dict` parameter.

The DataFrame approach uses data previously obtained and put in a
dataframe, the CSV approach loads data from a CSV file, while dict
approach uses data organized by keys representing columns and values
that are lists of the datapoints for each. For example a data set
consisting of two datapoints with a text and a class may be provided as
the following dict ``{'text_field_name}: ['text of the first datapoint',
text of the second datapoint'], 'class_filed_name':
['class_datapoints_1', 'class_datapoints_2']}`.

----

# Module functions

----

## kfold_cross_validate


```python
ludwig.api.kfold_cross_validate(
  k_fold,
  model_definition=None,
  model_definition_file=None,
  data_csv=None,
  output_directory='results',
  random_seed=42
)
```


Performs k-fold cross validation and returns result data structures.


__Inputs__


- __k_fold__ (int): number of folds to create for the cross-validation
- __model_definition__ (dict, default: None): a dictionary containing
   information needed to build a model. Refer to the
   [User Guide](http://ludwig.ai/user_guide/#model-definition)
   for details.
- __model_definition_file__ (string, optional, default: `None`): path to
   a YAML file containing the model definition. If available it will be
   used instead of the model_definition dict.
- __data_csv__ (dataframe, default: None):
- __data_csv__ (string, default: None):
- __output_directory__ (string, default: 'results'):
- __random_seed__ (int): Random seed used k-fold splits.

__Return__


- __return__ (tuple(kfold_cv_stats, kfold_split_indices), dict): a tuple of
    dictionaries `kfold_cv_stats`: contains metrics from cv run.
     `kfold_split_indices`: indices to split training data into
     training fold and test fold.
 