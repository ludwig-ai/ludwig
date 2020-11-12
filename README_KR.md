![Ludwig logo](https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/ludwig_hero.png "Ludwig logo")

<div align="center">

[![PyPI version](https://badge.fury.io/py/ludwig.svg)](https://badge.fury.io/py/ludwig)
[![Downloads](https://pepy.tech/badge/ludwig)](https://pepy.tech/project/ludwig)
[![Build Status](https://travis-ci.com/uber/ludwig.svg?branch=master)](https://travis-ci.com/github/uber/ludwig)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/uber/ludwig/blob/master/LICENSE)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fuber%2Fludwig.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fuber%2Fludwig?ref=badge_shield)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/4210/badge)](https://bestpractices.coreinfrastructure.org/projects/4210)

</div>

Ludwig은 사용자들이 코드를 작성할 필요 없이 딥러닝 모델을 학습시키고 사용할 수 있게 하는 TensorFlow 기반으로 만들어진 toolbox입니다.

이 모델을 학습시키기 위해서는 입출력 데이터가 들어있는 파일이 제공되어야 합니다. 나머지는 Ludwig가 알아서 처리해줄 것입니다. 간단한 명령어들은 단일기기 혹은 분산기기를 통해 모델을 학습하는 데에 사용될 수 있고, 또한 새로운 데이터를 예측하는 데에 사용됩니다.

파이썬 프로그램 API 또한 Ludwig에서 사용 가능합니다. 시각화 기능의 모음은 모델 훈련을 분석하고 모델 성능을 test하고 그것들을 비교하는 것을 가능하게 해줍니다.

Ludwig는 확장성을 염두에 두고 설계되었으며, 데이터 타입 추상화에 기반을 두고 있어서 새로운 모델 아키텍처뿐만 아니라 새로운 데이터 타입에 대한 지원을 쉽게 추가할 수 있습니다.

Ludwig은 사용자들이 딥러닝 모델을 빠르게 학습하고 테스트하는 것은 물론, 연구자들이 딥러닝 모델과 비교할 수 있는 강력한 기준과 동일한 데이터 평가를 통해 비교 가능성을 보장하는 테스트 설정을 확보할 수 있도록 활용됩니다.

Ludwig는 특정 use case의 end-to-end 모델을 작성하기 위해 조합할 수 있는 일련의 모델 아키텍처를 제공합니다. 도시를 설계하는 것을 예로 들어, 딥러닝 라이브러리가 도시 내부 건물의 구성요소(기둥, 바닥, 등)를 제공하고 있다면, Ludwig는 도시를 구성하는 건물을 제공하고 있습니다. 그러면 사용자는 도시 내부의 만들어진 건물 중 하나를 선택해 사용하거나, 새로운 건물을 추가할 수 있습니다.

toolbox에 반영된 핵심 설계 원리는 아래와 같습니다:
- No coding required: 모델을 학습시키고 예측된 데이터를 얻는 데에 코딩 스킬이 필요하지 않습니다.
- Generality: 딥러닝 모델 설계에 대한 새로운 데이터 유형 기반 접근방식은 다양한 use case들에 적용할 수 있는 tool을 만들어줍니다.
- Flexibility: 숙련된 사용자들은 모델 제작과 훈련을 광범위하게 제어하는 반면, 초보자들은 그것을 쉽게 사용할 수 있습니다.
- Extensibility: 새로운 모델 아키텍처와 새로운 데이터타입을 쉽게 추가할 수 있습니다.
- Understandability: 종종 딥러닝 모델 내부는 진행상황을 확인할 수 없는 것처럼 여겨지지만, Ludwig는 성능을 이해하고 예측된 데이터들을 비교하기위한 표준 시각화 기능을 제공합니다.
- Open Source: Apache License 2.0


Installation
============

Ludwig는 Python 3.6이상 버전을 요구합니다. 만약 Python 3가 설치 되어있지 않으면 다음 명령어를 이용해서 설치하세요.

```
sudo apt install python3  # on ubuntu
brew install python3      # on mac
```

만약 [파이썬 가상환경](https://docs.python-guide.org/dev/virtualenvs/)에서 사용하고 싶다면 아래 명령어를 사용하세요.

```
virtualenv -p python3 venv
```

Ludwig를 설치하려면 아래 명령어를 사용하세요.

```
pip install ludwig
```

위의 명령어는 Ludwig을 실행하기 위해 필요한 파일만 설치하게 되며, 더 많은 기능이 필요하다면 아래와 같은 파일들을 설치하면 됩니다.
 - `ludwig[text]` for text dependencies.
 - `ludwig[audio]` for audio and speech dependencies.
 - `ludwig[image]` for image dependencies.
 - `ludwig[hyperopt]` for hyperparameter optimization dependencies.
 - `ludwig[horovod]` for distributed training dependencies.
 - `ludwig[serve]` for serving dependencies.
 - `ludwig[viz]` for visualization dependencies.
 - `ludwig[test]` for dependencies needed for testing.

Distributed training is supported with [Horovod](https://github.com/horovod/horovod), which can be installed with `pip install ludwig[horovod]` or `HOROVOD_GPU_OPERATIONS=NCCL pip install ludwig[horovod]` for GPU support.
See Horovod's [installation guide](https://horovod.readthedocs.io/en/stable/install_include.html)  for full details on available installation options.

Any combination of extra packages can be installed at the same time with `pip install ludwig[extra1,extra2,...]` like for instance `pip install ludwig[text,viz]`.
The full set of dependencies can be installed with `pip install ludwig[full]`.

For developers who wish to build the source code from the repository:

```
git clone git@github.com:uber/ludwig.git
cd ludwig
virtualenv -p python3 venv
source venv/bin/activate
pip install -e '.[test]'
```

**Note:** that if you are running without GPUs, you may wish to use the CPU-only version of TensorFlow, 
which takes up much less space on disk.
To use a CPU-only TensorFlow version, uninstall `tensorflow` and  replace it with `tensorflow-cpu` after having installed `ludwig`.
Be sure to install a version within the compatible range as shown in `requirements.txt`.


Basic Principles
----------------

Ludwig provides three main functionalities: training models and using them to predict and evaluate them.
It is based on datatype abstraction, so that the same data preprocessing and postprocessing will be performed on different datasets that share datatypes and the same encoding and decoding models developed can be re-used across several tasks.

Training a model in Ludwig is pretty straightforward: you provide a dataset file and a config YAML file.

The config contains a list of input features and output features, all you have to do is specify names of the columns in the dataset that are inputs to your model alongside with their datatypes, and names of columns in the dataset that will be outputs, the target variables which the model will learn to predict.
Ludwig will compose a deep learning model accordingly and train it for you.

Currently, the available datatypes in Ludwig are:

- binary
- numerical
- category
- set
- bag
- sequence
- text
- timeseries
- image
- audio
- date
- h3
- vector

By choosing different datatype for inputs and outputs, users can solve many different tasks, for instance:

- text input + category output = text classifier
- image input + category output = image classifier
- image input + text output = image captioning
- audio input + binary output = speaker verification
- text input + sequence output = named entity recognition / summarization
- category, numerical and binary inputs + numerical output = regression
- timeseries input + numerical output = forecasting model
- category, numerical and binary inputs + binary output = fraud detection

take a look at the [Examples](https://ludwig-ai.github.io/ludwig-docs/examples/) to see how you can use Ludwig for several more tasks.

The config can contain additional information, in particular how to preprocess each column in the data, which encoder and decoder to use for each one, architectural  and training parameters, hyperparameters to optimize.
This allows ease of use for novices and flexibility for experts.


Training
--------

For example, given a text classification dataset like the following:

| doc_text                              | class    |
|---------------------------------------|----------|
| Former president Barack Obama ...     | politics |
| Juventus hired Cristiano Ronaldo ...  | sport    |
| LeBron James joins the Lakers ...     | sport    |
| ...                                   | ...      |

you want to learn a model that uses the content of the `doc_text` column as input to predict the values in the `class` column.
You can use the following config:

```yaml
{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}
```

and start the training typing the following command in your console:

```
ludwig train --dataset path/to/file.csv --config "{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}"
```

where `path/to/file.csv` is the path to a UTF-8 encoded CSV file containing the dataset in the previous table (many other data formats are supported).
Ludwig will:

1. Perform a random split of the data.
2. Preprocess the dataset.
3. Build a ParallelCNN model (the default for text features) that decodes output classes through a softmax classifier.
4. Train the model on the training set until the performance on the validation set stops improving.

Training progress will be displayed in the console, but the TensorBoard can also be used.

If you prefer to use an RNN encoder and increase the number of epochs to train for, all you have to do is to change the config to:

```yaml
{input_features: [{name: doc_text, type: text, encoder: rnn}], output_features: [{name: class, type: category}], training: {epochs: 50}}
```

Refer to the [User Guide](https://ludwig-ai.github.io/ludwig-docs/user_guide/) to find out all the options available to you in the config and take a look at the [Examples](https://ludwig-ai.github.io/ludwig-docs/examples/) to see how you can use Ludwig for several different tasks.

After training, Ludwig will create a `results` directory containing the trained model with its hyperparameters and summary statistics of the training process.
You can visualize them using one of the several visualization options available in the `visualize` tool, for instance:

```
ludwig visualize --visualization learning_curves --training_statistics path/to/training_statistics.json
```

This command will display a graph like the following, where you can see loss and accuracy during the training process:

![Learning Curves](https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/getting_started_learning_curves.png "Learning Curves")

Several more visualizations are available, please refer to [Visualizations](https://ludwig-ai.github.io/ludwig-docs/user_guide/#visualizations) for more details.


Distributed Training
--------------------

You can distribute the training of your models using [Horovod](https://github.com/horovod/horovod), which allows training on a single machine with multiple GPUs as well as on multiple machines with multiple GPUs.
Refer to the [User Guide](https://ludwig-ai.github.io/ludwig-docs/user_guide/#distributed-training) for full details.


Prediction and Evaluation
-------------------------

If you want your previously trained model to predict target output values on new data, you can type the following command in your console:

```
ludwig predict --dataset path/to/data.csv --model_path /path/to/model
```

Running this command will return model predictions.

If your dataset also contains ground truth values of the target outputs, you can compare them to the predictions obtained from the model to evaluate the model performance.

```
ludwig evaluate --dataset path/to/data.csv --model_path /path/to/model
```

This will produce evaluation performance statistics that can be visualized by the `visualize` tool, which can also be used to compare performances and predictions of different models, for instance:

```
ludwig visualize --visualization compare_performance --test_statistics path/to/test_statistics_model_1.json path/to/test_statistics_model_2.json
```

will return a bar plot comparing the models on different metrics:

![Performance Comparison](https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/compare_performance.png "Performance Comparison")

A handy `ludwig experiment` command that performs training and prediction one after the other is also available.


Programmatic API
----------------

Ludwig also provides a simple programmatic API that allows you to train or load a model and use it to obtain predictions on new data:

```python
from ludwig.api import LudwigModel

# train a model
config = {...}
model = LudwigModel(config)
train_stats = model.train(training_data)

# or load a model
model = LudwigModel.load(model_path)

# obtain predictions
predictions = model.predict(test_data)
```

`config` containing the same information of the YAML file provided to the command line interface.
More details are provided in the [User Guide](https://ludwig-ai.github.io/ludwig-docs/user_guide/) and in the [API documentation](https://ludwig-ai.github.io/ludwig-docs/api/).


Extensibility
-------------

Ludwig is built from the ground up with extensibility in mind.
It is easy to add an additional datatype that is not currently supported by adding a datatype-specific implementation of abstract classes that contain functions to preprocess the data, encode it, and decode it.

Furthermore, new models, with their own specific hyperparameters, can be easily added by implementing a class that accepts tensors (of a specific rank, depending on the datatype) as inputs and provides tensors as output.
This encourages reuse and sharing new models with the community.
Refer to the [Developer Guide](https://ludwig-ai.github.io/ludwig-docs/developer_guide/) for further details.


Full documentation
------------------

You can find the full documentation [here](https://ludwig-ai.github.io/ludwig-docs).


License
-------

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fuber%2Fludwig.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fuber%2Fludwig?ref=badge_large)
