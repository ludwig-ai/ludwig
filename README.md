![Ludwig logo](https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/ludwig_hero.png "Ludwig logo")

<div align="center">

[![PyPI version](https://badge.fury.io/py/ludwig.svg)](https://badge.fury.io/py/ludwig)
[![Build Status](https://github.com/ludwig-ai/ludwig/actions/workflows/pytest.yml/badge.svg)](https://github.com/ludwig-ai/ludwig/actions/workflows/pytest.yml)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/4210/badge)](https://bestpractices.coreinfrastructure.org/projects/4210)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)

[![DockerHub](https://img.shields.io/docker/pulls/ludwigai/ludwig.svg)](https://hub.docker.com/r/ludwigai)
[![Downloads](https://pepy.tech/badge/ludwig)](https://pepy.tech/project/ludwig)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/ludwig-ai/ludwig/blob/master/LICENSE)
[![Twitter](https://img.shields.io/twitter/follow/ludwig_ai.svg?style=social&logo=twitter)](https://twitter.com/ludwig_ai)

</div>

Translated in [🇰🇷 Korean](README_KR.md)/

# What is Ludwig?

Ludwig is a data-driven declarative ML framework hosted by the LF AI & Data Foundation that makes it easy to define
deep learning pipelines for many types of tasks with a simple and flexible data-driven configuration system.

![img](https://raw.githubusercontent.com/ludwig-ai/ludwig-docs/ludwig05/docs/images/ludwig_legos.gif)

Ludwig supports many different combinations of input and output types that solve many different machine learning tasks.

Ludwig supports users in their machine learning projects end-to-end; from experimenting with different training recipes,
exploring state-of-the-art model architectures, to scaling up to large out-of-memory datasets and multi-node clusters,
and finally serving the best model in production.

# Getting started

Check out the official [getting started guide](https://ludwig-ai.github.io/ludwig-docs/latest/getting_started/).

# How to Use Ludwig

Create a config that describes the schema of your data.

```yaml
input_features:
    -
        name: Pclass
        type: category
    -
        name: Sex
        type: category
    -
        name: Age
        type: number
        preprocessing:
            missing_value_strategy: fill_with_mean
    -
        name: SibSp
        type: number
    -
        name: Parch
        type: number
    -
        name: Fare
        type: number
        preprocessing:
            missing_value_strategy: fill_with_mean
    -
        name: Embarked
        type: category

output_features:
    -
        name: Survived
        type: binary
```

Simple commands can be used to train models and predict new data.

```sh
ludwig train --config config.yaml --dataset data.csv
ludwig predict --model_path results/experiment_run/model --dataset test.csv
```

A programmatic API is also available to use Ludwig from Python.

```python
from ludwig.api import LudwigModel
import pandas as pd

# train a model
config = {
    "input_features": [
        {"name": "Pclass", "type": "category"},
        {"name": "Sex", "type": "category"},
        {"name": "Age", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mean"}},
        {"name": "SibSp", "type": "number"},
        {"name": "Parch", "type": "number"},
        {"name": "Fare", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mean"}},
        {"name": "Embarked", "type": "category"},
    ],
    "output_features": [{"name": "Survived", "type": "binary"}],
}
model = LudwigModel(config)
data = pd.read_csv("data.csv")
train_stats, _, model_dir = model.train(data)

# or load a model
model = LudwigModel.load(model_dir)

# obtain predictions
predictions = model.predict(data)
```

A suite of visualization tools allows you to analyze models' training and test performance and to compare them.

```sh
ludwig visualize --visualization compare_performance --test_statistics path/to/test_statistics_model_1.json path/to/test_statistics_model_2.json
```

![img](https://raw.githubusercontent.com/ludwig-ai/ludwig-docs/ludwig05/docs/images/compare_performance.png)

Run hyperparameter optimization locally or using Ray Tune.

```sh
ludwig hyperopt --config rotten_tomatoes.yaml --dataset rotten_tomatoes.csv
```

Serve models using [FastAPI](https://fastapi.tiangolo.com/).

```sh
ludwig serve --model_path ./results/experiment_run/model
curl http://0.0.0.0:8000/predict -X POST -F "movie_title=Friends With Money" -F "content_rating=R" -F "genres=Art House & International, Comedy, Drama" -F "runtime=88.0" -F "top_critic=TRUE" -F "review_content=The cast is terrific, the movie isn't."
```

To learn more about [How Ludwig Works](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/how_ludwig_works/),
check out the official [Ludwig Docs](https://ludwig-ai.github.io/ludwig-docs/) for our guide on [how to get started](https://ludwig-ai.github.io/ludwig-docs/latest/getting_started/), or read our publications on [Ludwig](https://arxiv.org/pdf/1909.07930.pdf), [declarative ML](https://arxiv.org/pdf/2107.08148.pdf), and [Ludwig’s SoTA benchmarks](https://openreview.net/pdf?id=hwjnu6qW7E4).

If you are interested in contributing, have questions, comments, or thoughts to share, or if you just want to be in the
know, please consider [joining the Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ) and follow us on [Twitter](https://twitter.com/ludwig_ai)!

# Why Ludwig?

Ludwig is a profound utility for research scientists, data scientists, and machine learning engineers.

## Minimal Machine Learning Boilerplate

PyTorch and Tensorflow are popular libraries for deep learning research scientists who develop new training algorithms,
design and develop new model architectures, and run experiments with them.

However, experimenting with a new architecture often requires a formidable amount of code for scalably loading and
preprocessing data, and setting up pipelines for (distributed) training, evaluation, and hyperparameter optimization.

Ludwig takes care of the engineering complexity of deep learning out of the box, enabling research scientists to focus
on building models at the highest level of abstraction.

Without Ludwig, if you had a great idea for a novel architecture for image classification, you would implement your new
model as a PyTorch module.

```python
import torch.nn as nn


class MyImageClassificationModel(nn.Module):
    def __init__(self, hyperparameter_1, hyperparameter_2):
        # ...
        pass

    def forward(inputs):  # Image tensors
        # Encode image tensors.
        # ...

        # "Boilerplate" decoding code: reduce dimension to logits over num_classes.
        return outputs  # Output tensors
```

However, this is rather incomplete – we'll also need to figure out how to read images from disk with torchvision, write
a training-checkpoint-eval loop, post-process logits tensors into predictions, and compute metrics over predictions. All
these steps elongate model development time and introduce potential sources of error.

Instead of implementing all of this from scratch, research scientists can implement new models as PyTorch Modules in
Ludwig directly to take advantage of all of the engineering conveniences that Ludwig offers.

```python
import torch
from ludwig.constants import IMAGE
from ludwig.encoders.base import Encoder


@register("my_encoder", IMAGE)
class MyImageClassificationModel(Encoder):
    def __init__(self, hyperparameter_1, hyperparameter_2):
        # ...
        pass

    def forward(inputs):  # Preprocessed image tensors.
        # ...
        return outputs  # Dimension reduction, loss, and softmax handled by existing decoders.

    @property
    def input_shape() -> torch.Size:
        """Returns size of the input tensor without the batch dimension."""
        pass

    @property
    def output_shape() -> torch.Size:
        """Returns size of the output tensor without the batch dimension."""
        pass
```

The new encoder `my_encoder` can immediately be used in a new Ludwig configuration by just setting
`encoder: my_encoder`. Ludwig will take care of the rest of the pipeline for you!

## Seamless Benchmarking and Testing on Multiple Problems and Datasets

Benchmarking the new model against existing published and implemented models is a simple config change.

baseline_config.yaml

```yaml
input_features:
name: image_path
type: image
encoder: resnet
layers: 50
output_features:
name: class
type: category
```

```sh
ludwig experiment -–config baseline_config.yaml -–dataset my_dataset.csv
```

my_new_encoder_config.yaml

```yaml
input_features:
name: image_path
type: image
encoder: my_encoder
hyperparameter_1: #
hyperparameter_2: #
output_features:
name: class
type: category
```

```sh
ludwig experiment -–config my_new_encoder_config.yaml -–dataset my_dataset.csv
```

Now you can run exactly the same configuration as before, which guarantees exactly the same preprocessing, training and
evaluation is performed, to easily compare the performance of your new encoder, with new hyperparameters that can be set
directly in the config.

## Immediate integration with hyperparameter optimization on Ray Tune

Registered models and their constructor parameters can be used immediately alongside other Ludwig building blocks like
hyperparameter optimization.

```yaml
input_features:
-
    name: image_path
    type: image
    encoder: my_encoder
    hyperparameter_1: #
    hyperparameter_2: #
output_features:
-
    name: class
    type: category
hyperopt:
  parameters:
    image_path.hyperparameter_1:
      values: [10, 100, 1000]
    image_path.hyperparameter_2:
      low: 0.1
      high: 0.5
```

## Broadening to Multiple Problems and Datasets

Registered models can be subsequently applied across the extensive set of tasks and datasets that Ludwig supports.
Ludwig includes a [full benchmarking toolkit](https://arxiv.org/abs/2111.04260) accessible to any user, for running
experiments with multiple models across multiple datasets with just a simple configuration.

For more information on how to add your dataset or model to Ludwig, check out the [Developer Guide](https://ludwig-ai.github.io/ludwig-docs/latest/developer_guide) and the [Ludwig Dataset Zoo](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/datasets/dataset_zoo/).

## Low-code interface for state-of-the-art models, including pre-trained Huggingface Transformers

Ludwig strives to bring state of the art performance for any ML task without needing to write hundreds of lines of code.

Ludwig provides robust implementations of common architectures like CNNs, RNNs, Transformers, TabNet, MLP-Mixer.

Models can be trained from scratch, but Ludwig also natively integrates with pre-trained models, such as the ones
available in [Huggingface Transformers](https://huggingface.co/docs/transformers/index). Users can choose from a vast
collection of state-of-the-art pre-trained PyTorch models to use without needing to write any code at all. For example,
training a BERT-based sentiment analysis model with Ludwig is as simple as:

```sh
ludwig train -–dataset sst5 -–config_str “{input_features: [{name: sentence, type: text, encoder: bert}], output_features: [{name: label, type: category}]}”
```

## Low-code Interface for AutoML

[Ludwig AutoML](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/automl/) allows users to obtain trained models
by providing just a dataset, the target column, and a time budget. It’s as simple as that!

```python
auto_train_results = ludwig.automl.auto_train(
    dataset=my_dataset_df, target=target_column_name, time_limit_s=7200, tune_for_memory=False
)
```

## Highly Configurable Data Preprocessing, Modeling, and Metrics

Any and all aspects of the model architecture, training loop, hyperparameter search, and backend infrastructure can be
modified as additional fields in the declarative configuration to customize the pipeline to meet your requirements.

```yaml
input_features:
-
  name: title
  type: text
  encoder: rnn
  cell: lstm
  num_layers: 2
  state_size: 128
  preprocessing:
    tokenizer: space_punct
-
  name: author
  type: category
  embedding_size: 128
  preprocessing:
    most_common: 10000
-
  name: description
  type: text
  encoder: bert
-
  name: cover
  type: image
  encoder: resnet
  num_layers: 18

output_features:
-
  name: genre
  type: set
-
  name: price
  type: number
  preprocessing:
    normalization: zscore

trainer:
  epochs: 50
  batch_size: 256
  optimizer:
    type: adam
    beat1: 0.9
  learning_rate: 0.001

backend:
  type: local
  cache_format: parquet

hyperopt:
  metric: f1
  sampler: random
  parameters:
    title.num_layers:
      lower: 1
      upper: 5
    training.learning_rate:
      values: [0.01, 0.003, 0,001]
```

For details on what can be configured, check out
[Ludwig Configuration](https://ludwig-ai.github.io/ludwig-docs/latest/configuration/) docs. If a model, loss, evaluation
metric, preprocessing function or other parts of the pipeline are not already available, the modularity of the
underlying architecture allows users to very easily extend Ludwig’s capabilities by implementing simple abstract
interfaces, as described in the [Developer Guide](https://ludwig-ai.github.io/ludwig-docs/latest/developer_guide/).

## Effortless Device Management and Distributed Training

PyTorch provides great performance for training with one or multiple GPUs, including in our own benchmarks. However,
setting up GPU training in PyTorch adds the overhead of moving models, data, labels, and metrics to the right device.
Even for a single GPU, this overhead can be substantial for some types of advanced models, like we have experienced when
porting our TabNet implementation. The overhead is greater for distributed training on multiple GPUs, and even more so
when training on multiple nodes each with multiple GPUs.

Ludwig v0.5 brings effortless scaling to the PyTorch ecosystem. Training models on a single GPU, multiple GPUs, or
multiple nodes is as easy as setting a flag in the Ludwig configuration. The two code snippets below show examples of
using GPUs in PyTorch and Ludwig.

```python
import torch
import torch.nn as nn


inputs, labels = data

# create model
class Model(nn.Module):
    def __init__(self):
        # ...
        pass

    def forward(self, x):
        # Must manually move all new tensors to GPU.
        if torch.cuda.is_available():
            mask = torch.ones(x, device="cuda")
        # ...


# move model and data to the correct device
model = Model()
if torch.cuda.is_available():
    model = model.to("cuda")
    inputs = inputs.to("cuda")
    labels = labels.to("cuda")
```

In Ludwig, a GPU is used by default if it is available. However, explicitly setting GPU training is as easy as setting a
single parameter in the config.

```yaml
backend:
    trainer:
        use_gpu: true
```

Setting up distributed training on multiple GPUs or multiple nodes using [Horovod](https://github.com/horovod/horovod)
is a similarly trivial configuration change specifying the number of GPU workers.

```yaml
backend:
    trainer:
        use_gpu: true
        num_workers: 4  # Distributed training with 4 GPUs
```

## Easy Productionisation

Bringing machine learning models to fruition in production is usually a lengthy and complicated process. Ludwig provides a few paths to make the life of machine learning engineers responsible for deployment easier.

With [Ludwig Serving](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/serving/), Ludwig makes it easy to serve
deep learning models, including on GPUs. Use `ludwig serve` to launch a REST API for your trained Ludwig model.

```
ludwig serve --model_path=/path/to/model

curl http://0.0.0.0:8000/predict -X POST -F 'english_text=words to be translated'
```

For highly efficient deployments it’s often critical to minimize the overhead caused by the python runtime. Ludwig
supports exporting models to efficient Torschscript bundles.

```
ludwig export_torchscript -–model_path=/path/to/model
```

# Ludwig Principles

The core design principles baked into the toolbox are:

- No coding required: no coding skills are required to train a model, use it for obtaining predictions, and serving.
- Generalizability: a datatype-based approach to deep learning model design makes the tool usable across many different use cases.
- Flexibility: experienced users have extensive control over model building and training, while newcomers will find it easy to use.
- Extensibility: Easily add new model architecture and new feature datatypes.
- Understandability: Compelling visualizations to understand their performance and compare their predictions.
- Open Source: Apache License 2.0

<p><img src="https://raw.githubusercontent.com/lfai/artwork/master/lfaidata-assets/lfaidata/horizontal/color/lfaidata-horizontal-color.png" alt="LF AI & Data logo" width="200"/></p>

Ludwig is hosted by the Linux Foundation as part of the [LF AI & Data Foundation](https://lfaidata.foundation/). For
details about who's involved and how Ludwig fits into the larger open source AI landscape,
read the Linux Foundation [announcement](https://lfaidata.foundation/blog/2020/12/17/ludwig-joins-lf-ai--data-as-new-incubation-project/).

## Full documentation

You can find the full documentation [here](https://ludwig-ai.github.io/ludwig-docs/).

## License

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fuber%2Fludwig.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fuber%2Fludwig?ref=badge_large)

## Getting Involved

- [Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)
- [Twitter](https://twitter.com/ludwig_ai)
- [Medium](https://medium.com/ludwig-ai)
- [GitHub Issues](https://github.com/ludwig-ai/ludwig/issues)
