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

Ludwig is a data-driven declarative ML framework that makes it easy to define deep learning pipelines for many types of tasks using a simple and flexible data-driven configuration system. It was open sourced by Uber and is hosted by the Linux Foundation AI & Data.

A data-driven configuration system allows users to define their deep learning pipeline by providing a list of inputs and outputs with their data type. The data type determines how inputs and outputs are preprocessed, encoded, decoded and which metrics and losses to use. Ludwig will assemble and train a deep learning model based on the combination of data types for many machine learning tasks.

![img](https://raw.githubusercontent.com/ludwig-ai/ludwig-docs/ludwig05/docs/images/ludwig_legos.gif)

Configurations are simple and flexible, enabling deep control of every aspect of the end-to-end pipeline; from experimenting with different training recipes, exploring state-of-the-art model architectures, to scaling up to large out-of-memory datasets and multi-node clusters, and finally serving the best model in production -- all can be achieved through small configuration changes.

Finally, the use of abstract interfaces throughout the codebase makes it easy for users to extend Ludwig by adding new models, metrics, losses, preprocessing functions and register them to make them available immediately in the configuration system.

# Main Features

- [Data-Driven configuration system](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/how_ludwig_works)

  A config YAML file that describes the schema of your data (input features, output features, and their types) is all you need to start training deep learning models. Ludwig uses declared features to compose a deep learning model accordingly.

- [Training, Prediction, and Evaluation](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/command_line_interface)

  Simple commands can be used to train models and predict new data.

  ```
  ludwig train --config config.yaml --dataset data.csv
  ludwig predict --model_path results/experiment_run/model --dataset test.csv
  ludwig eval --model_path results/experiment_run/model --dataset test.csv
  ```

- [Distributed Training](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/distributed_training)

  Train models in a distributed setting using [Horovod](https://github.com/horovod/horovod), which allows training on a single machine with multiple GPUs or multiple machines with multiple GPUs.

- [Serving](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/serving)

  ```
  Serve models using FastAPI.
  ```

  > ludwig serve --model_path ./results/experiment_run/model
  > curl http://0.0.0.0:8000/predict -X POST -F "movie_title=Friends With Money" -F "content_rating=R" -F "genres=Art House & International, Comedy, Drama" -F "runtime=88.0" -F "top_critic=TRUE" -F "review_content=The cast is terrific, the movie isn't."

- [Hyperparameter optimization](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/hyperopt)

  Ludwig supports hyperparameter optimization using Ray Tune or a local executor.

- [AutoML](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/automl)

  Ludwig AutoML takes a dataset, the target column, and a time budget, and returns a trained Ludwig model.

- [Programmatic API](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/api/LudwigModel)

  Ludwig also provides a simple programmatic API for all of the functionality described above and more.

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

- [Third-Party Integrations](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/integrations)

  Ludwig provides an extendable interface to integrate with third-party systems for tracking experiments. Third-party integrations exist for Comet ML, Weights & Biases, WhyLabs and MLFlow.

- [Extensibility](https://ludwig-ai.github.io/ludwig-docs/latest/developer_guide)

  Ludwig is built from the ground up with extensibility in mind. It is easy to add new data types by implementing clear, well-documented abstract classes that define functions to preprocess, encode, and decode data.

  Furthermore, new `torch nn.Module` models can be easily added by them to a registry. This encourages reuse and sharing new models with the community. Refer to the [Developer Guide](https://ludwig-ai.github.io/ludwig-docs/latest/developer_guide) for further details.

# Advantages

Ludwig is a profound utility for research scientists, data scientists, and machine learning engineers.

- Minimal machine learning boilerplate

  Ludwig takes care of the engineering complexity of deep learning out of the box, enabling research scientists to focus on building models at the highest level of abstraction.

  Data preprocessing, hyperparameter optimization, device management, and distributed training for newly registered `torch.nn.Module` models come completely free.

- Compare models against existing SOTA models

  Comparing the new model against existing published and implemented models is a simple config change.

- Easily apply new architectures to multiple problems and datasets

  Apply new models across the extensive set of tasks and datasets that Ludwig supports. Ludwig includes a [full benchmarking toolkit](https://arxiv.org/abs/2111.04260) accessible to any user, for running experiments with multiple models across multiple datasets with just a simple configuration.

- Highly configurable data preprocessing, modeling, and metrics

  Any and all aspects of the model architecture, training loop, hyperparameter search, and backend infrastructure can be modified as additional fields in the declarative configuration to customize the pipeline to meet your requirements.

  For details on what can be configured, check out [Ludwig Configuration](https://ludwig-ai.github.io/ludwig-docs/latest/configuration/) docs.

- Low-code interface for state-of-the-art models, including pre-trained Huggingface Transformers

  Ludwig also natively integrates with pre-trained models, such as the ones available in [Huggingface Transformers](https://huggingface.co/docs/transformers/index). Users can choose from a vast collection of state-of-the-art pre-trained PyTorch models to use without needing to write any code at all. For example, training a BERT-based sentiment analysis model with Ludwig is as simple as:

  ```
  ludwig train --dataset sst5 -–config_str “{input_features: [{name: sentence, type: text, encoder: bert}], output_features: [{name: label, type: category}]}”
  ```

- Low-code Interface for AutoML

  [Ludwig AutoML](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/automl/) allows users to obtain trained models by providing just a dataset, the target column, and a time budget. It’s as simple as that!

  ```
  auto_train_results = ludwig.automl.auto_train(
      dataset=my_dataset_df, target=target_column_name, time_limit_s=7200, tune_for_memory=False
  )
  ```

- Easy Productionisation

  Ludwig makes it easy to serve deep learning models, including on GPUs. Launch a REST API for your trained Ludwig model.

  ```
  ludwig serve --model_path=/path/to/model
  ```

  Ludwig supports exporting models to efficient Torschscript bundles.

  ```
  ludwig export_torchscript -–model_path=/path/to/model
  ```

# Examples

- [Text Classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/text_classification)
- [Named Entity Recognition Tagging](https://ludwig-ai.github.io/ludwig-docs/latest/examples/ner_tagging)
- [Natural Language Understanding](https://ludwig-ai.github.io/ludwig-docs/latest/examples/nlu)
- [Machine Translation](https://ludwig-ai.github.io/ludwig-docs/latest/examples/machine_translation)
- [Chit-Chat Dialogue Modeling through seq2seq](https://ludwig-ai.github.io/ludwig-docs/latest/examples/seq2seq)
- [Sentiment Analysis](https://ludwig-ai.github.io/ludwig-docs/latest/examples/sentiment_analysis)
- [Image Classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/image_classification)
- [Image Classification (MNIST)](https://ludwig-ai.github.io/ludwig-docs/latest/examples/mnist)
- [Visual Question Answering](https://ludwig-ai.github.io/ludwig-docs/latest/examples/visual_qa)
- [One-shot Learning with Siamese Networks](https://ludwig-ai.github.io/ludwig-docs/latest/examples/oneshot)
- [Speaker Verification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/speaker_verification)
- [Spoken Digit Speech Recognition](https://ludwig-ai.github.io/ludwig-docs/latest/examples/speech_recognition)
- [Binary Classification (titanic)](https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic)
- [Timeseries forecasting](https://ludwig-ai.github.io/ludwig-docs/latest/examples/forecasting)
- [Timeseries forecasting (Weather)](https://ludwig-ai.github.io/ludwig-docs/latest/examples/weather)
- [Movie rating prediction](https://ludwig-ai.github.io/ludwig-docs/latest/examples/movie_ratings)
- [Multi-label classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/multi_label)
- [Multi-Task Learning](https://ludwig-ai.github.io/ludwig-docs/latest/examples/multi_task)
- [Simple Regression: Fuel Efficiency Prediction](https://ludwig-ai.github.io/ludwig-docs/latest/examples/fuel_efficiency)
- [Fraud Detection](https://ludwig-ai.github.io/ludwig-docs/latest/examples/fraud)

# More Information

[Full official documentation](https://ludwig-ai.github.io/ludwig-docs/)

Read our publications on [Ludwig](https://arxiv.org/pdf/1909.07930.pdf), [declarative ML](https://arxiv.org/pdf/2107.08148.pdf), and [Ludwig’s SoTA benchmarks](https://openreview.net/pdf?id=hwjnu6qW7E4).

Learn more about [how Ludwig works](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/how_ludwig_works/), [how to get started](https://ludwig-ai.github.io/ludwig-docs/latest/getting_started/), and work through more [examples](https://ludwig-ai.github.io/ludwig-docs/latest/examples).

If you are interested in contributing, have questions, comments, or thoughts to share, or if you just want to be in the
know, please consider [joining the Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ) and follow us on [Twitter](https://twitter.com/ludwig_ai)!

# Getting Involved

- [Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)
- [Twitter](https://twitter.com/ludwig_ai)
- [Medium](https://medium.com/ludwig-ai)
- [GitHub Issues](https://github.com/ludwig-ai/ludwig/issues)
