<p align="center">
  <a href="https://ludwig.ai">
    <img src="https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/ludwig_hero_smaller.jpg" height="150">
  </a>
</p>

<div align="center">

_Declarative deep learning framework built for scale and efficiency._

[![PyPI version](https://badge.fury.io/py/ludwig.svg)](https://badge.fury.io/py/ludwig)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)
[![DockerHub](https://img.shields.io/docker/pulls/ludwigai/ludwig.svg)](https://hub.docker.com/r/ludwigai)
[![Downloads](https://pepy.tech/badge/ludwig)](https://pepy.tech/project/ludwig)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/ludwig-ai/ludwig/blob/master/LICENSE)
[![Twitter](https://img.shields.io/twitter/follow/ludwig_ai.svg?style=social&logo=twitter)](https://twitter.com/ludwig_ai)

</div>

# 📖 What is Ludwig?

Ludwig is a **low-code** framework for building **custom** AI models like **LLMs** and other deep neural networks.

Key features:

- 🛠 **Build custom models with ease:** a declarative YAML configuration file is all you need to train a state-of-the-art LLM on your data. Support for multi-task and multi-modality learning. Comprehensive config validation detects invalid parameter combinations and prevents runtime failures.
- ⚡ **Optimized for scale and efficiency:** automatic batch size selection, distributed training ([DDP](https://pytorch.org/tutorials/beginner/ddp_series_theory.html), [DeepSpeed](https://github.com/microsoft/DeepSpeed)), parameter efficient fine-tuning ([PEFT](https://github.com/huggingface/peft)), 4-bit quantization (QLoRA), and larger-than-memory datasets.
- 📐 **Expert level control:** retain full control of your models down to the activation functions. Support for hyperparameter optimization, explainability, and rich metric visualizations.
- 🧱 **Modular and extensible:** experiment with different model architectures, tasks, features, and modalities with just a few parameter changes in the config. Think building blocks for deep learning.
- 🚢 **Engineered for production:** prebuilt [Docker](https://hub.docker.com/u/ludwigai) containers, native support for running with [Ray](https://www.ray.io/) on [Kubernetes](https://github.com/ray-project/kuberay), export models to [Torchscript](https://pytorch.org/docs/stable/jit.html) and [Triton](https://developer.nvidia.com/triton-inference-server), upload to [HuggingFace](https://huggingface.co/models) with one command.

Ludwig is hosted by the
[Linux Foundation AI & Data](https://lfaidata.foundation/).

![img](https://raw.githubusercontent.com/ludwig-ai/ludwig-docs/master/docs/images/ludwig_legos_unanimated.gif)

# 💾 Installation

Install from PyPi. Be aware that Ludwig requires Python 3.8+.

```shell
pip install ludwig
```

Or install with all optional dependencies:

```shell
pip install ludwig[full]
```

# 🚂 Getting Started

Want to take a quick peak at some of the Ludwig 0.8 features? Check out this Colab Notebook 🚀 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lB4ALmEyvcMycE3Mlnsd7I3bc0zxvk39)

For a full tutorial, check out the official [getting started guide](https://ludwig-ai.github.io/ludwig-docs/latest/getting_started/),
or take a look at end-to-end [Examples](https://ludwig-ai.github.io/ludwig-docs/latest/examples).

## Large Language Model Fine-Tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c3AO8l_H6V_x37RwQ8V7M6A-RmcBf2tG?usp=sharing)

Let's fine-tune a pretrained LLaMA-2-7b large language model to follow instructions like a chatbot ("instruction tuning").

### Prerequisites

- [HuggingFace API Token](https://huggingface.co/docs/hub/security-tokens)
- Access approval to [Llama2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- GPU with at least 12 GiB of VRAM (in our tests, we used an Nvidia T4)

### Running

We'll use the [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) dataset, which will be formatted as a table-like file that looks like this:

|                    instruction                    |      input       |                      output                       |
| :-----------------------------------------------: | :--------------: | :-----------------------------------------------: |
|       Give three tips for staying healthy.        |                  | 1.Eat a balanced diet and make sure to include... |
| Arrange the items given below in the order to ... | cake, me, eating |                  I eating cake.                   |
| Write an introductory paragraph about a famous... |  Michelle Obama  | Michelle Obama is an inspirational woman who r... |
|                        ...                        |       ...        |                        ...                        |

Create a YAML config file named `model.yaml` with the following:

```yaml
model_type: llm
base_model: meta-llama/Llama-2-7b-hf

quantization:
  bits: 4

adapter:
  type: lora

prompt:
  template: |
    Below is an instruction that describes a task, paired with an input that may provide further context.
    Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:

input_features:
  - name: prompt
    type: text

output_features:
  - name: output
    type: text

trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    decay: cosine
    warmup_fraction: 0.01

preprocessing:
  sample_ratio: 0.1

backend:
  type: local
```

And now let's train the model:

```bash
export HUGGING_FACE_HUB_TOKEN = "<api_token>"

ludwig train --config model.yaml --dataset "ludwig://alpaca"
```

## Supervised ML

Let's build a neural network that predicts whether a given movie critic's review on [Rotten Tomatoes](https://www.kaggle.com/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset) was positive or negative.

Our dataset will be a CSV file that looks like this:

|     movie_title      | content_rating |              genres              | runtime | top_critic | review_content                                                                                                                                                                                                   | recommended |
| :------------------: | :------------: | :------------------------------: | :-----: | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Deliver Us from Evil |       R        |    Action & Adventure, Horror    |  117.0  | TRUE       | Director Scott Derrickson and his co-writer, Paul Harris Boardman, deliver a routine procedural with unremarkable frights.                                                                                       | 0           |
|       Barbara        |     PG-13      | Art House & International, Drama |  105.0  | FALSE      | Somehow, in this stirring narrative, Barbara manages to keep hold of her principles, and her humanity and courage, and battles to save a dissident teenage girl whose life the Communists are trying to destroy. | 1           |
|   Horrible Bosses    |       R        |              Comedy              |  98.0   | FALSE      | These bosses cannot justify either murder or lasting comic memories, fatally compromising a farce that could have been great but ends up merely mediocre.                                                        | 0           |
|         ...          |      ...       |               ...                |   ...   | ...        | ...                                                                                                                                                                                                              | ...         |

Download a sample of the dataset from [here](https://ludwig.ai/latest/data/rotten_tomatoes.csv).

```bash
wget https://ludwig.ai/latest/data/rotten_tomatoes.csv
```

Next create a YAML config file named `model.yaml` with the following:

```yaml
input_features:
  - name: genres
    type: set
    preprocessing:
      tokenizer: comma
  - name: content_rating
    type: category
  - name: top_critic
    type: binary
  - name: runtime
    type: number
  - name: review_content
    type: text
    encoder:
      type: embed
output_features:
  - name: recommended
    type: binary
```

That's it! Now let's train the model:

```bash
ludwig train --config model.yaml --dataset rotten_tomatoes.csv
```

**Happy modeling**

Try applying Ludwig to your data. [Reach out](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)
if you have any questions.

# ❓ Why you should use Ludwig

- **Minimal machine learning boilerplate**

  Ludwig takes care of the engineering complexity of machine learning out of
  the box, enabling research scientists to focus on building models at the
  highest level of abstraction. Data preprocessing, hyperparameter
  optimization, device management, and distributed training for
  `torch.nn.Module` models come completely free.

- **Easily build your benchmarks**

  Creating a state-of-the-art baseline and comparing it with a new model is a
  simple config change.

- **Easily apply new architectures to multiple problems and datasets**

  Apply new models across the extensive set of tasks and datasets that Ludwig
  supports. Ludwig includes a
  [full benchmarking toolkit](https://arxiv.org/abs/2111.04260) accessible to
  any user, for running experiments with multiple models across multiple
  datasets with just a simple configuration.

- **Highly configurable data preprocessing, modeling, and metrics**

  Any and all aspects of the model architecture, training loop, hyperparameter
  search, and backend infrastructure can be modified as additional fields in
  the declarative configuration to customize the pipeline to meet your
  requirements. For details on what can be configured, check out
  [Ludwig Configuration](https://ludwig-ai.github.io/ludwig-docs/latest/configuration/)
  docs.

- **Multi-modal, multi-task learning out-of-the-box**

  Mix and match tabular data, text, images, and even audio into complex model
  configurations without writing code.

- **Rich model exporting and tracking**

  Automatically track all trials and metrics with tools like Tensorboard,
  Comet ML, Weights & Biases, MLFlow, and Aim Stack.

- **Automatically scale training to multi-GPU, multi-node clusters**

  Go from training on your local machine to the cloud without code changes.

- **Low-code interface for state-of-the-art models, including pre-trained Huggingface Transformers**

  Ludwig also natively integrates with pre-trained models, such as the ones
  available in [Huggingface Transformers](https://huggingface.co/docs/transformers/index).
  Users can choose from a vast collection of state-of-the-art pre-trained
  PyTorch models to use without needing to write any code at all. For example,
  training a BERT-based sentiment analysis model with Ludwig is as simple as:

  ```shell
  ludwig train --dataset sst5 --config_str "{input_features: [{name: sentence, type: text, encoder: bert}], output_features: [{name: label, type: category}]}"
  ```

- **Low-code interface for AutoML**

  [Ludwig AutoML](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/automl/)
  allows users to obtain trained models by providing just a dataset, the
  target column, and a time budget.

  ```python
  auto_train_results = ludwig.automl.auto_train(dataset=my_dataset_df, target=target_column_name, time_limit_s=7200)
  ```

- **Easy productionisation**

  Ludwig makes it easy to serve deep learning models, including on GPUs.
  Launch a REST API for your trained Ludwig model.

  ```shell
  ludwig serve --model_path=/path/to/model
  ```

  Ludwig supports exporting models to efficient Torchscript bundles.

  ```shell
  ludwig export_torchscript -–model_path=/path/to/model
  ```

# 📚 Tutorials

- [Text Classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/text_classification)
- [Tabular Data Classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/adult_census_income)
- [Image Classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/mnist)
- [Multimodal Classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/multimodal_classification)

# 🔬 Example Use Cases

- [Named Entity Recognition Tagging](https://ludwig-ai.github.io/ludwig-docs/latest/examples/ner_tagging)
- [Natural Language Understanding](https://ludwig-ai.github.io/ludwig-docs/latest/examples/nlu)
- [Machine Translation](https://ludwig-ai.github.io/ludwig-docs/latest/examples/machine_translation)
- [Chit-Chat Dialogue Modeling through seq2seq](https://ludwig-ai.github.io/ludwig-docs/latest/examples/seq2seq)
- [Sentiment Analysis](https://ludwig-ai.github.io/ludwig-docs/latest/examples/sentiment_analysis)
- [One-shot Learning with Siamese Networks](https://ludwig-ai.github.io/ludwig-docs/latest/examples/oneshot)
- [Visual Question Answering](https://ludwig-ai.github.io/ludwig-docs/latest/examples/visual_qa)
- [Spoken Digit Speech Recognition](https://ludwig-ai.github.io/ludwig-docs/latest/examples/speech_recognition)
- [Speaker Verification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/speaker_verification)
- [Binary Classification (Titanic)](https://ludwig-ai.github.io/ludwig-docs/latest/examples/titanic)
- [Timeseries forecasting](https://ludwig-ai.github.io/ludwig-docs/latest/examples/forecasting)
- [Timeseries forecasting (Weather)](https://ludwig-ai.github.io/ludwig-docs/latest/examples/weather)
- [Movie rating prediction](https://ludwig-ai.github.io/ludwig-docs/latest/examples/movie_ratings)
- [Multi-label classification](https://ludwig-ai.github.io/ludwig-docs/latest/examples/multi_label)
- [Multi-Task Learning](https://ludwig-ai.github.io/ludwig-docs/latest/examples/multi_task)
- [Simple Regression: Fuel Efficiency Prediction](https://ludwig-ai.github.io/ludwig-docs/latest/examples/fuel_efficiency)
- [Fraud Detection](https://ludwig-ai.github.io/ludwig-docs/latest/examples/fraud)

# 💡 More Information

Read our publications on [Ludwig](https://arxiv.org/pdf/1909.07930.pdf), [declarative ML](https://arxiv.org/pdf/2107.08148.pdf), and [Ludwig’s SoTA benchmarks](https://openreview.net/pdf?id=hwjnu6qW7E4).

Learn more about [how Ludwig works](https://ludwig-ai.github.io/ludwig-docs/latest/user_guide/how_ludwig_works/), [how to get started](https://ludwig-ai.github.io/ludwig-docs/latest/getting_started/), and work through more [examples](https://ludwig-ai.github.io/ludwig-docs/latest/examples).

If you are interested in contributing, have questions, comments, or thoughts to share, or if you just want to be in the
know, please consider [joining the Ludwig Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ) and follow us on [Twitter](https://twitter.com/ludwig_ai)!

# 🤝 Join the community to build Ludwig with us

Ludwig is an actively managed open-source project that relies on contributions from folks just like
you. Consider joining the active group of Ludwig contributors to make Ludwig an even
more accessible and feature rich framework for everyone to use!

<a href="https://github.com/ludwig-ai/ludwig/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ludwig-ai/ludwig" />
</a><br/>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ludwig-ai/ludwig&type=Date)](https://star-history.com/#ludwig-ai/ludwig&Date)

# 👋 Getting Involved

- [Slack](https://join.slack.com/t/ludwig-ai/shared_invite/zt-mrxo87w6-DlX5~73T2B4v_g6jj0pJcQ)
- [Twitter](https://twitter.com/ludwig_ai)
- [Medium](https://medium.com/ludwig-ai)
- [GitHub Issues](https://github.com/ludwig-ai/ludwig/issues)
