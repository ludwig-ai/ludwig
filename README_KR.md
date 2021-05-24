![Ludwig logo](https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/ludwig_hero.png "Ludwig logo")

<div align="center">

[![PyPI version](https://badge.fury.io/py/ludwig.svg)](https://badge.fury.io/py/ludwig)
[![Downloads](https://pepy.tech/badge/ludwig)](https://pepy.tech/project/ludwig)
[![Build Status](https://github.com/ludwig-ai/ludwig/actions/workflows/pytest.yml/badge.svg)](https://github.com/ludwig-ai/ludwig/actions/workflows/pytest.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/ludwig-ai/ludwig/blob/master/LICENSE)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fuber%2Fludwig.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2Fuber%2Fludwig?ref=badge_shield)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/4210/badge)](https://bestpractices.coreinfrastructure.org/projects/4210)

</div>

Ludwig은 사용자들이 코드를 작성할 필요 없이 딥러닝 모델을 학습시키고 사용할 수 있게 하는 TensorFlow 기반으로 만들어진 toolbox입니다.

이 모델을 학습시키기 위해서는 입출력 데이터가 들어있는 파일이 제공되어야 합니다. 나머지는 Ludwig가 알아서 처리해 줄 것입니다. 간단한 명령어들은 단일 기기 혹은 분산 기기를 통해 모델을 학습하는 데에 사용될 수 있고, 또한 새로운 데이터를 예측하는 데에 사용됩니다.

파이썬 프로그램 API 또한 Ludwig에서 사용 가능합니다. 시각화 기능의 모음은 모델 훈련을 분석하고 모델 성능을 test하고 그것들을 비교하는 것을 가능하게 해줍니다.

Ludwig는 확장성을 염두에 두고 설계되었으며, 데이터 타입 추상화에 기반을 두고 있어서 새로운 모델 아키텍처뿐만 아니라 새로운 데이터 타입에 대한 지원을 쉽게 추가할 수 있습니다.

Ludwig은 사용자들이 딥러닝 모델을 빠르게 학습하고 테스트하는 것은 물론, 연구자들이 딥러닝 모델과 비교할 수 있는 강력한 기준과 동일한 데이터 평가를 통해 비교 가능성을 보장하는 테스트 설정을 확보할 수 있도록 활용됩니다.

Ludwig는 특정 use case의 end-to-end 모델을 작성하기 위해 조합할 수 있는 일련의 모델 아키텍처를 제공합니다. 도시를 설계하는 것을 예로 들어, 딥러닝 라이브러리가 도시 내부 건물의 구성요소(기둥, 바닥, 등)를 제공하고 있다면, Ludwig는 도시를 구성하는 건물을 제공하고 있습니다. 그러면 사용자는 도시 내부의 만들어진 건물 중 하나를 선택해 사용하거나, 새로운 건물을 추가할 수 있습니다.

Toolbox에 반영된 핵심 설계 원리는 아래와 같습니다:
- No coding required: 모델을 학습시키고 예측된 데이터를 얻는 데에 코딩 스킬이 필요하지 않습니다.
- Generality: 딥러닝 모델 설계에 대한 새로운 데이터 유형 기반 접근 방식은 다양한 use case들에 적용할 수 있는 tool을 만들어줍니다.
- Flexibility: 숙련된 사용자들은 모델 제작과 훈련을 광범위하게 제어하는 반면, 초보자들은 그것을 쉽게 사용할 수 있습니다.
- Extensibility: 새로운 모델 아키텍처와 새로운 데이터 타입을 쉽게 추가할 수 있습니다.
- Understandability: 종종 딥러닝 모델 내부는 진행 상황을 확인할 수 없는 것처럼 여겨지지만, Ludwig는 성능을 이해하고 예측된 데이터들을 비교하기 위한 표준 시각화 기능을 제공합니다.
- Open Source: Apache License 2.0


Installation
============

Ludwig는 Python 3.6이상 버전을 요구합니다. 만약 Python 3가 설치되어 있지 않으면 다음 명령어를 이용해서 설치하세요.

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

[Horovod](https://github.com/horovod/horovod)를 통해 분산 학습이 지원되며, `pip install ludwig[horovod]` 또는 `HOROVOD_GPU_OPERATIONS=NCCL pip install ludwig[horovod]` 와 같이 GPU 환경에서 설치가 가능합니다.
설치 가능한 옵션들을 더 확인하고 싶으시다면 Horovod's [installation guide](https://horovod.readthedocs.io/en/stable/install_include.html) 를 참고하시기 바랍니다.

추가하려는 package들은  `pip install ludwig[extra1,extra2,...]` 의 명령어를 통해 설치가 가능합니다. 예를 들어, `pip install ludwig[text,viz]` 와 같은 조합으로 설치가 가능합니다. 모든 파일들을 한 번에 설치하려면 `pip install ludwig[full]`을 사용하면 됩니다.

소스코드를 repository에서 build하려는 개발자들은 아래와 같은 방법을 사용하면 됩니다.

```
git clone git@github.com:ludwig-ai/ludwig.git
cd ludwig
virtualenv -p python3 venv
source venv/bin/activate
pip install -e '.[test]'
```

**Note:** 만약 GPU 없이 실행 중이라면, 가벼운 용량의 CPU로 만 사용할 수 있는 TensorFlow를 사용하고 싶으실 겁니다. CPU로 만 사용할 수 있는 TensorFlow 버전을 사용하고 싶다면 `tensorflow`를 삭제하고 `ludwig`를 설치한 후 `tensorflow-cpu`로 대체하면 됩니다. `requirements.txt`에 명시되어 있는 대로 호환 가능한 범위 내의 버전을 설치해야만 합니다.


Basic Principles
----------------

Ludwig는 모델학습, 학습된 모델을 이용한 예측, 평가의 3가지 주요 기능을 제공합니다. 이것은 데이터 유형 추상화에 기반합니다. 그래서 같은 데이터를 이용해 사전, 사후 처리 과정을 데이터 유형을 공유하는 서로 다른 dataset으로 실행되고, 개발된 encoding 및 decoding 모델을 다른 여러 작업에서 재사용이 가능합니다.

Ludwig로 모델을 학습시키는 것은 굉장히 간단합니다. 단지 dataset file과 yaml file만 제공해 주면 됩니다.

config파일에는, 입출력 값의 속성을 포함합니다. 당신이 해야 할 것은 dataset파일에서 열에 해당하는 데이터들의 이름만 정의해 주면 됩니다. 여기에 필요한 것은 모델에 대한 입력 데이터 유형, 그리고 모델이 예측하는 대상 변수인 출력 dataset파일에서의 열 이름입니다. Ludwig는 그에 따라 딥러닝 모델을 만들어 당신을 위해 학습할 것입니다.

현재, Ludwig에서 사용 가능한 데이터 유형은 아래와 같습니다.

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

서로 다른 입력과 출력의 데이터 유형을 사용하는 경우, 사용자들은 다양한 작업을 진행할 수 있습니다. 아래는 그 예시입니다.

- text input + category output = text classifier
- image input + category output = image classifier
- image input + text output = image captioning
- audio input + binary output = speaker verification
- text input + sequence output = named entity recognition / summarization
- category, numerical and binary inputs + numerical output = regression
- timeseries input + numerical output = forecasting model
- category, numerical and binary inputs + binary output = fraud detection

[예시](https://ludwig-ai.github.io/ludwig-docs/examples/)를 참고하여 Ludwig을 통해 어떤 작업을 실행하는지 확인하세요.

Config 파일은 인코더와 디코더가 사용할 각 열에 저장된 데이터를 사전 처리하는 방법, 최적화할 아키텍처 및 학습 매개변수, 하이퍼 파라미터 등의 추가 정보를 저장합니다. 이를 통해 초보자는 쉽게 사용할 수 있고, 전문가도 유연하게 사용할 수 있습니다.


Training
--------

예를 들어, 아래와 같이 분류된 dataset형식의 파일을 보면:

| doc_text                              | class    |
|---------------------------------------|----------|
| Former president Barack Obama ...     | politics |
| Juventus hired Cristiano Ronaldo ...  | sport    |
| LeBron James joins the Lakers ...     | sport    |
| ...                                   | ...      |

`doc_text`열의 내용을 입력으로 사용하여 `class`열의 값을 예측하는 모델을 학습시키려고 할 때 다음과 같은 config파일 구성을 사용할 수 있습니다:

```yaml
{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}
```

그리고 사용자의 콘솔 창에서 다음의 명령을 입력하여 학습을 시작합니다:

```
ludwig train --dataset path/to/file.csv --config "{input_features: [{name: doc_text, type: text}], output_features: [{name: class, type: category}]}"
```

위의 명령어에서 `path/to/file.csv`부분은 위의 표(이외에 많은 데이터 타입이 지원됩니다)에서 UTF-8로 인코딩 되어 있는 dataset파일을 포함하는 경로입니다.

Ludwig은 다음과 같은 동작을 합니다:

1. data의 무작위 분할을 실시합니다
2. dataset을 사전 처리합니다.
3. Softmax classifier를 통해 결과를 해석하는 ParallelCNN모델(text 기능의 기본값)을 구축합니다.
4. 검증 세트의 성능이 더 이상 개선되지 않을 때까지 학습을 반복합니다.

학습 과정이 콘솔창에서 보일 것이고 TensorBoard 또한 사용될 수 있습니다.

만약 RNN encoder를 사용하거나 epoch의 숫자를 더 키워 학습시키는 것을 더 선호한다면 아래와 같은 형식의 config파일 형식을 사용하면 됩니다:

```yaml
{input_features: [{name: doc_text, type: text, encoder: rnn}], output_features: [{name: class, type: category}], training: {epochs: 50}}
```

사용자가 config파일에서 사용 가능한 명령어들을 확인하고 싶으시다면 [User Guide](https://ludwig-ai.github.io/ludwig-docs/user_guide/)를 참고하고, [Examples](https://ludwig-ai.github.io/ludwig-docs/examples/)을 통해 여러 가지 다른 작업에 Ludwig을 사용하는 방법을 확인하세요.

학습 후, Ludwig는 학습된 모델과 hyperparameter, 학습 과정의 통계 요약이 포함된 `results`폴더를 생성할 것입니다.
사용자들은 시각화 방법들 중 하나인 도구를 사용하여 시각화를 할 수 있습니다. 예를 들어:

```
ludwig visualize --visualization learning_curves --training_statistics path/to/training_statistics.json
```

위의 명령어는 아래와 같이 그래프를 나타낼 것이고 학습 과정에 있어서의 손실과 정확도를 확인할 수 있습니다.

![Learning Curves](https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/getting_started_learning_curves.png "Learning Curves")

시각화하는 더 다양한 방법을 알고 싶으시다면 [Visualizations](https://ludwig-ai.github.io/ludwig-docs/user_guide/#visualizations)에서 확인해 주시기 바랍니다.


Distributed Training
--------------------

사용자는 [Horovod](https://github.com/horovod/horovod)를 통해 사용자가 훈련시킨 모델을 배포할 수 있고 여러 GPU가 있는 단일 기계 및 여러 GPU가 있는 다중 기계를 통해 학습하는 것을 허용합니다. 더 자세한 정보를 알고 싶으시다면 [User Guide](https://ludwig-ai.github.io/ludwig-docs/user_guide/#distributed-training)를 확인해 주시기 바랍니다. 


Prediction and Evaluation
-------------------------

이전에 학습시킨 모델로 새로운 data의 출력값을 예측하고 싶다면 콘솔 창에서 다음의 명령어를 입력하면 됩니다:

```
ludwig predict --dataset path/to/data.csv --model_path /path/to/model
```

이 명령어를 실행하면 모델이 예측 값을 반환합니다.

dataset에 출력의 진리 값이 포함된 경우 모델에서 얻은 예측 값과 비교하여 모델 성능을 평가할 수 있습니다.

```
ludwig evaluate --dataset path/to/data.csv --model_path /path/to/model
```

위 명령어는 `visualize` tool에 의해 시각화되고 다른 모델들 간의 성능과 예측을 비교하는데 사용되는 평가 성능 통계를 만들어 냅니다. 예를 들어:

```
ludwig visualize --visualization compare_performance --test_statistics path/to/test_statistics_model_1.json path/to/test_statistics_model_2.json
```

여러 측정 기준에 대한 모델들을 비교하는 막대그래프를 반환합니다:

![Performance Comparison](https://github.com/ludwig-ai/ludwig-docs/raw/master/docs/images/compare_performance.png "Performance Comparison")

학습과 예측을 교대로 수행하는 간단한 `ludwig experiment`명령어 또한 사용 가능합니다.


Programmatic API
----------------

Ludwig는 사용자가 모델을 학습시키거나 불러오게 해주고 새로운 데이터에 대한 예측 값을 얻는 데에 사용하는 간단한 프로그램 API를 제공합니다:

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

YAML 파일에 대한 같은 정보를 포함하고 있는 `config`는 CLI(Command Line Interface)에 제공됩니다. 더 자세한 정보는 [User Guide](https://ludwig-ai.github.io/ludwig-docs/user_guide/)과 [API documentation](https://ludwig-ai.github.io/ludwig-docs/api/)에서 제공됩니다.


Extensibility
-------------

Ludwig는 처음부터 확장성을 염두에 두고 제작되었습니다. 
데이터를 사전 처리, 부호화 및 복호화 기능을 포함한 추상 클래스의 데이터 유형별 구현을 추가하면 현재 지원되지 않는 데이터형을 쉽게 추가할 수 있습니다.

나아가 자체적인 특정 hyperparameters가 있는 새로운 모델들은 (데이터 타입에 따라, 특정 등급의) tensor들을 입력으로 받아들이고 tensor들을 출력으로 제공하는 클래스를 구현함으로써 쉽게 추가할 수 있습니다.
이것은 모델의 재사용과 커뮤니티와의 공유를 장려합니다.
자세한 내용은 [Developer Guide](https://ludwig-ai.github.io/ludwig-docs/developer_guide/)를 참조하십시오.


Full documentation
------------------

전체 문서는 [여기](https://ludwig-ai.github.io/ludwig-docs)에서 확인할 수 있습니다.


License
-------

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fuber%2Fludwig.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fuber%2Fludwig?ref=badge_large)
