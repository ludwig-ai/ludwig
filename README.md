Ludwig
======

Introduction
------------

Ludwig is a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code.

All you need to provide is a CSV file containing your data, a list of columns to use as inputs, and a list of columns to use as outputs, Ludwig will do the rest.
Simple to use script can be used to train models both locally and in a distributed way, and to use them to predict on new data.

A programmatic API is also available in order to use Ludwig from your python code.
A suite of visualization tools allows to analyze models' training and test performance and to compare them.

Ludwig is built with extensibility principles in mind and is based on data type abstractions, making it easy to add support for new data types as well as new model architectures.

It can be used by practitioners to quickly implement deep learning models as well as by researchers to obtain strong baselines to compare against and have an experimentation setting that ensures comparability by performing standard data preprocessing and visualization.

Core Features:
- No coding: no coding skills are required to train and use a model.
- Compositional: type-based approach to deep learning model construction.
- Flexible: experienced users have deep control over model building and training.
- Extensible: easy to add new models and new feature types.
- Visualizations: provides standard visualizations to understand and compare performances and predictions.
- Open Source: Apache License 2.0.

Installation
------------

Ludwig's requirements are the following:
- numpy
- pandas
- scipy
- scikit-learn
- imageio
- spacy
- tensorflow
- matplotlib
- seaborn
- Cython
- h5py
- tqdm
- tabulate
- PyYAML

Ludwig has been developed and tested with python 3 in mind.
If you don’t have python 3 installed, install it
```
sudo apt install python3  # on ubuntu
brew install python3  # on mac
```
At the time of writing TensorFlow is not compatible with python 3.7, so the suggested version of python for Ludwig is 3.6

In order to install Ludwig just run:
```
pip install ludwig
python -m spacy download en
```
or install it after cloning the repository:
```
git clone git@github.com:uber/ludwig.git
cd ludwig
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en
python setup.py install
```

Beware that in the `requirements.txt` file the `tensorflow` package is the regular one, not the GPU enabled one.
To install the GPU enabled one replace it with `tensorflow-gpu`.

If you want to train Ludwig models in a distributed way, you need to also install the `horovod` package.
Please follow the instructions on [Horovod's repository](https://github.com/uber/horovod) to install it.


Full documentation
------------------

You can find the full documentation [here](http://ludwig.github.io), inside the `docs` directory, with the documentation sources available in `mkdocs/doc/documentation.md`.
