
Ludwig is a toolbox built on top of TensorFlow that allows to train and test deep learning models without the need to write code.

All you need to provide is a CSV file containing your data, a list of columns to use as inputs, and a list of columns to use as outputs, Ludwig will do the rest.
Simple to use commands can be used to train models both locally and in a distributed way, and to use them to predict on new data.

A programmatic API is also available in order to use Ludwig from your python code.
A suite of visualization tools allows to analyze models' training and test performance and to compare them.

Ludwig is built with extensibility principles in mind and is based on data type abstractions, making it easy to add support for new data types as well as new model architectures.

It can be used by practitioners to quickly train and test deep learning models as well as by researchers to obtain strong baselines to compare against and have an experimentation setting that ensures comparability by performing standard data preprocessing and visualization.

Core Features:

- No coding: no coding skills are required to train and use a model.
- Compositional: type-based approach to deep learning model construction.
- Flexible: experienced users have deep control over model building and training, while newcomers will find it easy to apporach.
- Extensible: easy to add new models and new data types.
- Visualizations: provides standard visualizations to understand and compare performances and predictions.
- Open Source: Apache License 2.0.
