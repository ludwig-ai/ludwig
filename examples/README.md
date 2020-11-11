# Examples
This directory contains example programs demonstrating Ludwig's Python APIs.

|Directory|Examples Provided|
|---------|-----------------|
|hyperopt|Demonstrates Ludwig's to hyper-parameter optimization capability.|
|kfold_cv|Provides two examples for performing a k-fold cross validation analysis.  One example uses the `ludwig experiment` cli.  The other example uses the `ludwig.experiment.kfold_cross_validate()` api function.|
|mnist|Creates a model config data structure from a yaml file and trains a model.  Programmatically modify the model config data structure to evaluate several different neural network architectures.  Jupyter notebook demonstrates using a hold-out test data set to visualize model performance for alternative model architectures.|
|titanic|Trains a simple model with model config contained in a yaml file.  Trains multiple models from yaml files and generate visualizations to compare training results.  Jupyter notebook demonstrating how to programmatically create visualizations.|
|serve|Demonstrates running Ludwig http model server.  A sample Python program illustrates how to invoke the REST API to get predictions from input features.|


 