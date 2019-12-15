# Demonstrates use of Hyperopt with Ludwig

This API example is based on [Ludwig's Kaggle Titanic example](https://uber.github.io/ludwig/examples/#kaggles-titanic-predicting-survivors) for predicting probability of surviving. 

### Preparatory Steps
* Create `data` directory
* Download [Kaggle competition dataset](https://www.kaggle.com/c/titanic/data) into the `data` directory.  Directory should
appear as follows:
```
titanic/
    data/
        train.csv
        test.csv
```

### Examples
|File|Description|
|----|-----------|
|model_hyperopt_example.ipynb|Jupyter notebook illustrates how to use the hyperopt package to perform Ludwig hyperparameter optimiztion| 