# Hyperparameter Optimization

Demonstrates hyperparameter optimization using the [hyperopt package](https://github.com/hyperopt/hyperopt).

### Preparatory Steps
* Create `data` directory
* Download [Kaggle wine quality data set](https://www.kaggle.com/rajyellow46/wine-quality) into the `data` directory.  Directory should
appear as follows:
```
hyperopt/
    data/
        winequalityN.csv
```

### Description
Jupyter notebook `model_hyperopt_example.ipynb` demonstrates using the `hyperopt` package to perform hyperparameter optimization during Ludwig model training. Key features demonstrated in the notebook:
* Programmatically building a Ludwig model definition from training data set
* Setup hyperparameter search space for optimization
* Using `hyperopt` `fmin()` function to determine optimal hyperparameters