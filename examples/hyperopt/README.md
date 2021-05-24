# Hyperparameter Optimization

Demonstrates hyperparameter optimization using Ludwig's in-built capabilities.

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
Jupyter notebook `model_hyperopt_example.ipynb` demonstrates several hyperparameter optimization capabilities. Key features demonstrated in the notebook:
* Training data is prepared for use
* Programmatically create Ludwig config dictionary from the training data dataframe
* Setup parameter space for hyperparameter optimization
* Perform two hyperparameter runs
  * Parallel workers using random search strategy
  * Serial processing using random search strategy
  * Parallel workers using grid search strategy (Note: takes about 35 minutes)
* Demonstrate various Ludwig visualizations for hyperparameter optimization
