# Credit Card Fraud Detection Example

This API example is based on Kaggle's [Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset for detecting instances of credit card fraud.

### Preparatory Steps

Create and download your [Kaggle API Credentials](https://github.com/Kaggle/kaggle-api#api-credentials).

The Credit Card Fraud dataset is hosted by Kaggle, and as such Ludwig will need to authenticate you through the Kaggle API to download the dataset.

### Examples

| File                         | Description                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------ |
| simple_model_training.py     | Demonstrates using Ludwig api for training a model.                            |
| multiple_model_training.py   | Trains two models and generates a visualization for results of training.       |
| model_training_results.ipynb | Example for extracting training statistics and generate custom visualizations. |

Enter `python simple_model_training.py` will train a single model.  Results of model training will be stored in this location.

```
./results/
    simple_experiment_simple_model/
```

Enter `python multiple_model_training.py` will train two models and generate standard Ludwig visualizations comparing the
two models.  Results will in the following directories:

```
./results/
    multiple_model_model1/
    multiple_model_model2/
./visualizations/
    learning_curves_Survived_accuracy.png
    learning_curves_Survived_loss.png
```

This is the standard Ludwig learning curve plot from training the two models
![](../images/learning_curves_Survived_accuracy.png)

This is the custom visualization created by the Jupyter notebook `model_training_results.ipynb`.
![](../images/custom_learning_curve.png)
