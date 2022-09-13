## Ludwig Datasets API

The Ludwig Dataset Zoo provides datasets that can be directly plugged into a Ludwig model.

The simplest way to use a dataset is to import it:

```python
from ludwig.datasets import titanic

# Loads into single dataframe with a 'split' column:
dataset_df = titanic.load()

# Loads into split dataframes:
train_df, _, test_df = titanic.load(split=True)
```

The `ludwig.datasets` API also provides functions to list, describe, and get datasets.  For example:

```python
import ludwig.datasets

# Gets a list of all available dataset names.
dataset_names = ludwig.datasets.list_datasets()

# Prints the description of the titanic dataset.
print(ludwig.datasets.describe_dataset("titanic"))

titanic = ludwig.datasets.get_dataset("titanic")

# Loads into single dataframe with a 'split' column:
dataset_df = titanic.load()

# Loads into split dataframes:
train_df, _, test_df = titanic.load(split=True)
```

Some datasets are hosted on [Kaggle](https://www.kaggle.com) and require a kaggle account. To use these, you'll need to
[set up Kaggle credentials](https://www.kaggle.com/docs/api) in your environment. If the dataset is part of a Kaggle
competition, you'll need to accept the terms on the competition page.

To check programmatically, datasets have an `.is_kaggle_dataset` property.
