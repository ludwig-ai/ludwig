## Ludwig Datasets API

The Ludwig Dataset Zoo provides datasets that can be directly plugged into a Ludwig model. For each dataset, we've also
included an example Ludwig config which should train reasonably fast on a current-generation laptop.

The simplest way to use a dataset is to import it:

```python
from ludwig.datasets import titanic

# Loads into single dataframe with a 'split' column:
dataset_df = titanic.load()

# Loads into split dataframes:
train_df, test_df, _ = titanic.load(split=True)
```

The `ludwig.datasets` API provides functions to list, describe, and get datasets:

______________________________________________________________________

### list_datasets

Gets a list of the names of available datasets.

**Example:**

```python
dataset_names = ludwig.datasets.list_datasets()
```

______________________________________________________________________

### get_datasets_output_features

If a specific dataset name is passed in, then returns the output features associated with that dataset. Otherwise,
returns an ordered dictionary with dataset names as keys and dictionaries containing the output features for each
dataset as values.

**Example:**

```python
output_features = ludwig.datasets.get_datasets_output_features(dataset="titanic")
```

______________________________________________________________________

### describe_dataset

Gets a human-readable description string for a dataset

**Example:**

```python
print(ludwig.datasets.describe_dataset("titanic"))
```

______________________________________________________________________

### get_dataset

Get a dataset module by name

**Example:**

```python
titanic_dataset = ludwig.datasets.get_dataset("titanic")
```

______________________________________________________________________

### model_configs_for_dataset

Gets a dictionary of model configs for the specified dataset. Keys are the config names, and may
contain the special keys:

- `default` - The default config for the dataset. Should train to decent performance under 10 minutes on a typical
  laptop without GPU.
- `best` - The best known config for the dataset. Should be replaced when a better config is found. This is a good
  opportunity for contributions, if you find a better one please check it in and open a PR!

**Example:**

```python
configs = ludwig.datasets.model_configs_for_dataset("higgs")
default_higgs_config = configs["default"]
best_higgs_config = configs["best"]
```

______________________________________________________________________

## Training a model using builtin dataset and config

This example code trains a model on the Titanic dataset using the default config:

```python
from ludwig.api import LudwigModel
import ludwig.datasets

titanic = ludwig.datasets.get_dataset("titanic")

dataset_df = titanic.load()

titanic_config = titanic.default_model_config

model = LudwigModel(titanic_config)
model.train(dataset_df)
```

Some datasets are hosted on [Kaggle](https://www.kaggle.com) and require a kaggle account. To use these, you'll need to
[set up Kaggle credentials](https://www.kaggle.com/docs/api) in your environment. If the dataset is part of a Kaggle
competition, you'll need to accept the terms on the competition page.

To check programmatically, datasets have an `.is_kaggle_dataset` property.

## Downloading, Processing, and Exporting

Datasets are first downloaded into `LUDWIG_CACHE`, which may be set as an environment variable and defaults to
`$HOME/.ludwig_cache`.

Datasets are automatically loaded, processed, and re-saved as parquet files. The processed dataset is saved in
LUDWIG_CACHE.

If the dataset contains media files including images or audio, media files are saved in subdirectories and referenced by
relative paths from the dataset location. To ensure Ludwig can read these files during training, they should be
accessible from Ludwig's working directory.

To export the processed dataset, including any media files it depends on, use the `.export` method:

```python
from ludwig.datasets import twitter_bots

# Exports twitter bots dataset and image files to the current working directory.
twitter_bots.export(".")

# The working directory should now contain:
# ./twitter_bots.parquet        - The twitter bots dataset
# ./profile_images              - Account profile image files
# ./profile_background_images   - Account profile background image files
```
