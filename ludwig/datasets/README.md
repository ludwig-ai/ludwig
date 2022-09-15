## Ludwig Datasets API

The Ludwig Dataset Zoo provides datasets that can be directly plugged into a Ludwig model.

The simplest way to use a dataset is to import it:

```python
from ludwig.datasets import titanic

# Loads into single dataframe with a 'split' column:
dataset_df = titanic.load()

# Loads into split dataframes:
train_df, test_df, _ = titanic.load(split=True)
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
train_df, test_df, _ = titanic.load(split=True)
```

Some datasets are hosted on [Kaggle](https://www.kaggle.com) and require a kaggle account. To use these, you'll need to
[set up Kaggle credentials](https://www.kaggle.com/docs/api) in your environment. If the dataset is part of a Kaggle
competition, you'll need to accept the terms on the competition page.

To check programmatically, datasets have an `.is_kaggle_dataset` property.

## Downloading, Processing, and Exporting

Datasets are first downloaded into `LUDWIG_CACHE`, which may be set as an environment variable and defaults to
`$HOME/.ludwig_cache`.

Datasets are automatically loaded, processed, and re-saved as parquet files.  The processed dataset is saved in
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
