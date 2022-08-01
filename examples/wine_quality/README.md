# Ludwig Defaults Config Section Example

Demonstrates how to use Ludwig's defaults section introduced in v0.6.

### Preparatory Steps

- Create `data` directory
- Download [Kaggle wine quality data set](https://www.kaggle.com/rajyellow46/wine-quality) into the `data` directory.  Directory should
  appear as follows:

```
wine_quality/
    data/
        winequalityN.csv
```

### Description

Jupyter notebook `model_defaults_example.ipynb` demonstrates how to use the defaults section of Ludwig.
Key features demonstrated in the notebook:

- Training data is prepared for use
- Programmatically create Ludwig config dictionary from the training data dataframe
- How to define preprocessing, encoder, decoder and loss sub-sections under the defaults section
