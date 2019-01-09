<span style="float:right;">[[source]](https://github.com/uber/ludwig/blob/master/ludwig.py#L48)</span>
# LudwigModel class

```python
ludwig.LudwigModel(
  model_definition,
  model_definition_file=None
)
```


---
# LudwigModel methods

## load


```python
load(
  metadata_json
)
```

---
## predict


```python
predict(
  data_df=None,
  data_csv=None,
  data_dict=None,
  batch_size=128,
  gpus=None,
  gpu_fraction=1,
  debug=False
)
```

---
## test


```python
test(
  data_df=None,
  data_csv=None,
  data_dict=None,
  batch_size=128,
  gpus=None,
  gpu_fraction=1,
  debug=False
)
```

---
## train


```python
train(
  data_df=None,
  data_train_df=None,
  data_validation_df=None,
  data_test_df=None,
  data_csv=None,
  data_train_csv=None,
  data_validation_csv=None,
  data_test_csv=None,
  data_hdf5=None,
  data_train_hdf5=None,
  data_validation_hdf5=None,
  data_test_hdf5=None,
  metadata_json=None,
  model_name='run',
  model_load_path=None,
  model_resume_path=None,
  skip_save_progress_weights=False,
  dataset_type='generic',
  skip_save_processed_input=False,
  output_directory='results',
  gpus=None,
  gpu_fraction=1.0,
  random_seed=42,
  debug=False
)
```
