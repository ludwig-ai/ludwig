# Native Optuna Hyperparameter Optimization

> **Requires Ludwig 0.15 / PR #4090 (data-pipeline-hyperopt-modernization branch).**

Ludwig 0.15 adds a native Optuna executor that runs HPO trials directly without requiring
Ray Tune. This is the right choice for single-machine HPO: you get AutoSampler, GPSampler
(Bayesian optimization), TPE, CMA-ES, median / Hyperband pruning, SQLite-backed resumable
studies, and the optional Optuna dashboard — without the overhead of a Ray cluster.

If you need distributed trials across many GPUs or nodes, keep using the `ray` executor
(it wraps `OptunaSearch` as its search algorithm). The native executor in this tutorial is
faster, simpler, and single-process.

## Config

```yaml
hyperopt:
  executor:
    type: optuna
    num_samples: 50                # how many trials to run
    sampler: auto                  # auto | gp | tpe | cmaes | random
    pruner: null                   # null | median | hyperband (optional early stopping)
    study_name: ludwig_wine_rmse
    storage: null                  # or sqlite:///wine_hpo.db to persist and resume
    time_budget_s: 1800

  parameters:
    trainer.learning_rate:
      space: loguniform
      lower: 1e-5
      upper: 1e-1
    trainer.batch_size:
      space: int
      lower: 32
      upper: 256
    combiner.num_fc_layers:
      space: int
      lower: 1
      upper: 4
    combiner.output_size:
      space: choice
      categories: [32, 64, 128, 256]

  output_feature: quality
  metric: root_mean_squared_error
  goal: minimize
  split: validation
```

### Sampler options

| `sampler` | Description                                              | Rule of thumb                    |
| --------- | -------------------------------------------------------- | -------------------------------- |
| `auto`    | Optuna AutoSampler (falls back to TPE on older versions) | Default choice                   |
| `gp`      | Gaussian-Process Bayesian optimization                   | Continuous spaces, \<100 trials  |
| `tpe`     | Tree-structured Parzen Estimator                         | Mixed spaces, 50–500 trials      |
| `cmaes`   | Covariance Matrix Adaptation Evolution Strategy          | Purely-continuous, medium budget |
| `random`  | Random search (sanity-check baseline)                    | Sanity check                     |

### Persistence and resuming

Set `storage: sqlite:///wine_hpo.db` to persist trials to disk. Re-running with the same
`study_name` continues the study — failed trials are retried, successful trials are kept.

### Pruning

Set `pruner: median` or `pruner: hyperband` to stop clearly-losing trials early. Requires
the model code to report intermediate values back (Ludwig's Optuna integration reports the
validation metric at each epoch so this works out of the box).

## Running

```bash
pip install 'ludwig[hyperopt]'   # pulls in optuna
python optuna_executor.py
```

Expected output (numbers are illustrative):

```
[Optuna] Best trial:
  value: 0.6184
  params:
    trainer.learning_rate: 0.0032
    trainer.batch_size:    64
    combiner.num_fc_layers: 2
    combiner.output_size:  128
  completed in: 412.8s
```

## Files

| File                 | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `config_optuna.yaml` | Full hyperopt config using the native Optuna executor |
| `optuna_executor.py` | Runs `ludwig.hyperopt` with the above config          |
| `README_optuna.md`   | This file                                             |

## References

- Optuna — Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework",
  KDD 2019. <https://arxiv.org/abs/1907.10902>
- AutoSampler — Optuna v4 AutoSampler documentation.
- Hyperband — Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter
  Optimization", JMLR 2018. <https://arxiv.org/abs/1603.06560>
