# Ludwig Docker Images

These images provide Ludwig, a toolbox to train and evaluate deep learning models
without the need to write code. Ludwig Docker images contain the full set of pre-requisite
packages to support these capabilities

- text features
- image features
- audio features
- visualizations
- hyperparameter optimization
- distributed training
- model serving

## Repositories

These four repositories contain a version of Ludwig with full features built
from the project's `master` branch.

- `ludwigai/ludwig` Ludwig packaged with PyTorch
- `ludwigai/ludwig-gpu` Ludwig packaged with gpu-enabled version of PyTorch
- `ludwigai/ludwig-ray` Ludwig packaged with PyTorch
  and Ray 2.2.0 (https://github.com/ray-project/ray)
- `ludwigai/ludwig-ray-gpu` Ludwig packaged with gpu-enabled versions of PyTorch
  and Ray 2.2.0 (https://github.com/ray-project/ray)

## Image Tags

- `master` - built from Ludwig's `master` branch
- `nightly` - nightly build of Ludwig's software.
- `sha-<commit point>` - version of Ludwig software at designated git sha1
  7-character commit point.

## Running Containers

Examples of using the `ludwigai/ludwig:master` image to:

- run the `ludwig cli` command or
- run Python program containing Ludwig api or
- view Ludwig results with Tensorboard

For purposes of the examples assume this host directory structure

```
/top/level/directory/path/
    data/
        train.csv
    src/
        config.yaml
        ludwig_api_program.py
```

### Run Ludwig CLI

```
# set shell variable to parent directory
parent_path=/top/level/directory/path

# invoke docker run command to execute the ludwig cli
# map host directory ${parent_path}/data to container /data directory
# map host directory ${parent_path}/src to container /src directory
docker run -v ${parent_path}/data:/data  \
    -v ${parent_path}/src:/src \
    ludwigai/ludwig:master \
    experiment --config /src/config.yaml \
        --dataset /data/train.csv \
        --output_directory /src/results
```

Experiment results can be found in host directory `/top/level/directory/path/src/results`

### Run Python program using Ludwig APIs

```
# set shell variable to parent directory
parent_path=/top/level/directory/path

# invoke docker run command to execute Python interpreter
# map host directory ${parent_path}/data to container /data directory
# map host directory ${parent_path}/src to container /src directory
# set current working directory to container /src directory
# change default entrypoint from ludwig to python
docker run  -v ${parent_path}/data:/data  \
    -v ${parent_path}/src:/src \
    -w /src \
    --entrypoint python \
    ludwigai/ludwig:master /src/ludwig_api_program.py
```

Ludwig results can be found in host
directory `/top/level/directory/path/src/results`

### View Ludwig Tensorboard results

```
# set shell variable to parent directory
parent_path=/top/level/directory/path

# invoke docker run command to execute Tensorboard
# map host directory ${parent_path}/src to container /src directory
# set up mapping from localhost port 6006 to container port 6006
# change default entrypoint from ludwig to tensorboard
# --logdir container location of tenorboard logs /src/results/<experiment_name>_<model_name>/model/logs
# --bind_all Tensorboard serves on all public container interfaces
docker run  -v ${parent_path}/src:/src \
    -p 6006:6006 \
    --entrypoint tensorboard \
    ludwigai/ludwig:master \
      --logdir /src/results/experiment_run/model/logs \
      --bind_all
```

Point browser to `http://localhost:6006` to see Tensorboard dashboard.

### Devcontainer

If you want to contribute to Ludwig, you can setup a Docker container with all the dependencies
installed as a full featured development environment. This can be done using devcontainers with VS Code:
https://code.visualstudio.com/docs/devcontainers/containers

You can find the `devcontainer.json` file within the top level `.devcontainer` folder.
