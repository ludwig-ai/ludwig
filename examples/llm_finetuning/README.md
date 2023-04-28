# LLM Fine-tuning

These examples show you how to fine-tune Large Language Models by taking advantage of model parallelism
with [DeepSpeed](https://www.deepspeed.ai/), allowing Ludwig to scale to very large models with billions of
parameters.

## Prerequisites

- Installed Ludwig with `ludwig[distributed]` dependencies
- Have a CUDA-enabled version of PyTorch installed
- Have access to a machine or cluster of machines with multiple GPUs
- The IMDB dataset used in these examples comes from Kaggle, so make sure you have your credentials set (e.g., `$HOME/.kaggle.kaggle.json`)

## Running DeepSpeed on Ray

This is the recommended way to use DeepSpeed, which supports auto-batch size tuning and distributed data processing. 
There is some overhead from using Ray with small datasets (<100MB), but in most cases performance should be comparable
to using native DeepSpeed.

From the head node of your Ray cluster:

```bash
./run_train_dsz3_ray.sh
```

### Python API

If you want to run Ludwig programatically (from a notebook or as part of a larger workflow), you can run the following
Python script using the Ray cluster launcher from your local machine.

```bash
ray submit cluster.yaml train_imdb_ray.py
```

If running directly on the Ray head node, you can omit the `ray submit` portion and run like an ordinary Python script:

```bash
python train_imdb_ray.py
```

## Running DeepSpeed Native

This mode is suitable for datasets small enough to fit in memory on a single machine, as it doesn't make use of
distributed data processing (requires use of the Ray backend).

The following example assumes you have 4 GPUs available, but can easily be modified to support your preferred
setup.

From a terminal on your machine:

```bash
./run_train_dsz3.sh
```