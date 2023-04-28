# LLM Fine-tuning

These examples show you how to fine-tune Large Language Models by taking advantage of model parallelism
with [DeepSpeed](https://www.deepspeed.ai/), allowing Ludwig to scale to very large models with billions of
parameters.

## Prerequisites

- Installed Ludwig with `ludwig[distributed]` dependencies
- Have a CUDA-enabled version of PyTorch installed
- Have access to a machine or cluster of machines with multiple GPUs