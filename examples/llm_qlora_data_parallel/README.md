# Data-Parallel QLoRA Fine-Tuning

If you have a single-node multi-GPU setup with a large dataset that you would like to train using QLoRA, you can use DeepSpeed Stage 0, 1, or 2.

## DeepSpeed Background

As a refresher, here is what each DeepSpeed Zero stage corresponds to:

- **Stage 0**: Disabled, i.e., no partitioning of optimizer state, gradients or model parameters. You can still perform optimizer and parameter offloading, as well training using bf16 or fp16 etc.
- **Stage 1**: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.
- **Stage 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.
- **Stage 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

_NOTE: Data Parallel QLoRA based training only works with DeepSpeed stages \<= 2. This is because DeepSpeed isn't
compatible with partitioning/sharding of quantized weights as of DeepSpeed 0.10.3 when weights are a mixture of dtypes_. See:

- https://github.com/microsoft/DeepSpeed/issues/4295
- https://github.com/microsoft/DeepSpeed/issues/3620

In particular, this comment here summarizes it well:

> Some code for ZeRO3 assumes that all parameters in a model has the same dtype. This model has uint8 and float32 parameters and it throws the error.

## Example Config

The example `train.py` uses DeepSpeed Stage 2 with the Ray backend as follows to fine-tune a model for natural language to code generation task via instruction fine-tuning.

```yaml
backend:
  type: ray
  trainer:
    use_gpu: true
    strategy:
      type: deepspeed
      zero_optimization:
        stage: 2
```

In most cases, stage 2 lets you train large models in distributed fashion across multiple GPUs. However, if you want to use Stage 0 or 1, you can just replace `stage: 2` to the desired zero optimization stage.

## DeepSpeed Zero Stage Benefits

### Benefits of DeepSpeed Stage 0

- **Ease of Use**: Stage 0 is relatively easy to set up and use, making it a good starting point for users looking for memory-efficient training without the complexity of more advanced optimization techniques.
- **Gradient Accumulation**: Stage 0 enables gradient accumulation, which is beneficial for simulating larger batch sizes even on hardware with memory constraints. This can lead to more stable model training and potentially faster convergence.
- **Mixed Precision Training**: It supports mixed-precision training, which utilizes lower-precision data types (e.g., float16) to reduce memory usage while maintaining training stability.

### Benefits of DeepSpeed Stage 1

- **Optimizer State Partitioning**: Stage 1 is primarily focused on partitioning the optimizer state, allowing you to train very large models that wouldn't fit within a single GPU's memory.
- **Memory Efficiency**: It efficiently manages memory by dividing the optimizer state into segments distributed across multiple GPUs. This makes training larger models feasible.
- **Single-GPU Training**: Stage 1 is especially valuable when you need to train large models on a single GPU, making it an essential step before scaling up to more advanced stages for distributed training.
- **Limited Configuration Complexity**: It introduces memory efficiency while maintaining a relatively simple configuration setup compared to the more advanced stages like Stage 2 and Stage 3.

### Benefits of DeepSpeed Stage 2

- **Training Extremely Large Models**: ZeRO Stage 2 partitions both the gradients and the optimizer state to reduce memory requirements significantly. By contrast, Stage 0 and Stage 1 do not have the same level of memory optimization to handle models of such magnitude.
- **Advanced Distributed Training**: ZeRO Stage 2 is designed to handle distributed training at an unprecedented scale. It optimizes communication, gradient aggregation, and synchronization between GPUs and nodes, making it ideal for training large models efficiently in a distributed environment. This advanced distributed training capability is not present in Stage 0 and is more sophisticated than that of Stage 1, which helps in achieving faster training times and handling larger workloads.
