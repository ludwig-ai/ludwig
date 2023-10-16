# Data-Parallel QLoRA Fine-Tuning

If you have a single-node multi-GPU setup with a large dataset that you would like to train using QLoRA, you can use DeepSpeed Stage 0, 1, or 2.

As a refresher, here is what each DeepSpeed Zero stage corresponds to:

- **Stage 0**: Disabled, i.e., no partitioning of optimizer state, gradients or model parameters. You can still perform optimizer and parameter offloading, as well training using bf16 or fp16 etc.
- **Stage 1**: The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.
- **Stage 2**: The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.
- **Stage 3**: The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

_NOTE: Data Parallel QLoRA based training only works with DeepSpeed stages \<= 2. This is because DeepSpeed isn't
compatible with partitioning/sharding of quantized weights as of DeepSpeed 0.10.3_.

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

If you want to use Stage 0 or 1, you can just replace `stage: 2` to the desired zero optimization stage.
