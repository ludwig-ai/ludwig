# Llama2-7b Fine-Tuning 4bit (QLoRA)

This example shows how to fine-tune [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) to follow instructions.
Instruction tuning is the first step in adapting a general purpose Large Language Model into a chatbot.

This example uses no distributed training or big data functionality. It is designed to run locally on any machine
with GPU availability.

## Prerequisites

- [HuggingFace API Token](https://huggingface.co/docs/hub/security-tokens)
- Access approval to [Llama2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- GPU with at least 12 GiB of VRAM (in our tests, we used an Nvidia T4)

## Running the example

Set your token environment variable from the terminal, then run the API script:

```bash
export HUGGING_FACE_HUB_TOKEN="<api_token>"
python train_alpaca.py
```
