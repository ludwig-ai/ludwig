# Llama2-7b Fine-Tuning 4bit (QLoRA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c3AO8l_H6V_x37RwQ8V7M6A-RmcBf2tG?usp=sharing]

This example shows how to fine-tune [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) to follow instructions.
Instruction tuning is the first step in adapting a general purpose Large Language Model into a chatbot.

This example uses no distributed training or big data functionality. It is designed to run locally on any machine
with GPU availability.

## Prerequisites

- [HuggingFace API Token](https://huggingface.co/docs/hub/security-tokens)
- Access approval to [Llama2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- GPU with at least 12 GiB of VRAM (in our tests, we used an Nvidia T4)

## Running

### Command Line

Set your token environment variable from the terminal, then run the API script:

```bash
export HUGGING_FACE_HUB_TOKEN="<api_token>"
./run_train.sh
```

### Python API

Set your token environment variable from the terminal, then run the API script:

```bash
export HUGGING_FACE_HUB_TOKEN="<api_token>"
python train_alpaca.py
```

## Upload to HuggingFace

You can upload to the HuggingFace Hub from the command line:

```bash
ludwig upload hf_hub -r <your_org>/<model_name> -m <path/to/model>
```
