# Vision-Language Model Fine-Tuning with Ludwig

Fine-tune a vision-language model (VLM) on a visual-question-answering dataset using
Ludwig's `is_multimodal: true` flag. The example uses **Qwen2-VL-7B-Instruct** with
LoRA + 4-bit quantisation to fit on a single 24 GB GPU, but the same config works with
any HuggingFace `AutoModelForVision2Seq`-compatible model (LLaVA, InternVL, etc.).

## Dataset format

A CSV file with three columns:

| column       | description                               |
| ------------ | ----------------------------------------- |
| `image_path` | Path to image file (JPEG / PNG)           |
| `question`   | Natural-language question about the image |
| `answer`     | Expected answer (fine-tuning target)      |

## Setup

```bash
pip install "ludwig[llm]"          # transformers, peft, bitsandbytes
# Authenticate with HuggingFace if using a gated model
huggingface-cli login
```

## Run

```bash
python run.py --dataset /path/to/vqa.csv --output_dir ./results
```

Override the base model:

```bash
python run.py \
  --dataset /path/to/vqa.csv \
  --base_model llava-hf/llava-1.5-7b-hf
```

## Config highlights

```yaml
is_multimodal: true        # use AutoModelForVision2Seq + AutoProcessor
trust_remote_code: true    # required for Qwen2-VL custom architecture

adapter:
  type: lora
  r: 16
  alpha: 32

quantization:
  bits: 4
  quantization_type: nf4
  compute_dtype: bfloat16
```

## Supported VLM architectures

Any model loadable via `AutoModelForVision2Seq` works out of the box:

- `Qwen/Qwen2-VL-*`
- `llava-hf/llava-1.5-*`
- `llava-hf/llava-v1.6-*`
- `OpenGVLab/InternVL2-*`
- `microsoft/phi-3-vision-*` (also needs `trust_remote_code: true`)
