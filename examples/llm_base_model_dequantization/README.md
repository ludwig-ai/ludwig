# Convert quantized base model to fp16

Ludwig has utility functions to convert nf4 quantized bitsandbytes base models back to fp16
for more efficient inference. This is desireable since inference with bitsandbytes is slow because
every forward pass through the model requires dequantizing the model weights from nf4 to fp16 layer
by layer and then quantizing it back to nf4 to keep memory usage constant.

By dequantizing the base model in fp16 upfront, you can get the same effect of the quantized weights
without sacrificing on inference performance.

## Visual Illustration

### Without dequantization upfront

| **Request 1:**                             | **Request 2:**                             | **Request 3:**                             |
| ------------------------------------------ | ------------------------------------------ | ------------------------------------------ |
| - Quantized bitsandbytes model             | - Quantized bitsandbytes model             | - Quantized bitsandbytes model             |
| - Dequantization of layer 1 (nf4 to fp16)  | - Dequantization of layer 1 (nf4 to fp16)  | - Dequantization of layer 1 (nf4 to fp16)  |
| - Forward Pass (using dequantized weights) | - Forward Pass (using dequantized weights) | - Forward Pass (using dequantized weights) |
| - Quantization of layer 1 (fp16 to nf4)    | - Quantization of layer 1 (fp16 to nf4)    | - Quantization of layer 1 (fp16 to nf4)    |
| - Dequantization of layer 2 (nf4 to fp16)  | - Dequantization of layer 2 (nf4 to fp16)  | - Dequantization of layer 2 (nf4 to fp16)  |
| - Forward Pass (using dequantized weights) | - Forward Pass (using dequantized weights) | - Forward Pass (using dequantized weights) |
| - Quantization of layer 2 (fp16 to nf4)    | - Quantization of layer 2 (fp16 to nf4)    | - Quantization of layer 2 (fp16 to nf4)    |
| - ...                                      | - ...                                      | - ...                                      |
| - Final Output                             | - Final Output                             | - Final Output                             |

### With dequantization upfront

| **Request 1:**                   | **Request 2:**                   | **Request 3:**                   |
| -------------------------------- | -------------------------------- | -------------------------------- |
| - Dequantized base model in fp16 | - Dequantized base model in fp16 | - Dequantized base model in fp16 |
| - Forward pass through layer 1   | - Forward pass through layer 1   | - Forward pass through layer 1   |
| - Forward pass through layer 2   | - Forward pass through layer 2   | - Forward pass through layer 2   |
| - ...                            | - ...                            | - ...                            |
| - Final Output                   | - Final Output                   | - Final Output                   |

## Running the example script

The example `phi_2_dequantization.py` shows how you how you can quantize and then dequantized Phi-2. This process
can be repeated for any other base model supported by Ludwig that is quantized using 4 bits nf4 bitsandbytes quantization. You will need a GPU to run the script successfully.

Beneath the surface, this script:

1. Loads the base model in 4 bit nf4 quantization
1. Dequantizes the model layer by layer back into fp16 in-place.
1. Write the new dequantized weights to disk at `save_path`
1. Write the tokenizer to disk at `save_path`

Make sure you update the paths at the top of the file for base model, save path, and huggingface repo ID!

## Bonus

If desired, you can also use Ludwig to push the new dequantized model weights straight to HuggingFace hub!

```python
from ludwig.utils.hf_utils import upload_folder_to_hfhub

upload_folder_to_hfhub(repo_id=hfhub_repo_id, folder_path=save_path)
```

### Dequantized base models already on huggingface hub

- [CodeLlama 7b Instruct](https://huggingface.co/arnavgrg/codallama-7b-instruct-nf4-fp16-upscaled)
- [CodeLlama 13b Instruct](https://huggingface.co/arnavgrg/codellama-13b-instruct-nf4-fp16-upscaled)
- [CodeLlama 70b Instruct](https://huggingface.co/arnavgrg/codellama-70b-instruct-nf4-fp16-upscaled)
- [Llama 2 7b](https://huggingface.co/arnavgrg/llama-2-7b-nf4-fp16-upscaled)
- [Llama 2 7b Chat](https://huggingface.co/arnavgrg/llama-2-7b-chat-nf4-fp16-upscaled)
- [Llama 2 13b Chat](https://huggingface.co/arnavgrg/llama-2-13b-chat-nf4-fp16-upscaled)
- [Llama 2 70b Chat](https://huggingface.co/arnavgrg/llama-2-70b-chat-nf4-fp16-upscaled)
- [Mistral 7b](https://huggingface.co/arnavgrg/mistral-7b-nf4-fp16-upscaled)
- [Mistral 7b Instruct](https://huggingface.co/arnavgrg/mistral-7b-instruct-nf4-fp16-upscaled)
- [NousMistral Yarn 7b 128K](https://huggingface.co/arnavgrg/NousResearch-Yarn-Mistral-7b-128k-nf4-fp16-upscaled)
- [Microsoft Phi-2](https://huggingface.co/arnavgrg/phi-2-nf4-fp16-upscaled)
- [Zephyr 7b Beta](https://huggingface.co/arnavgrg/zephyr-7b-beta-nf4-fp16-upscaled)
