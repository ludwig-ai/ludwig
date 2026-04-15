# Semantic Segmentation: UNet, SegFormer, and FPN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ludwig-ai/ludwig/blob/main/examples/semantic_segmentation/semantic_segmentation.ipynb)

Semantic segmentation assigns a class label to every pixel in an image.
This example trains three different decoder architectures on the **CamSeq01**
urban driving dataset (101 images, 32 semantic classes) and compares their
accuracy/speed trade-offs.

## Decoder comparison

| Decoder     | Architecture                                                               | Recommended encoder                | Approx. extra params     | Best for                                                                 |
| ----------- | -------------------------------------------------------------------------- | ---------------------------------- | ------------------------ | ------------------------------------------------------------------------ |
| `unet`      | Symmetric encoder-decoder with skip connections; configurable `num_stages` | Built-in `unet` encoder            | ~31M (depth 4)           | General purpose baseline, no pretrained backbone needed                  |
| `segformer` | Lightweight all-MLP head fusing multi-scale ViT features                   | `dinov2` (DINOv2-base, pretrained) | ~2M head + ~86M backbone | Highest accuracy; transformer features transfer well to dense prediction |
| `fpn`       | Feature Pyramid Network top-down pathway with lateral connections          | `efficientnet` (pretrained)        | ~2M head + ~5M backbone  | Fast inference; handles objects at multiple scales efficiently           |

## Dataset

[CamSeq01](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamSeq01/) is a
set of 101 road-scene images captured in Cambridge, UK at 960×720 resolution
with 32 semantic class annotations.

Ludwig ships a built-in downloader — see [`camseq.py`](camseq.py) for the
standalone script or use `from ludwig.datasets import camseq` in Python.

## Config files

| File                     | Decoder     | Notes                                    |
| ------------------------ | ----------- | ---------------------------------------- |
| `config_camseq.yaml`     | `unet`      | Original baseline config                 |
| `config_unet_depth.yaml` | `unet`      | Shows the `num_stages` parameter         |
| `config_segformer.yaml`  | `segformer` | DINOv2 backbone, fine-tuned end-to-end   |
| `config_fpn.yaml`        | `fpn`       | EfficientNet backbone, larger batch size |

## Running the examples

**Prerequisites**: a CUDA-capable GPU. An A100 or equivalent is recommended
for the SegFormer run; the UNet and FPN configs run well on a single V100/3090.

```bash
pip install 'ludwig[vision]'
```

### UNet (configurable depth)

```bash
python camseq.py  # uses config_camseq.yaml (depth 4 by default)
```

Or with the explicit depth config:

```bash
ludwig train --config config_unet_depth.yaml
```

### SegFormer

```bash
ludwig train --config config_segformer.yaml
```

### FPN

```bash
ludwig train --config config_fpn.yaml
```

### UNet depth ablation

```bash
python unet_depth_sweep.py
```

This script trains models with `num_stages` ∈ {2, 3, 4, 5} and prints a
summary table of parameter count vs. best validation loss vs. training time.

### Interactive notebook

Open `semantic_segmentation.ipynb` locally or click the Colab badge above.
The notebook walks through all three decoders and produces side-by-side
visualisations of their predictions.

## Key config parameters

### UNet decoder

```yaml
decoder:
  type: unet
  num_stages: 4    # 2–5; input size must be divisible by 2^num_stages
  num_fc_layers: 0
  conv_norm: batch
```

### SegFormer decoder

```yaml
decoder:
  type: segformer
  hidden_size: 256  # MLP projection width
  dropout: 0.1
```

### FPN decoder

```yaml
decoder:
  type: fpn
  num_channels: 256  # lateral projection width at each pyramid level
  num_levels: 4      # number of pyramid levels (typical range 2–5)
```
