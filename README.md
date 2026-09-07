<p align="center">
  <a href="https://ludwig.ai">
    <img src="https://github.com/ludwig-ai/ludwig-docs/raw/main/docs/images/ludwig_hero_smaller.jpg" height="150">
  </a>
</p>

<div align="center">

**Declarative deep learning framework for LLMs, multimodal models, and tabular AI.**

[![PyPI version](https://badge.fury.io/py/ludwig.svg)](https://badge.fury.io/py/ludwig)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/CBgdrGnZjy)
[![DockerHub](https://img.shields.io/docker/pulls/ludwigai/ludwig.svg)](https://hub.docker.com/r/ludwigai)
[![Downloads](https://pepy.tech/badge/ludwig)](https://pepy.tech/project/ludwig)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/ludwig-ai/ludwig/blob/main/LICENSE)
[![X](https://img.shields.io/twitter/follow/ludwig_ai.svg?style=social&logo=twitter)](https://twitter.com/ludwig_ai)

[**Docs**](https://ludwig.ai) · [**Getting Started**](https://ludwig.ai/latest/getting_started/) · [**Examples**](https://ludwig.ai/latest/examples) · [**Discord**](https://discord.gg/CBgdrGnZjy)

</div>

______________________________________________________________________

## What is Ludwig?

Ludwig is a **declarative deep learning framework** that lets you train, fine-tune, and deploy AI models — from LLM fine-tuning to tabular classification — using a YAML config file and zero boilerplate Python.

```yaml
# Fine-tune Llama-3.1 with LoRA in one config file
model_type: llm
base_model: meta-llama/Llama-3.1-8B
adapter:
  type: lora
trainer:
  type: finetune
  epochs: 3
input_features:
  - name: instruction
    type: text
output_features:
  - name: response
    type: text
```

```bash
ludwig train --config model.yaml --dataset my_data.csv
```

**Tech stack:** Python 3.12 · PyTorch 2.7+ · Pydantic 2 · Transformers 5 · Ray 2.54

Ludwig is hosted by the [Linux Foundation AI & Data](https://lfaidata.foundation/).

______________________________________________________________________

## What's New in Ludwig 0.16

| Feature                         | Description                                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **PatchTST & N-BEATS encoders** | State-of-the-art timeseries forecasting encoders with MASE/sMAPE metrics                               |
| **Advanced PEFT adapters**      | PiSSA, EVA, CorDA/LoftQ initializers; TinyLoRA, OFT, HRA, WaveFT, LN-Tuning, VBLoRA, C3A adapter types |
| **VLM fine-tuning**             | Train LLaVA, Qwen2-VL, InternVL via `is_multimodal: true` with gated cross-attention                   |
| **HyperNetwork combiner**       | Conditioning-based feature fusion — one feature generates weights for others                           |
| **Nash-MTL & Pareto-MTL**       | Game-theoretic and preference-based multi-task loss balancing                                          |
| **LLM config generation**       | `ludwig generate_config "describe your task"` — LLM writes the YAML for you                            |
| **ModelInspector**              | Architecture analysis, weight collection, feature importance proxy                                     |
| **Ray Serve & KServe**          | Distributed and Kubernetes-native model deployment shims                                               |
| **GRPO alignment**              | Reward-model-free RLHF via Group Relative Policy Optimization                                          |
| **torchao quantization + QAT**  | PyTorch-native `int4/int8/float8` with Quantization-Aware Training                                     |
| **Multi-adapter PEFT**          | Multiple named LoRA adapters with weighted merging (TIES, DARE, SVD)                                   |
| **Native Optuna executor**      | GPT/TPE/CMA-ES samplers, pruning, resumable SQLite/PostgreSQL storage                                  |
| **Timeseries forecasting**      | `model.forecast(dataset, horizon=N)` API with `TimeseriesOutputFeature`                                |
| **Muon & ScheduleFreeAdamW**    | New optimizers for large-scale pretraining and fine-tuning                                             |
| **Image segmentation decoders** | UNet, SegFormer, FPN decoders for semantic segmentation                                                |

______________________________________________________________________

## Installation

```bash
pip install ludwig           # core
pip install ludwig[full]     # all optional dependencies
pip install ludwig[llm]      # LLM fine-tuning only
```

Requires Python 3.12+. See [contributing](https://github.com/ludwig-ai/ludwig/blob/main/CONTRIBUTING.md) for a full dependency matrix.

______________________________________________________________________

## Quick Start

### Fine-tune an LLM (instruction tuning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c3AO8l_H6V_x37RwQ8V7M6A-RmcBf2tG?usp=sharing)

Ludwig supports the full LLM fine-tuning spectrum:

| Technique                         | Config key                                                               |
| --------------------------------- | ------------------------------------------------------------------------ |
| Supervised fine-tuning (SFT)      | `trainer.type: finetune`                                                 |
| DPO / KTO / ORPO / GRPO alignment | `trainer.type: dpo` (or `kto`, `orpo`, `grpo`)                           |
| LoRA / DoRA / VeRA / PiSSA        | `adapter.type: lora` (or `dora`, `vera`, `lora` + `init_weights: pissa`) |
| 4-bit QLoRA (bitsandbytes)        | `quantization.bits: 4`                                                   |
| torchao + QAT                     | `quantization.backend: torchao`                                          |
| Multi-adapter with merging        | `adapters:` dict + `merge:` block                                        |
| VLM (vision-language)             | `is_multimodal: true`                                                    |

```yaml
model_type: llm
base_model: meta-llama/Llama-3.1-8B

quantization:
  bits: 4

adapter:
  type: lora

prompt:
  template: |
    ### Instruction: {instruction}
    ### Input: {input}
    ### Response:

input_features:
  - name: prompt
    type: text

output_features:
  - name: output
    type: text

trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    decay: cosine
    warmup_fraction: 0.01

backend:
  type: local
```

```bash
export HUGGING_FACE_HUB_TOKEN="<your_token>"
ludwig train --config model.yaml --dataset "ludwig://alpaca"
```

### Train a multimodal classifier

```yaml
input_features:
  - name: review_text
    type: text
    encoder:
      type: bert
  - name: star_rating
    type: number
  - name: product_image
    type: image
    encoder:
      type: dinov2

output_features:
  - name: recommended
    type: binary
```

```bash
ludwig train --config model.yaml --dataset reviews.csv
```

### Generate a config from natural language

```bash
ludwig generate_config "I have a CSV with age, income, education level, and I want to predict loan default"
```

### Make predictions

```bash
ludwig predict --model_path results/experiment_run/model --dataset new_data.csv
```

### Launch a REST API

```bash
ludwig serve --model_path results/experiment_run/model
# POST http://localhost:8000/predict
```

______________________________________________________________________

## Capabilities

<details>
<summary><strong>LLM Fine-Tuning</strong></summary>

- **Supervised fine-tuning (SFT)** on instruction/response pairs
- **Alignment training**: DPO, KTO, ORPO, GRPO (reward-model-free RLHF)
- **PEFT adapters**: LoRA, DoRA, VeRA, LoRA+, TinyLoRA, OFT, HRA, WaveFT, LN-Tuning, VBLoRA, C3A
- **LoRA initializers**: PiSSA, EVA, CorDA, LoftQ for improved convergence
- **Multi-adapter PEFT**: multiple named adapters on one base model, switchable at runtime; merge with TIES, DARE, SVD, magnitude pruning
- **Quantization**: 4-bit/8-bit QLoRA (bitsandbytes), torchao int4/int8/float8 with QAT
- **VLM fine-tuning**: LLaVA, Qwen2-VL, InternVL via `is_multimodal: true`
- **Sequence packing** for efficient training on variable-length inputs
- **Paged and 8-bit optimizers** for memory-efficient training

</details>

<details>
<summary><strong>Multimodal & Tabular Models</strong></summary>

- **Input modalities**: text, numbers, categories, binary, sets, bags, sequences, images, audio, timeseries, vectors, dates
- **Text encoders**: any HuggingFace Transformer (BERT, RoBERTa, ModernBERT, Qwen3, Llama-3.1, etc.), plus Mamba-2, Jamba
- **Image encoders**: DINOv2, ConvNeXt, EfficientNet, ViT, CAFormer, ConvFormer, PoolFormer, TIMM (1000+ models)
- **Timeseries encoders**: PatchTST, N-BEATS, CNN, RNN, Transformer; MASE and sMAPE metrics; `model.forecast()` API
- **Combiners**: concat, transformer, tab_transformer, FT-Transformer, TabNet, TabPFN v2, HyperNetwork, ProjectAggregate, GatedFusion, Perceiver
- **Multi-task learning**: multiple output features in a single model; Nash-MTL, Pareto-MTL, FAMO, GradNorm, uncertainty loss balancing
- **Image segmentation**: UNet, SegFormer, FPN decoders

</details>

<details>
<summary><strong>Training Infrastructure</strong></summary>

- **Distributed training**: HuggingFace Accelerate with DDP, FSDP, DeepSpeed (zero-code changes)
- **Ray backend**: training across a Ray cluster, larger-than-memory datasets via Ray Data
- **Automatic batch size selection** and learning rate range test
- **Mixed precision** (fp16/bf16), gradient checkpointing, gradient accumulation
- **Optimizers**: AdamW, Adafactor, SGD, Muon, ScheduleFreeAdamW, Lion, paged/8-bit variants
- **Learning rate schedulers**: cosine, linear, polynomial, reduce-on-plateau, OneCycleLR
- **Model Soup**: uniform and greedy checkpoint averaging for better generalization at zero inference cost
- **Modality dropout** for robust multimodal models

</details>

<details>
<summary><strong>Hyperparameter Optimization</strong></summary>

- **Executors**: Ray Tune (ASHA, PBT, Bayesian) and native Optuna (auto/GP/TPE/CMA-ES)
- **Optuna persistence**: SQLite or PostgreSQL for resumable HPO runs
- **Pruning** with Optuna's MedianPruner and HyperbandPruner
- **Search spaces**: uniform, log-uniform, choice, randint, quantized
- **Full Ludwig config** is searchable — any nested parameter can be a hyperparameter

</details>

<details>
<summary><strong>Production & Deployment</strong></summary>

- **REST API**: FastAPI server with Prometheus metrics and structured logging (`ludwig serve`)
- **vLLM serving**: OpenAI-compatible API with PagedAttention and continuous batching
- **Ray Serve**: distributed deployment with auto-scaling and traffic splitting
- **KServe**: Kubernetes-native deployment with Open Inference Protocol v2
- **Model export**: SafeTensors (default), `torch.export` `.pt2` bundles, ONNX
- **HuggingFace Hub**: `ludwig upload hf_hub` — push model + auto-generated model card
- **Docker**: prebuilt containers at [ludwigai/ludwig](https://hub.docker.com/u/ludwigai)

</details>

<details>
<summary><strong>Tooling & Integrations</strong></summary>

- **Experiment tracking**: TensorBoard, Weights & Biases, Comet ML, MLflow, Aim Stack
- **Model inspection**: `ModelInspector` — weight enumeration, architecture summary, feature importance proxy
- **Visualizations**: learning curves, confusion matrices, calibration plots, ROC curves, hyperopt analysis
- **AutoML**: `ludwig.automl.auto_train()` — give it a dataset and a time budget; the YAML-driven search space samples encoder/combiner/decoder combinations and validates them before training
- **Dataset quality checks**: `from ludwig.utils.dataset_quality import check_dataset_quality` — validates a DataFrame before training (missing values, class imbalance, near-duplicate columns, ID leakage, …)
- **OpenML integration**: load any OpenML task directly — `OpenMLLoader` fetches by task ID and caches locally as Parquet
- **LLM config generation**: `ludwig generate_config "describe your task"` — LLM writes the YAML
- **K-fold cross-validation**: `ludwig experiment --k_fold N`
- **Dataset Zoo**: 70+ built-in benchmark datasets (`ludwig://mnist`, `ludwig://alpaca`, …)

</details>

______________________________________________________________________

## Examples

### LLM & Alignment

| Use Case                              | Link                                                                                |
| ------------------------------------- | ----------------------------------------------------------------------------------- |
| LLM instruction tuning (LoRA + QLoRA) | [examples/llm](https://ludwig.ai/latest/examples/llm/llm_finetuning)                |
| DPO / GRPO alignment                  | [examples/llm/alignment](https://ludwig.ai/latest/examples/llm/alignment)           |
| Advanced PEFT (PiSSA, OFT, VBLoRA, …) | [examples/llms/peft_advanced](https://ludwig.ai/latest/examples/llms/peft_advanced) |
| VLM fine-tuning (LLaVA, Qwen2-VL)     | [examples/vlm](https://github.com/ludwig-ai/ludwig/tree/main/examples/vlm)          |

### Tabular & Multimodal

| Use Case                               | Link                                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Binary classification (Titanic)        | [examples/titanic](https://ludwig.ai/latest/examples/titanic)                                     |
| Tabular classification (census income) | [examples/adult_census_income](https://ludwig.ai/latest/examples/adult_census_income)             |
| Multimodal classification              | [examples/multimodal_classification](https://ludwig.ai/latest/examples/multimodal_classification) |
| Multi-task learning                    | [examples/multi_task](https://ludwig.ai/latest/examples/multi_task)                               |

### Timeseries & Vision

| Use Case                                   | Link                                                                                      |
| ------------------------------------------ | ----------------------------------------------------------------------------------------- |
| Timeseries forecasting (PatchTST, N-BEATS) | [examples/forecasting](https://ludwig.ai/latest/examples/forecasting)                     |
| Weather forecasting                        | [examples/weather](https://ludwig.ai/latest/examples/weather)                             |
| Image classification (MNIST)               | [examples/mnist](https://ludwig.ai/latest/examples/mnist)                                 |
| Semantic segmentation                      | [examples/semantic_segmentation](https://ludwig.ai/latest/examples/semantic_segmentation) |

### NLP & Audio

| Use Case                 | Link                                                                                    |
| ------------------------ | --------------------------------------------------------------------------------------- |
| Text classification      | [examples/text_classification](https://ludwig.ai/latest/examples/text_classification)   |
| Named entity recognition | [examples/ner_tagging](https://ludwig.ai/latest/examples/ner_tagging)                   |
| Machine translation      | [examples/machine_translation](https://ludwig.ai/latest/examples/machine_translation)   |
| Speech recognition       | [examples/speech_recognition](https://ludwig.ai/latest/examples/speech_recognition)     |
| Speaker verification     | [examples/speaker_verification](https://ludwig.ai/latest/examples/speaker_verification) |

______________________________________________________________________

## Why Ludwig?

- **Zero boilerplate** — no training loop, no data pipeline, no evaluation code. The YAML config is the entire program.
- **Best-in-class LLM support** — full spectrum from LoRA to GRPO alignment, torchao QAT, and VLM fine-tuning, all in config.
- **Multimodal out of the box** — mix text, images, numbers, audio, and timeseries with one config change.
- **Scale without code changes** — go from laptop → multi-GPU → Ray cluster by changing `backend.type`.
- **Expert control when you need it** — every activation function, scheduler, and optimizer is configurable.
- **Reproducible research** — every run is logged and the full config is saved. Compare experiments with `ludwig visualize`.

______________________________________________________________________

## Publications

- [Ludwig: A Type-Based Declarative Deep Learning Toolbox](https://arxiv.org/pdf/1909.07930.pdf) (2019)
- [Declarative Machine Learning Systems](https://arxiv.org/pdf/2107.08148.pdf) (2021)
- [Ludwig's State-of-the-Art Benchmarks](https://openreview.net/pdf?id=hwjnu6qW7E4)

______________________________________________________________________

## Community

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/CBgdrGnZjy)

- [Discord](https://discord.gg/CBgdrGnZjy) — ask questions, share what you've built
- [GitHub Issues](https://github.com/ludwig-ai/ludwig/issues) — bugs and feature requests
- [X / Twitter](https://twitter.com/ludwig_ai) — announcements
- [Medium](https://medium.com/ludwig-ai) — tutorials and deep-dives

<a href="https://github.com/ludwig-ai/ludwig/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ludwig-ai/ludwig" />
</a>


## 🌐 Web Resources & Interactive Index
- [CUBICA](https://eduquests.netlify.app/cubica.html)
- [CATEGORY ROBOT49](https://learnaction.netlify.app/category-robot49.html)
- [JELI2D](https://learnaction.netlify.app/jeli2d.html)
- [ZOO SHAP](https://ieduquests.web.app/zoo-shap.html)
- [CATEGORY CAR](https://welearnaction.onrender.com/category-car.html)
- [THE BRANCH RUNNER](https://eduquests.pages.dev/the-branch-runner.html)
- [CATEGORY LOGIC536](https://eduquests.netlify.app/category-logic536.html)
- [CATEGORY SOCCER 2](https://eduquests.pages.dev/category-soccer-2.html)
- [CATEGORY SCHOOL](https://ieduquests.web.app/category-school.html)
- [LINE ON HOLE](https://ieduquests.web.app/line-on-hole.html)
- [BOMBAMAN 3D](https://welearnaction.onrender.com/bombaman-3d.html)
- [BLOCKS STACK RUSH](https://ieduquests.web.app/blocks-stack-rush.html)
- [CAT SORTER PUZZLE](https://ieduquests.web.app/cat-sorter-puzzle.html)
- [INDEX20](https://eduquests.onrender.com/index20.html)
- [CATEGORY MINECRAFT 2](https://eduquests.github.io/category-minecraft-2.html)
- [CATEGORY CAR](https://ieduquests.web.app/category-car.html)
- [UNTANGLE RINGS MASTER](https://ieduquests.web.app/untangle-rings-master.html)
- [BEAR VS HUMANS](https://eduquests.netlify.app/bear-vs-humans.html)
- [BILLIARDS 3D RUSSIAN PYRAMID](https://eduquests.netlify.app/billiards-3d-russian-pyramid.html)
- [RED STICKMAN VS CRAFTMANS 2](https://eduquests.netlify.app/red-stickman-vs-craftmans-2.html)
- [CATEGORY DEFENSE](https://eduquests.github.io/category-defense.html)
- [BLOCKS BREAKER](https://welearnaction.onrender.com/blocks-breaker.html)
- [CATEGORY GROW GAMES](https://ieduquests.web.app/category-grow-games.html)
- [CATEGORY DRAWING](https://ieduquests.web.app/category-drawing.html)
- [FOOTBALL HEADS 2025](https://ieduquests.web.app/football-heads-2025.html)
- [SPACE SURVIVOR](https://eduquests.netlify.app/space-survivor.html)
- [SWEEPER CURLING](https://learnaction.netlify.app/sweeper-curling.html)
- [DRAGON HUNTER](https://eduquests.netlify.app/dragon-hunter.html)
- [TRAFFIC ESCAPE PUZZLE](https://ieduquests.web.app/traffic-escape-puzzle.html)
- [MAHJONG LINES](https://eduquests.onrender.com/mahjong-lines.html)
- [RACING BALL ADVENTURE](https://learnaction.netlify.app/racing-ball-adventure.html)
- [BOAT MANIA](https://ieduquests.web.app/boat-mania.html)
- [CATEGORY SOCCER 2](https://eduquests.onrender.com/category-soccer-2.html)
- [CATEGORY CARTOON76](https://eduquests.netlify.app/category-cartoon76.html)
- [THE PATAGONIANS](https://ieduquests.web.app/the-patagonians.html)
- [IDLE HOTEL EMPIRE](https://ieduquests.web.app/idle-hotel-empire.html)
- [CATEGORY PUZZLE 5](https://eduquests.netlify.app/category-puzzle-5.html)
- [CATEGORY BOARDGAMES](https://eduquests.netlify.app/category-boardgames.html)
- [OBBY PINATA PARTY](https://ieduquests.web.app/obby-pinata-party.html)
- [FINGER SOCCER TOURNAMENT](https://ieduquests.web.app/finger-soccer-tournament.html)
- [RUNIC BLOCK COLLAPSE](https://eduquests.netlify.app/runic-block-collapse.html)
- [CATEGORY BATTLE524](https://eduquests.netlify.app/category-battle524.html)
- [MERGE TIKTOK GRAVITY KNIFE](https://ieduquests.web.app/merge-tiktok-gravity-knife.html)
- [INDEX9](https://learnaction.netlify.app/index9.html)
- [ZOMBIE RAFT](https://ieduquests.web.app/zombie-raft.html)
- [GRANNY GTA VEGAS](https://ieduquests.web.app/granny-gta-vegas.html)
- [HIDDEN OBJECTS ISLAND](https://ieduquests.web.app/hidden-objects-island.html)
- [ONLINE PORTAL](https://eduquests.github.io/)
- [BALL EATING SIMULATOR](https://ieduquests.web.app/ball-eating-simulator.html)
- [CATEGORY PARTY23](https://eduquests.onrender.com/category-party23.html)
- [BURGER TYCOON](https://ieduquests.web.app/burger-tycoon.html)
- [CATEGORY BASKETBALL 2](https://eduquests.onrender.com/category-basketball-2.html)
- [CATEGORY ADVENTURE 2](https://eduquests.netlify.app/category-adventure-2.html)
- [LOVE IN STYLE](https://ieduquests.web.app/love-in-style.html)
- [99 BALLS](https://eduquests.onrender.com/99-balls.html)
- [CRAZY 2248 LINK MATCHING PUZZLE GAME](https://ieduquests.web.app/crazy-2248-link-matching-puzzle-game.html)
- [CATEGORY MANAGEMENT](https://welearnaction.onrender.com/category-management.html)
- [SAVE STRANDING FISH](https://ieduquests.web.app/save-stranding-fish.html)
- [CATEGORY FLASH 2](https://eduquests.github.io/category-flash-2.html)
- [VALENTINES HIDDEN ALPHAWORDS](https://welearnaction.onrender.com/valentines-hidden-alphawords.html)
- [SITEMAP](https://skillcrafts.github.io/sitemap.html)
- [CATEGORY 2048](https://eduquests.pages.dev/category-2048.html)
- [ZOMBIE CONQUER COUNTRIES](https://ieduquests.web.app/zombie-conquer-countries.html)
- [TOY CLAW SIMULATOR](https://eduquests.netlify.app/toy-claw-simulator.html)
- [CATEGORY CARDS](https://ieduquests.web.app/category-cards.html)
- [FUN OBBY EXTREME](https://eduquests.netlify.app/fun-obby-extreme.html)
- [OBBY CARDS THE LEGEND HUNT](https://eduquests.netlify.app/obby-cards-the-legend-hunt.html)
- [SWEET BUSINESS OF CATS CAKES](https://learnaction.netlify.app/sweet-business-of-cats-cakes.html)
- [SWEET MATCH](https://ieduquests.web.app/sweet-match.html)
- [FLOWER SORT](https://learnaction.netlify.app/flower-sort.html)
- [CATEGORY MINECRAFT 2](https://eduquests.onrender.com/category-minecraft-2.html)
- [GIANT RUN 3D](https://ieduquests.web.app/giant-run-3d.html)
- [SUPER SPRUNKI CLICKER](https://welearnaction.onrender.com/super-sprunki-clicker.html)
- [SQUAD ASSEMBLER](https://learnaction.netlify.app/squad-assembler.html)
- [CATEGORY FASHION105](https://learnaction.netlify.app/category-fashion105.html)
- [CATEGORY PIXEL](https://eduquests.github.io/category-pixel.html)
- [CAPYBARA SUIKA](https://eduquests.netlify.app/capybara-suika.html)
- [CATEGORY CASUAL 2](https://eduquests.netlify.app/category-casual-2.html)
- [UP HERO](https://ieduquests.web.app/up-hero.html)
- [GUNS VS MAGIC](https://eduquests.github.io/guns-vs-magic.html)
- [CATEGORY ESCAPE187](https://eduquests.onrender.com/category-escape187.html)
- [CATEGORY COLLECT565](https://welearnaction.onrender.com/category-collect565.html)
- [LINK FLOW](https://welearnaction.onrender.com/link-flow.html)
- [TETRO MERGE](https://learnaction.netlify.app/tetro-merge.html)
- [MOSCOW METRO DRIVER 3D](https://learnaction.netlify.app/moscow-metro-driver-3d.html)
- [COFFEE MATCH RUSH SORT PUZZLE](https://ieduquests.web.app/coffee-match-rush-sort-puzzle.html)
- [PORTAL MASTER](https://eduquests.github.io/portal-master.html)
- [BLOCK EATING SIMULATOR](https://learnaction.netlify.app/block-eating-simulator.html)
- [MAHJONG PET QUEST](https://welearnaction.onrender.com/mahjong-pet-quest.html)
- [BUBBLE SHOOTER AURA](https://eduquests.netlify.app/bubble-shooter-aura.html)
- [MERGE 2048 GUN RUSH](https://ieduquests.web.app/merge-2048-gun-rush.html)
- [BLOCK CUT CLEANER](https://ieduquests.web.app/block-cut-cleaner.html)
- [JELLO BUBBLES](https://learnaction.netlify.app/jello-bubbles.html)
- [CATEGORY PLATFORM](https://eduquests.github.io/category-platform.html)
- [CATEGORY DRAGON](https://ieduquests.web.app/category-dragon.html)
- [CATEGORY MOBILE2 112](https://eduquests.pages.dev/category-mobile2-112.html)
- [CATEGORY DESTROY256](https://ieduquests.web.app/category-destroy256.html)
- [SWORD AND SPIN](https://learnaction.netlify.app/sword-and-spin.html)
- [CATEGORY THINKY](https://eduquests.github.io/category-thinky.html)
- [CATEGORY BOXING12](https://eduquests.pages.dev/category-boxing12.html)
- [BACKROOMS AMONG IMPOSTOR ROLLING GIANT](https://ieduquests.web.app/backrooms-among-impostor-rolling-giant.html)
- [PLANT GIRL DEFENSE ZOMBIE](https://welearnaction.onrender.com/plant-girl-defense-zombie.html)
- [CATEGORY CARDS](https://eduquests.pages.dev/category-cards.html)
- [FISHING THE RUSSIAN WAY](https://ieduquests.web.app/fishing-the-russian-way.html)
- [TRICKY CHALLENGES MINI GAMES](https://ieduquests.web.app/tricky-challenges-mini-games.html)
- [HEROIC KNIGHT](https://ieduquests.web.app/heroic-knight.html)
- [ARROW SORTING](https://ieduquests.web.app/arrow-sorting.html)
- [LOST THINGS HIDDEN OBJECTS](https://eduquests.github.io/lost-things-hidden-objects.html)
- [OHPEACH IT](https://eduquests.netlify.app/ohpeach-it.html)
- [SWORDSMAN ADVENTURE](https://eduquests.netlify.app/swordsman-adventure.html)
- [FRUIT SURVIVOR](https://eduquests.netlify.app/fruit-survivor.html)
- [CATEGORY FASHION](https://eduquests.pages.dev/category-fashion.html)
- [BATTLE ZONE 2D](https://welearnaction.onrender.com/battle-zone-2d.html)
- [CATEGORY CAT55](https://eduquests.pages.dev/category-cat55.html)
- [CATEGORY ESCAPE 2](https://learnaction.netlify.app/category-escape-2.html)
- [CATEGORY DRESS UP97](https://learnaction.netlify.app/category-dress-up97.html)
- [BUILD YOUR AQUARIUM](https://eduquests.onrender.com/build-your-aquarium.html)
- [MERGE HOSPITAL](https://eduquests.github.io/merge-hospital.html)
- [MEMEVOIO](https://eduquests.netlify.app/memevoio.html)
- [INDEX31](https://eduquests.pages.dev/index31.html)
- [POLICE CAR LINE DRIVING](https://eduquests.onrender.com/police-car-line-driving.html)
- [EAT AND GROW FISH](https://learnaction.netlify.app/eat-and-grow-fish.html)
- [HALLOWEEN FRUIT SLICE](https://learnaction.netlify.app/halloween-fruit-slice.html)
- [SNIPER FREEZE](https://eduquests.github.io/sniper-freeze.html)
- [MERGE WAR](https://eduquests.onrender.com/merge-war.html)
- [CATEGORY UNBLOCKED WEBSITES](https://eduquests.github.io/category-unblocked-websites.html)
- [CATEGORY BIKE](https://eduquests.pages.dev/category-bike.html)
- [CATEGORY DRESS UP](https://eduquests.netlify.app/category-dress-up.html)
- [LAZY DOG](https://eduquests.netlify.app/lazy-dog.html)
- [ANIMAL MERGE ZOO PARK](https://eduquests.netlify.app/animal-merge-zoo-park.html)
