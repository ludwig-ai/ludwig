"""Ultra-slow end-to-end tests for advanced PEFT adapters.

Uses sshleifer/tiny-gpt2 (3MB) to verify each adapter configuration trains
for one step without crashing. These are NOT run in CI — run locally before releases.

Sections
--------
- LoRA with default init
- LoRA with PiSSA initializer
- LoRA with Gaussian initializer
- LoRA with per-layer rank_pattern
- TinyLoRA (LoRA-XS equivalent)
- LN-Tuning (layer-norm-only fine-tuning)
- IA3
"""

import tempfile

import numpy as np
import pandas as pd

from ludwig.api import LudwigModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
VOCAB = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "sat", "on", "a", "mat"]
N = 40  # small enough for a single-step smoke test


def _llm_train(adapter_config: dict) -> None:
    """Train tiny-gpt2 for 1 epoch with the given adapter config and assert training succeeds."""
    df = pd.DataFrame(
        {
            "prompt": [" ".join(RNG.choice(VOCAB, size=8).tolist()) for _ in range(N)],
            "response": [" ".join(RNG.choice(VOCAB, size=4).tolist()) for _ in range(N)],
        }
    )
    config = {
        "model_type": "llm",
        "base_model": "sshleifer/tiny-gpt2",
        "input_features": [{"name": "prompt", "type": "text"}],
        "output_features": [{"name": "response", "type": "text"}],
        "adapter": adapter_config,
        "trainer": {"type": "finetune", "epochs": 1, "batch_size": 4, "learning_rate": 1e-5},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        model = LudwigModel(config, logging_level=40)
        result = model.train(dataset=df, output_directory=tmpdir)
        assert result.train_stats is not None, "Training produced no stats"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPEFTAdaptersLocal:
    """Local (non-Ray) end-to-end smoke tests for each advanced PEFT adapter."""

    def test_lora_default(self):
        """Standard LoRA with default Kaiming initialization."""
        _llm_train({"type": "lora", "r": 4})

    def test_lora_pissa(self):
        """LoRA with PiSSA initializer (principal singular vectors, Meng et al., 2024)."""
        _llm_train({"type": "lora", "r": 4, "init_lora_weights": "pissa"})

    def test_lora_gaussian(self):
        """LoRA with Gaussian initialization (alternative to default Kaiming zeros init)."""
        _llm_train({"type": "lora", "r": 4, "init_lora_weights": "gaussian"})

    def test_lora_rank_pattern(self):
        """LoRA with per-module rank override via rank_pattern."""
        # tiny-gpt2 attention weight is named c_attn; override rank to 8 for that layer.
        _llm_train({"type": "lora", "r": 4, "rank_pattern": {"c_attn": 8}})

    def test_tinylora(self):
        """TinyLoRA: SVD-projected extreme parameter-efficient fine-tuning (LoRA-XS variant)."""
        _llm_train({"type": "tinylora", "r": 4})

    def test_ln_tuning(self):
        """LN-Tuning: tunes only the layer normalization parameters (~0.1% of params)."""
        _llm_train({"type": "ln_tuning"})

    def test_ia3(self):
        """IA3: scaling-vector-based adapter, even lighter than LoRA."""
        _llm_train({"type": "ia3"})
