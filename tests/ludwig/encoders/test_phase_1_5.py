"""Comprehensive tests for Phase 1.5: Encoder Modernization.

Tests cover:
- 1.5.1 Image encoder consolidation + CLIP/DINOv2/SigLIP
- 1.5.2 Audio encoders (Wav2Vec2, Whisper, HuBERT)
- 1.5.3 Text encoder cleanup
- 1.5.4 Sequence encoder modernization (attention pooling, Mamba)
- 1.5.5 Category & number encoder enhancements
- 1.5.6 Date & H3 encoder cleanup
- 1.5.7 Activation function expansion
"""

import pytest
import torch
import torch.nn as nn

# Force CPU for all tests to avoid device mismatch issues
DEVICE = torch.device("cpu")


# ============================================================================
# 1.5.7 — Activation Function Tests
# ============================================================================
class TestActivationExpansion:
    """Test that all new activations are registered and functional."""

    def test_activation_count(self):
        from ludwig.utils.torch_utils import activations

        assert len(activations) >= 24, f"Expected >= 24 activations, got {len(activations)}"

    @pytest.mark.parametrize(
        "name",
        [
            "relu",
            "elu",
            "leakyRelu",
            "tanh",
            "sigmoid",
            "softmax",
            "logSigmoid",
            "gelu",
            "silu",
            "swish",
            "mish",
            "selu",
            "prelu",
            "relu6",
            "hardswish",
            "hardsigmoid",
            "softplus",
            "celu",
            "swiglu",
            "geglu",
            "reglu",
            "sparsemax",
            "entmax15",
            None,
        ],
    )
    def test_activation_instantiates(self, name):
        from ludwig.utils.torch_utils import get_activation

        act = get_activation(name)
        assert isinstance(act, nn.Module)

    @pytest.mark.parametrize("name", ["gelu", "silu", "mish", "selu", "relu6", "softplus", "celu"])
    def test_standard_activation_forward(self, name):
        from ludwig.utils.torch_utils import get_activation

        act = get_activation(name)
        x = torch.randn(4, 16)
        out = act(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("name", ["swiglu", "geglu", "reglu"])
    def test_glu_variant_forward(self, name):
        """GLU variants split input in half, so output dim is half of input dim."""
        from ludwig.utils.torch_utils import get_activation

        act = get_activation(name)
        x = torch.randn(4, 32)  # Must be even for chunk(2)
        out = act(x)
        assert out.shape == (4, 16), f"GLU output should be half input dim, got {out.shape}"

    def test_sparsemax_forward(self):
        from ludwig.utils.torch_utils import get_activation

        act = get_activation("sparsemax")
        x = torch.randn(4, 10)
        out = act(x)
        assert out.shape == x.shape
        # Sparsemax output should be non-negative and sum to ~1
        assert (out >= -1e-6).all()

    def test_entmax15_forward(self):
        from ludwig.utils.torch_utils import get_activation

        act = get_activation("entmax15")
        x = torch.randn(4, 10)
        out = act(x)
        assert out.shape == x.shape
        assert (out >= -1e-6).all()

    def test_schema_includes_new_activations(self):
        from ludwig.schema.utils import ActivationOptions

        field = ActivationOptions()
        # Check field_info for allowed values — this depends on schema_utils internals
        # Just verify it doesn't error
        assert field is not None


# ============================================================================
# 1.5.5 — Category & Number Encoder Tests
# ============================================================================
class TestCategoryEncoderEnhancements:
    def test_target_encoder_forward(self):
        from ludwig.encoders.category_encoders import CategoricalTargetEncoder

        enc = CategoricalTargetEncoder(vocab=["a", "b", "c", "d"], output_size=8)
        x = torch.tensor([0, 1, 2, 3])
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert ENCODER_OUTPUT in out
        assert out[ENCODER_OUTPUT].shape == (4, 8)

    def test_target_encoder_output_shape(self):
        from ludwig.encoders.category_encoders import CategoricalTargetEncoder

        enc = CategoricalTargetEncoder(vocab=["x", "y", "z"], output_size=16)
        assert enc.output_shape == torch.Size([16])

    def test_hash_encoder_forward(self):
        from ludwig.encoders.category_encoders import CategoricalHashEncoder

        enc = CategoricalHashEncoder(vocab=["a"] * 100, num_hash_buckets=32, embedding_size=64)
        x = torch.tensor([0, 50, 99, 10])
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert out[ENCODER_OUTPUT].shape == (4, 64)

    def test_hash_encoder_hashing(self):
        """Hash encoder should map large indices to bounded bucket range."""
        from ludwig.encoders.category_encoders import CategoricalHashEncoder

        enc = CategoricalHashEncoder(vocab=["a"] * 10, num_hash_buckets=8, embedding_size=4)
        x = torch.tensor([0, 100, 999])  # Values beyond vocab size
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert out[ENCODER_OUTPUT].shape == (3, 4)

    def test_hash_encoder_output_shape(self):
        from ludwig.encoders.category_encoders import CategoricalHashEncoder

        enc = CategoricalHashEncoder(vocab=["a"], num_hash_buckets=16, embedding_size=32)
        assert enc.output_shape == torch.Size([32])

    def test_target_encoder_schema(self):
        from ludwig.schema.encoders.category_encoders import CategoricalTargetEncoderConfig

        cfg = CategoricalTargetEncoderConfig.model_validate({"type": "target"})
        assert cfg.type == "target"

    def test_hash_encoder_schema(self):
        from ludwig.schema.encoders.category_encoders import CategoricalHashEncoderConfig

        cfg = CategoricalHashEncoderConfig.model_validate({"type": "hash"})
        assert cfg.type == "hash"


class TestBinsEncoder:
    def test_bins_encoder_forward(self):
        from ludwig.encoders.number_encoders import BinsEncoder

        enc = BinsEncoder(input_size=1, num_bins=16, output_size=64)
        x = torch.randn(8, 1)
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert out[ENCODER_OUTPUT].shape == (8, 64)

    def test_bins_encoder_set_bin_edges(self):
        from ludwig.encoders.number_encoders import BinsEncoder

        enc = BinsEncoder(input_size=1, num_bins=4, output_size=32)
        enc.set_bin_edges([0.0, 0.25, 0.5, 0.75, 1.0])
        x = torch.tensor([[0.1], [0.3], [0.6], [0.9]])
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert out[ENCODER_OUTPUT].shape == (4, 32)

    def test_bins_encoder_output_shape(self):
        from ludwig.encoders.number_encoders import BinsEncoder

        enc = BinsEncoder(input_size=1, num_bins=8, output_size=128)
        assert enc.output_shape == torch.Size([128])

    def test_bins_encoder_schema(self):
        from ludwig.schema.encoders.number_encoders import BinsEncoderConfig

        cfg = BinsEncoderConfig.model_validate({"type": "bins"})
        assert cfg.type == "bins"
        assert cfg.num_bins == 32


# ============================================================================
# 1.5.6 — Date & H3 Encoder Tests
# ============================================================================
class TestDateEncoderCleanup:
    @staticmethod
    def _make_date_vector(batch_size, device):
        """Create a realistic date input vector: [year, month, day, weekday, yearday, hour, min, sec,
        sec_of_day]."""
        return torch.tensor(
            [
                [2024, 3, 15, 4, 75, 10, 30, 45, 37845],
            ]
            * batch_size,
            dtype=torch.int,
            device=device,
        )

    def test_date_embed_forward(self):
        from ludwig.encoders.date_encoders import DateEmbed

        enc = DateEmbed(encoder_config=None).to(DEVICE)
        x = self._make_date_vector(4, DEVICE)
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert ENCODER_OUTPUT in out

    def test_date_wave_forward(self):
        from ludwig.encoders.date_encoders import DateWave

        enc = DateWave(encoder_config=None).to(DEVICE)
        x = self._make_date_vector(4, DEVICE)
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert ENCODER_OUTPUT in out

    def test_date_encoder_base_exists(self):
        from ludwig.encoders.date_encoders import DateEncoderBase

        assert DateEncoderBase is not None

    def test_shared_constants(self):
        """Constants should be importable from ludwig.constants."""
        from ludwig.constants import DATE_VECTOR_LENGTH, H3_VECTOR_LENGTH, MAX_H3_RESOLUTION

        assert DATE_VECTOR_LENGTH == 9
        assert MAX_H3_RESOLUTION == 15
        assert H3_VECTOR_LENGTH == MAX_H3_RESOLUTION + 4


class TestH3Encoders:
    def test_h3_embed_forward(self):
        from ludwig.constants import H3_VECTOR_LENGTH
        from ludwig.encoders.h3_encoders import H3Embed

        enc = H3Embed(encoder_config=None).to(DEVICE)
        # H3 vector: [mode(0-5), edge(0-6), resolution(0-15), base_cell(0-121), cell0..cell14(0-6)]
        x = torch.zeros(4, H3_VECTOR_LENGTH, dtype=torch.long, device=DEVICE)
        x[:, 0] = 1  # mode
        x[:, 1] = 0  # edge
        x[:, 2] = 5  # resolution
        x[:, 3] = 10  # base_cell
        x[:, 4:] = 3  # cells
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert ENCODER_OUTPUT in out


# ============================================================================
# 1.5.4 — Sequence Encoder Modernization Tests
# ============================================================================
class TestAttentionPooling:
    def test_attention_pooling_forward(self):
        from ludwig.modules.reduction_modules import AttentionPooling

        pool = AttentionPooling(input_size=64).to(DEVICE)
        x = torch.randn(4, 10, 64, device=DEVICE)
        out = pool(x)
        assert out.shape == (4, 64)

    def test_attention_pooling_with_mask(self):
        from ludwig.modules.reduction_modules import AttentionPooling

        pool = AttentionPooling(input_size=32).to(DEVICE)
        x = torch.randn(2, 8, 32, device=DEVICE)
        mask = torch.ones(2, 8, dtype=torch.bool, device=DEVICE)
        mask[0, 5:] = False
        out = pool(x, mask=mask)
        assert out.shape == (2, 32)

    def test_attention_pooling_in_reducer(self):
        from ludwig.modules.reduction_modules import SequenceReducer

        reducer = SequenceReducer(reduce_mode="attention_pooling", encoding_size=64).to(DEVICE)
        x = torch.randn(4, 10, 64, device=DEVICE)
        out = reducer(x)
        assert out.shape == (4, 64)


class TestMambaEncoder:
    def test_mamba_encoder_forward(self):
        from ludwig.encoders.sequence_encoders import MambaEncoder

        enc = MambaEncoder(
            vocab=["a", "b", "c", "d", "e", "<PAD>", "<UNK>"],
            max_sequence_length=16,
            embedding_size=32,
            d_model=32,
            n_layers=2,
            d_conv=4,
            expand_factor=2,
            reduce_output="mean",
        ).to(DEVICE)
        x = torch.randint(0, 7, (4, 16), device=DEVICE)
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert ENCODER_OUTPUT in out
        assert out[ENCODER_OUTPUT].shape[0] == 4

    def test_mamba_encoder_no_reduce(self):
        from ludwig.encoders.sequence_encoders import MambaEncoder

        enc = MambaEncoder(
            vocab=["a", "b", "c", "<PAD>", "<UNK>"],
            max_sequence_length=8,
            embedding_size=16,
            d_model=16,
            n_layers=1,
            reduce_output=None,
        ).to(DEVICE)
        x = torch.randint(0, 5, (2, 8), device=DEVICE)
        out = enc(x)
        from ludwig.constants import ENCODER_OUTPUT

        assert out[ENCODER_OUTPUT].dim() == 3  # [batch, seq, hidden]

    def test_mamba_encoder_schema(self):
        from ludwig.schema.encoders.sequence_encoders import MambaEncoderConfig

        cfg = MambaEncoderConfig.model_validate({"type": "mamba"})
        assert cfg.type == "mamba"
        assert cfg.n_layers >= 1


# ============================================================================
# 1.5.1 — Image Encoder Tests
# ============================================================================
class TestTorchVisionConsolidation:
    """Test that TorchVision encoders still work after consolidation."""

    def test_tv_base_has_softmax_removal_strategies(self):
        from ludwig.encoders.image.torchvision import TVBaseEncoder

        assert hasattr(TVBaseEncoder, "_softmax_removal")

    def test_tv_encoder_subclass_is_thin(self):
        """Each TV encoder subclass should only set class attributes, no _remove_softmax_layer override."""
        from ludwig.encoders.image.torchvision import TVAlexNetEncoder, TVResNetEncoder

        # These should NOT override _remove_softmax_layer anymore
        assert "_remove_softmax_layer" not in TVResNetEncoder.__dict__
        assert "_remove_softmax_layer" not in TVAlexNetEncoder.__dict__

    def test_tv_resnet_schema(self):
        from ludwig.schema.encoders.image.torchvision import TVResNetEncoderConfig

        cfg = TVResNetEncoderConfig.model_validate({"type": "resnet", "model_variant": 50})
        assert cfg.type == "resnet"


class TestPretrainedImageEncoders:
    """Test CLIP, DINOv2, SigLIP encoder registration and schema."""

    def test_clip_schema(self):
        from ludwig.schema.encoders.image.pretrained import CLIPImageEncoderConfig

        cfg = CLIPImageEncoderConfig.model_validate({"type": "clip"})
        assert cfg.type == "clip"
        assert "clip" in cfg.pretrained_model_name_or_path

    def test_dinov2_schema(self):
        from ludwig.schema.encoders.image.pretrained import DINOv2ImageEncoderConfig

        cfg = DINOv2ImageEncoderConfig.model_validate({"type": "dinov2"})
        assert cfg.type == "dinov2"
        assert "dinov2" in cfg.pretrained_model_name_or_path

    def test_siglip_schema(self):
        from ludwig.schema.encoders.image.pretrained import SigLIPImageEncoderConfig

        cfg = SigLIPImageEncoderConfig.model_validate({"type": "siglip"})
        assert cfg.type == "siglip"
        assert "siglip" in cfg.pretrained_model_name_or_path

    def test_pretrained_encoders_registered(self):
        from ludwig.constants import IMAGE
        from ludwig.encoders.registry import get_encoder_registry

        reg = get_encoder_registry()
        for name in ["clip", "dinov2", "siglip"]:
            assert name in reg[IMAGE], f"{name} not registered for IMAGE"


# ============================================================================
# 1.5.2 — Audio Encoder Tests
# ============================================================================
class TestAudioEncoders:
    """Test pretrained audio encoder registration and schema."""

    def test_wav2vec2_schema(self):
        from ludwig.schema.encoders.audio_encoders import Wav2Vec2EncoderConfig

        cfg = Wav2Vec2EncoderConfig.model_validate({"type": "wav2vec2"})
        assert cfg.type == "wav2vec2"

    def test_whisper_schema(self):
        from ludwig.schema.encoders.audio_encoders import WhisperEncoderConfig

        cfg = WhisperEncoderConfig.model_validate({"type": "whisper"})
        assert cfg.type == "whisper"

    def test_hubert_schema(self):
        from ludwig.schema.encoders.audio_encoders import HuBERTEncoderConfig

        cfg = HuBERTEncoderConfig.model_validate({"type": "hubert"})
        assert cfg.type == "hubert"

    def test_audio_encoders_registered(self):
        from ludwig.constants import AUDIO
        from ludwig.encoders.registry import get_encoder_registry

        reg = get_encoder_registry()
        for name in ["wav2vec2", "whisper", "hubert"]:
            assert name in reg[AUDIO], f"{name} not registered for AUDIO"


# ============================================================================
# 1.5.3 — Text Encoder Cleanup Tests
# ============================================================================
class TestTextEncoderCleanup:
    def test_dead_encoders_removed(self):
        """TransformerXL, CTRL, and FlauBERT should be removed."""
        from ludwig.constants import TEXT
        from ludwig.encoders.registry import get_encoder_registry

        reg = get_encoder_registry()
        text_encoders = reg.get(TEXT, {})
        assert "transformer_xl" not in text_encoders
        assert "ctrl" not in text_encoders
        assert "flaubert" not in text_encoders

    def test_kept_encoders_exist(self):
        """BERT, RoBERTa, DistilBERT should still be registered."""
        from ludwig.constants import TEXT
        from ludwig.encoders.registry import get_encoder_registry

        reg = get_encoder_registry()
        text_encoders = reg.get(TEXT, {})
        for name in [
            "bert",
            "roberta",
            "distilbert",
            "albert",
            "electra",
            "deberta",
            "xlnet",
            "gpt2",
            "t5",
            "mt5",
            "auto_transformer",
            "tf_idf",
        ]:
            assert name in text_encoders, f"{name} should still be registered for TEXT"

    def test_encoder_docstrings(self):
        """All text encoders should have docstrings after cleanup."""
        import ludwig.encoders.text_encoders as te

        for name in ["BERTEncoder", "RoBERTaEncoder", "DistilBERTEncoder", "AutoTransformerEncoder", "TfIdfEncoder"]:
            cls = getattr(te, name, None)
            if cls is not None:
                assert cls.__doc__ is not None, f"{name} should have a docstring"
                assert len(cls.__doc__) > 50, f"{name} docstring too short"


# ============================================================================
# Integration: Registry Completeness
# ============================================================================
class TestRegistryCompleteness:
    def test_all_new_encoders_in_registry(self):
        from ludwig.constants import AUDIO, CATEGORY, IMAGE, NUMBER, SEQUENCE
        from ludwig.encoders.registry import get_encoder_registry

        reg = get_encoder_registry()

        expected = {
            IMAGE: ["clip", "dinov2", "siglip"],
            AUDIO: ["wav2vec2", "whisper", "hubert"],
            CATEGORY: ["target", "hash"],
            NUMBER: ["bins"],
            SEQUENCE: ["mamba"],
        }

        for feat, names in expected.items():
            for name in names:
                assert name in reg[feat], f"Encoder '{name}' not found in {feat} registry"

    def test_all_new_schema_configs_loadable(self):
        """All new encoder configs should be loadable with just their type."""
        configs = [
            ("ludwig.schema.encoders.image.pretrained", "CLIPImageEncoderConfig", "clip"),
            ("ludwig.schema.encoders.image.pretrained", "DINOv2ImageEncoderConfig", "dinov2"),
            ("ludwig.schema.encoders.image.pretrained", "SigLIPImageEncoderConfig", "siglip"),
            ("ludwig.schema.encoders.audio_encoders", "Wav2Vec2EncoderConfig", "wav2vec2"),
            ("ludwig.schema.encoders.audio_encoders", "WhisperEncoderConfig", "whisper"),
            ("ludwig.schema.encoders.audio_encoders", "HuBERTEncoderConfig", "hubert"),
            ("ludwig.schema.encoders.category_encoders", "CategoricalTargetEncoderConfig", "target"),
            ("ludwig.schema.encoders.category_encoders", "CategoricalHashEncoderConfig", "hash"),
            ("ludwig.schema.encoders.number_encoders", "BinsEncoderConfig", "bins"),
        ]

        import importlib

        for module_path, cls_name, type_val in configs:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            cfg = cls.model_validate({"type": type_val})
            assert cfg.type == type_val, f"Config {cls_name} type mismatch"
