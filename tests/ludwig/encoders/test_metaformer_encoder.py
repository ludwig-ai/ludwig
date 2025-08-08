import os
import pytest
import torch

from ludwig.constants import ENCODER_OUTPUT
from ludwig.utils.misc_utils import set_random_seed

# Ensure we do not attempt to download pretrained weights during tests.
os.environ.setdefault("METAFORMER_PRETRAINED", "0")

try:
    # Verify backbone registry availability early; skip if integration not present.
    from metaformer_integration.metaformer_models import default_cfgs as _mf_cfgs  # noqa: F401
    META_INTEGRATION_AVAILABLE = True
except Exception:
    META_INTEGRATION_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not META_INTEGRATION_AVAILABLE,
    reason="MetaFormer integration not available (metaformer_integration.metaformer_models import failed).",
)

def test_metaformer_encoder_basic_forward_and_shape():
    from ludwig.encoders.image.metaformer import MetaFormerEncoder

    set_random_seed(1234)

    # Use small dimensions (will be internally adapted to model expected size).
    encoder = MetaFormerEncoder(
        height=28,
        width=28,
        num_channels=1,
        model_name="caformer_s18",
        use_pretrained=False,
        trainable=True,
        output_size=64,
    )

    batch_size = 2
    x = torch.rand(batch_size, 1, 28, 28)
    out = encoder(x)
    assert ENCODER_OUTPUT in out, "Encoder output key missing."
    rep = out[ENCODER_OUTPUT]
    assert rep.shape[0] == batch_size, "Batch dimension mismatch."
    assert tuple(rep.shape[1:]) == tuple(encoder.output_shape), "Representation shape mismatch."

def test_metaformer_encoder_parameter_updates():
    from ludwig.encoders.image.metaformer import MetaFormerEncoder
    from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

    set_random_seed(5678)

    encoder = MetaFormerEncoder(
        height=32,
        width=32,
        num_channels=3,
        model_name="caformer_s18",
        use_pretrained=False,
        trainable=True,
        output_size=32,
    )

    inputs = torch.rand(2, 3, 32, 32)
    outputs = encoder(inputs)
    target = torch.randn_like(outputs[ENCODER_OUTPUT])

    fpc, tpc, upc, not_updated = check_module_parameters_updated(encoder, (inputs,), target)
    assert tpc == upc, f"Some parameters did not update: {not_updated}"
