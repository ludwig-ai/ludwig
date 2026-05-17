# Adding a New Feature Type to Ludwig

This guide walks through every file you need to touch when adding a brand-new feature type (e.g. a hypothetical `"widget"` type). Use `ludwig/features/binary_feature.py` and `ludwig/schema/features/binary_feature.py` as living reference implementations — they are among the simplest complete examples.

______________________________________________________________________

## Conceptual overview

Each feature type lives in two parallel places:

| Layer              | Location                                   | Purpose                                                                       |
| ------------------ | ------------------------------------------ | ----------------------------------------------------------------------------- |
| **Schema**         | `ludwig/schema/features/<type>_feature.py` | Pydantic-backed config classes; declares hyperparameters and their defaults   |
| **Feature module** | `ludwig/features/<type>_feature.py`        | PyTorch modules; implements preprocessing, encoding, decoding, postprocessing |

The schema classes are used for config validation and serialization. The feature module classes are instantiated at model-build time using those configs. Neither layer knows the other exists at import time — they are wired together through the feature registry.

______________________________________________________________________

## Step 1 ��� Define the constant

Add the type string to `ludwig/constants.py`:

```python
WIDGET = "widget"
```

______________________________________________________________________

## Step 2 — Write the schema file

Create `ludwig/schema/features/widget_feature.py`. The minimal required structure is:

```python
from ludwig.constants import WIDGET, MODEL_ECD
from ludwig.schema import utils as schema_utils
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField
from ludwig.schema.features.utils import (
    ecd_defaults_config_registry,
    ecd_input_config_registry,
    ecd_output_config_registry,
    input_mixin_registry,
    output_mixin_registry,
)
from ludwig.schema.utils import LudwigBaseConfig


@input_mixin_registry.register(WIDGET)
class WidgetInputFeatureConfigMixin(LudwigBaseConfig):
    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=WIDGET)


class WidgetInputFeatureConfig(WidgetInputFeatureConfigMixin, BaseInputFeatureConfig):
    type: str = schema_utils.ProtectedString(WIDGET)
    encoder: BaseEncoderConfig = None


@ecd_input_config_registry.register(WIDGET)
class ECDWidgetInputFeatureConfig(WidgetInputFeatureConfig):
    encoder: BaseEncoderConfig = EncoderDataclassField(
        MODEL_ECD,
        feature_type=WIDGET,
        default="dense",  # default encoder for this type
    )


# For output features only:
@output_mixin_registry.register(WIDGET)
class WidgetOutputFeatureConfigMixin(LudwigBaseConfig):
    # add loss, calibration, etc. fields here
    pass


class WidgetOutputFeatureConfig(WidgetOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    type: str = schema_utils.ProtectedString(WIDGET)
    default_validation_metric: str = "some_metric"


@ecd_output_config_registry.register(WIDGET)
class ECDWidgetOutputFeatureConfig(WidgetOutputFeatureConfig):
    pass
```

**Key rules:**

- `type` must be a `ProtectedString` with your constant — this prevents accidental overwrite via user YAML.
- `@input_mixin_registry.register` / `@output_mixin_registry.register` make the preprocessing config available to `global_defaults` in Ludwig configs.
- `@ecd_input_config_registry.register` / `@ecd_output_config_registry.register` wire the schema into the ECD model config builder.

______________________________________________________________________

## Step 3 — Write the preprocessing config

Create `ludwig/schema/features/preprocessing/widget_feature_preprocessing.py` if your feature needs non-default preprocessing parameters, or register your type against an existing one (e.g. `number_feature` for scalars). For a new type, create the file:

```python
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import register_preprocessor
from ludwig.constants import WIDGET


@register_preprocessor(WIDGET)
class WidgetPreprocessingConfig(BasePreprocessingConfig):
    # add preprocessing hyperparameters here
    pass
```

______________________________________________________________________

## Step 4 — Write the feature module

Create `ludwig/features/widget_feature.py`. The required classes are:

### Inner preprocessing module

```python
import torch
from ludwig.features.base_feature import BaseFeatureMixin, InputFeature, OutputFeature


class _WidgetPreprocessing(torch.nn.Module):
    """Runs inside the model graph during inference to preprocess raw input."""

    def __init__(self, metadata: dict, preprocessing_config, is_input_feature: bool = True):
        super().__init__()
        # store everything needed to preprocess at inference time

    def forward(self, v):
        # v is the raw column value; return a tensor
        raise NotImplementedError
```

### FeatureMixin (shared preprocessing logic)

`BaseFeatureMixin` (formerly `BaseFeatureMixin`) provides the Python-side preprocessing used during dataset preparation (not inside the model graph). You must implement `add_feature_data` and `get_preprocessing_module`:

```python
class WidgetFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return WIDGET

    @staticmethod
    def cast_column(column, backend):
        """Cast the raw DataFrame column to the expected dtype."""
        return column

    @staticmethod
    def add_feature_data(
        feature_config,
        input_df,
        proc_df,
        metadata,
        preprocessing_parameters,
        backend,
        skip_save_processed_input,
    ):
        """Populate proc_df[feature_config[PROC_COLUMN]] with preprocessed values."""
        proc_df[feature_config[PROC_COLUMN]] = input_df[feature_config[COLUMN]].values
        return proc_df

    @staticmethod
    def fill_missing_values(feature_config, input_df, backend):
        """Replace NaN/None with a fill value appropriate for this type."""
        return input_df

    @staticmethod
    def feature_meta(column, preprocessing_parameters, backend):
        """Compute and return the training-set-level metadata dict for this feature."""
        return {}

    @staticmethod
    def get_preprocessing_module(feature_config, metadata):
        """Return the _WidgetPreprocessing module for use during inference."""
        return _WidgetPreprocessing(metadata, feature_config.preprocessing)
```

### InputFeature class

```python
from ludwig.schema.features.widget_feature import WidgetInputFeatureConfig


class WidgetInputFeature(WidgetFeatureMixin, InputFeature):
    def __init__(self, input_feature_config: WidgetInputFeatureConfig, encoder_obj=None, **kwargs):
        super().__init__(input_feature_config, **kwargs)
        self._input_shape = torch.Size([1])  # set to actual encoded shape

        if encoder_obj:
            self.encoder_obj = encoder_obj
        else:
            self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)

    def forward(self, inputs, mask=None):
        assert inputs.dtype == torch.float32
        encoder_output = self.encoder_obj(inputs, mask=mask)
        return {"encoder_output": encoder_output}

    @property
    def input_dtype(self):
        return torch.float32

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self.encoder_obj.output_shape

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        pass

    @staticmethod
    def create_sample_input(batch_size=2):
        return torch.zeros(batch_size, 1)

    @staticmethod
    def get_schema_cls():
        return WidgetInputFeatureConfig
```

### OutputFeature class (only if this type can be a target)

```python
from ludwig.schema.features.widget_feature import WidgetOutputFeatureConfig


class WidgetOutputFeature(WidgetFeatureMixin, OutputFeature):
    def __init__(self, output_feature_config: WidgetOutputFeatureConfig, output_features: dict, **kwargs):
        super().__init__(output_feature_config, output_features, **kwargs)
        self._input_shape = torch.Size([output_feature_config.input_size])
        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, target=None):
        return self.decoder_obj(inputs)

    def create_predict_module(self):
        return _WidgetPredict()  # see PredictModule below

    def get_prediction_set(self):
        return {LOGITS, PREDICTIONS, PROBABILITIES}

    @classmethod
    def update_config_with_metadata(cls, feature_config, feature_metadata, *args, **kwargs):
        feature_config.input_size = feature_metadata["input_size"]

    @staticmethod
    def get_schema_cls():
        return WidgetOutputFeatureConfig
```

### PredictModule (for output features)

```python
from ludwig.features.base_feature import PredictModule


class _WidgetPredict(PredictModule):
    def forward(self, inputs, feature_name):
        logits = inputs[f"{feature_name}_{LOGITS}"]
        predictions = (logits > 0.5).float()
        return {PREDICTIONS: predictions, LOGITS: logits}
```

______________________________________________________________________

## Step 5 — Register in the feature registries

Open `ludwig/features/feature_registries.py` and add your classes to all relevant registry functions:

```python
# at the top — add import
from ludwig.features.widget_feature import WidgetFeatureMixin, WidgetInputFeature

# in get_base_type_registry(), inside the returned dict:
#     WIDGET: WidgetFeatureMixin,
#
# in get_input_type_registry(), inside the returned dict:
#     WIDGET: WidgetInputFeature,
#
# in get_output_type_registry() if applicable, inside the returned dict:
#     WIDGET: WidgetOutputFeature,
```

The model builder uses `get_input_type_registry()` and `get_output_type_registry()` to instantiate feature objects from config at training time.

______________________________________________________________________

## Step 6 — Register the constant in constants.py (feature sets)

If the feature appears in `FEATURE_TYPES`, `INPUT_FEATURE_TYPES`, or similar sets, add `WIDGET` there too.

______________________________________________________________________

## Step 7 — Write tests

Create `tests/ludwig/features/test_widget_feature.py`. At minimum test:

1. `WidgetFeatureMixin.add_feature_data` — correct column values written to `proc_df`
1. `_WidgetPreprocessing.forward` — correct tensor shape for a known input
1. `WidgetInputFeature.forward` — correct output keys and shapes with a random input
1. Encoder round-trip via `create_sample_input`

```python
import torch
import pytest
from tests.integration_tests.utils import generate_data, run_api_test


def test_widget_preprocessing_forward():
    meta = {}
    module = _WidgetPreprocessing(meta, preprocessing_config=None)
    out = module(torch.zeros(4))
    assert out.shape == (4, 1)
```

______________________________________________________________________

## Checklist

- [ ] `ludwig/constants.py` — add `WIDGET = "widget"`
- [ ] `ludwig/schema/features/widget_feature.py` — schema classes + registry decorators
- [ ] `ludwig/schema/features/preprocessing/` — preprocessing config class (or reuse existing)
- [ ] `ludwig/features/widget_feature.py` — preprocessing module, mixin, input/output feature classes
- [ ] `ludwig/features/feature_registries.py` — add to `get_base_type_registry`, `get_input_type_registry`, optionally `get_output_type_registry`
- [ ] `tests/ludwig/features/test_widget_feature.py` — unit tests for preprocessing and forward pass

______________________________________________________________________

## Common pitfalls

**`proc_df[PROC_COLUMN]` vs `proc_df[COLUMN]`** — always write to `PROC_COLUMN` (the internal column name), not `COLUMN` (the raw user column name). They can differ when the user renames features.

**`get_preprocessing_module` vs `add_feature_data`** — `add_feature_data` runs in Python at dataset preparation time (CPU, pandas). `get_preprocessing_module` returns a `torch.nn.Module` that runs inside the model graph at inference time. Both must produce compatible representations.

**`input_shape` vs `output_shape`** — `InputFeature.input_shape` is the shape of the *raw preprocessed* tensor going into the encoder. `InputFeature.output_shape` is the encoder's output shape that feeds into the combiner. Return `self.encoder_obj.output_shape` for the latter.

**Registry order matters** — the registry in `feature_registries.py` is read at import time. If you import your feature class before `feature_registries.py` is loaded, the registry will be empty. The correct order is always: define constants → define schema → define feature → add to registry.

**Schema `type` field** — always use `schema_utils.ProtectedString(WIDGET)` not `str = WIDGET`. The protected string raises an error if a user tries to override it in their config YAML, which prevents subtle type mismatches.
