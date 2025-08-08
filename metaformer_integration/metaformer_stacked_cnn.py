import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import sys
import os

# Ensure local package path
sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)

try:
    from metaformer_integration.metaformer_models import get_registered_model, default_cfgs
    METAFORMER_AVAILABLE = True
    logger.info(" MetaFormer family models imported successfully")
except ImportError as e:
    logger.warning(f" MetaFormer models not available: {e}")
    METAFORMER_AVAILABLE = False

META_PREFIXES = (
    "caformer_",
    "convformer_",
    "identityformer_",
    "randformer_",
    "poolformerv2_",
)

_PATCHED_LUDWIG_META = False

class MetaFormerStackedCNN(nn.Module):
    """Generic MetaFormer family encoder wrapper (ConvFormer, CAFormer, IdentityFormer, RandFormer, PoolFormerV2).
    Backward compatible alias retained for legacy but prefer MetaFormerStackedCNN.
    """
    def __init__(self,
                 height: int = 224,
                 width: int = 224,
                 num_channels: int = 3,
                 output_size: int = 128,
                 custom_model: Optional[str] = None,
                 use_pretrained: bool = True,
                 trainable: bool = True,
                 conv_layers: Optional[List[Dict]] = None,
                 num_conv_layers: Optional[int] = None,
                 conv_activation: str = "relu",
                 conv_dropout: float = 0.0,
                 conv_norm: Optional[str] = None,
                 conv_use_bias: bool = True,
                 fc_layers: Optional[List[Dict]] = None,
                 num_fc_layers: int = 1,
                 fc_activation: str = "relu", 
                 fc_dropout: float = 0.0,
                 fc_norm: Optional[str] = None,
                 fc_use_bias: bool = True,
                 **kwargs):
        print(f" MetaFormerStackedCNN encoder instantiated! ")
        print(f" Using MetaFormer model: {custom_model} ")
        super().__init__()
        
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.output_size = output_size
        self.custom_model = custom_model
        if self.custom_model is None:
            self.custom_model = sorted(default_cfgs.keys())[0]
        self.use_pretrained = use_pretrained
        self.trainable = trainable

        env_flag = os.getenv("METAFORMER_PRETRAINED")
        if env_flag is not None and env_flag.lower() in ("0", "false", "no", "off"):
            self.use_pretrained = False

        cfg_input = default_cfgs.get(self.custom_model, {}).get("input_size", (3, 224, 224))
        self.target_height, self.target_width = cfg_input[1], cfg_input[2]
        logger.info(f"Target backbone input size: {self.target_height}x{self.target_width}")
        
        logger.info(f"Initializing MetaFormerStackedCNN with model: {self.custom_model}")
        logger.info(f"Input: {num_channels}x{height}x{width} -> Output: {output_size}")
        
        self.channel_adapter = None
        if num_channels != 3:
            self.channel_adapter = nn.Conv2d(num_channels, 3, kernel_size=1, stride=1, padding=0)
            logger.info(f"Added channel adapter: {num_channels} -> 3 channels")
        
        self.size_adapter = None
        if height != self.target_height or width != self.target_width:
            self.size_adapter = nn.AdaptiveAvgPool2d((self.target_height, self.target_width))
            logger.info(f"Added size adapter: {height}x{width} -> {self.target_height}x{self.target_width}")
        
        self.backbone = self._load_metaformer_backbone()
        self.feature_dim = self._get_feature_dim()
        
        self.fc_layers = self._create_fc_layers(
            input_dim=self.feature_dim,
            output_dim=output_size,
            num_layers=num_fc_layers,
            activation=fc_activation,
            dropout=fc_dropout,
            norm=fc_norm,
            use_bias=fc_use_bias,
            fc_layers_config=fc_layers
        )
        
        if not trainable:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("MetaFormer backbone frozen (trainable=False)")
        
        logger.info(f"MetaFormerStackedCNN initialized successfully")

    def _load_metaformer_backbone(self):
        print(f" Loading MetaFormer backbone: {self.custom_model} ")
        if not METAFORMER_AVAILABLE:
            raise ImportError("MetaFormer models are not available")
        if self.custom_model not in default_cfgs:
            raise ValueError(f"Unknown MetaFormer model: {self.custom_model}. Available: {list(default_cfgs.keys())[:10]} ...")
        model = get_registered_model(self.custom_model, pretrained=self.use_pretrained)
        print(f"Successfully loaded weights (if requested) for {self.custom_model}")
        logger.info(f"Loaded MetaFormer backbone: {self.custom_model} (pretrained={self.use_pretrained})")
        return model

    def _get_feature_dim(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.target_height, self.target_width)
            features = self.backbone.forward_features(dummy_input)
            feature_dim = features.shape[-1]
        logger.info(f"MetaFormer feature dimension: {feature_dim}")
        return feature_dim

    def _create_fc_layers(self, input_dim, output_dim, num_layers, activation, dropout, norm, use_bias, fc_layers_config):
        layers = []
        if fc_layers_config:
            current_dim = input_dim
            for i, layer_config in enumerate(fc_layers_config):
                layer_output_dim = layer_config.get('output_size', output_dim if i == len(fc_layers_config) - 1 else current_dim)
                layers.append(nn.Linear(current_dim, layer_output_dim, bias=use_bias))
                if i < len(fc_layers_config) - 1:
                    if activation == "relu":
                        layers.append(nn.ReLU())
                    elif activation == "tanh":
                        layers.append(nn.Tanh())
                    elif activation == "sigmoid":
                        layers.append(nn.Sigmoid())
                    elif activation == "leaky_relu":
                        layers.append(nn.LeakyReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if norm == "batch":
                    layers.append(nn.BatchNorm1d(layer_output_dim))
                elif norm == "layer":
                    layers.append(nn.LayerNorm(layer_output_dim))
                current_dim = layer_output_dim
        else:
            if num_layers == 1:
                layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            else:
                intermediate_dims = [input_dim]
                for i in range(num_layers - 1):
                    intermediate_dim = int(input_dim * (0.5 ** (i + 1)))
                    intermediate_dim = max(intermediate_dim, output_dim)
                    intermediate_dims.append(intermediate_dim)
                intermediate_dims.append(output_dim)
                for i in range(num_layers):
                    layers.append(nn.Linear(intermediate_dims[i], intermediate_dims[i+1], bias=use_bias))
                    if i < num_layers - 1:
                        if activation == "relu":
                            layers.append(nn.ReLU())
                        elif activation == "tanh":
                            layers.append(nn.Tanh())
                        elif activation == "sigmoid":
                            layers.append(nn.Sigmoid())
                        elif activation == "leaky_relu":
                            layers.append(nn.LeakyReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    if norm == "batch":
                        layers.append(nn.BatchNorm1d(intermediate_dims[i+1]))
                    elif norm == "layer":
                        layers.append(nn.LayerNorm(intermediate_dims[i+1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.shape[1] != 3:
            if self.channel_adapter is None:
                self.channel_adapter = nn.Conv2d(x.shape[1], 3, kernel_size=1, stride=1, padding=0).to(x.device)
                logger.info(f"Created dynamic channel adapter: {x.shape[1]} -> 3 channels")
            x = self.channel_adapter(x)
        if x.shape[2] != self.target_height or x.shape[3] != self.target_width:
            if self.size_adapter is None:
                self.size_adapter = nn.AdaptiveAvgPool2d((self.target_height, self.target_width)).to(x.device)
                logger.info(f"Created dynamic size adapter: {x.shape[2]}x{x.shape[3]} -> {self.target_height}x{self.target_width}")
            x = self.size_adapter(x)
        features = self.backbone.forward_features(x)
        output = self.fc_layers(features)
        return {'encoder_output': output}

    @property
    def output_shape(self):
        return [self.output_size]

    # Minimal schema hook so Ludwig's schema-based encoder initialization does not fail.
    # Ludwig expects: encoder_cls.get_schema_cls().Schema() to return a marshmallow schema instance.
    # We provide a lightweight stand-in that simply echoes the config (no validation).
    @classmethod
    def get_schema_cls(cls):
        class _MetaFormerStackedCNNSchemaContainer:
            class Schema:
                def dump(self, obj, *args, **kwargs):
                    # Return a plain dict representation (passthrough)
                    if hasattr(obj, "to_dict"):
                        return obj.to_dict()
                    if isinstance(obj, dict):
                        return obj
                    return {}
                def load(self, data, **kwargs):
                    # Passthrough load (no validation)
                    if hasattr(data, "to_dict"):
                        return data.to_dict()
                    return data
        return _MetaFormerStackedCNNSchemaContainer

# Legacy alias (retain but deprecated)
CAFormerStackedCNN = MetaFormerStackedCNN

def create_metaformer_stacked_cnn(model_name: str, **kwargs) -> MetaFormerStackedCNN:
    print(f" CREATE_METAFORMER_STACKED_CNN called with model_name: {model_name} ")
    print(f"Creating MetaFormer stacked CNN encoder: {model_name}")
    if 'custom_model' in kwargs:
        kwargs.pop('custom_model', None)
    encoder = MetaFormerStackedCNN(custom_model=model_name, **kwargs)
    print(f" MetaFormer encoder created successfully: {type(encoder)} ")
    return encoder

def create_caformer_stacked_cnn(model_name: str, **kwargs):
    # Backward compatibility
    return create_metaformer_stacked_cnn(model_name, **kwargs)

def list_metaformer_models():
    return sorted(default_cfgs.keys())

def get_metaformer_backbone_names(prefix: Optional[str] = None):
    names = list_metaformer_models()
    if prefix:
        names = [n for n in names if n.startswith(prefix)]
    return names

def metaformer_model_exists(name: str) -> bool:
    return name in default_cfgs

def describe_metaformer_model(name: str) -> Dict[str, Any]:
    cfg = default_cfgs.get(name, {}).copy()
    cfg["exists"] = name in default_cfgs
    return cfg

def patch_ludwig_stacked_cnn():
    return patch_ludwig_direct()

def patch_ludwig_robust():
    try:
        from ludwig.encoders.registry import get_encoder_cls
        original_get_encoder_cls = get_encoder_cls
        def patched_get_encoder_cls(*args, **kwargs):
            # Support both legacy signature get_encoder_cls(encoder_type)
            # and current signature get_encoder_cls(feature_type, encoder_type).
            if args:
                encoder_type = args[-1]
            else:
                encoder_type = kwargs.get("encoder_type")
            if encoder_type == "stacked_cnn":
                return MetaFormerStackedCNN
            return original_get_encoder_cls(*args, **kwargs)
        import ludwig.encoders.registry
        ludwig.encoders.registry.get_encoder_cls = patched_get_encoder_cls
        from ludwig.encoders.image.base import Stacked2DCNN
        original_stacked_cnn_init = Stacked2DCNN.__init__
        def patched_stacked_cnn_init(self, *args, **kwargs):
            custom_model = None
            if 'custom_model' in kwargs:
                custom_model = kwargs['custom_model']
            elif 'encoder_config' in kwargs:
                enc_cfg = kwargs['encoder_config']
                if hasattr(enc_cfg, 'to_dict'):
                    enc_cfg = enc_cfg.to_dict()
                if isinstance(enc_cfg, dict):
                    custom_model = enc_cfg.get('custom_model', None)
            if custom_model and any(str(custom_model).startswith(p) for p in META_PREFIXES):
                original_stacked_cnn_init(self, *args, **kwargs)
                print(f"DETECTED MetaFormer model: {custom_model}")
                print(f"MetaFormer encoder is being loaded and used (robust patch).")
                build_kwargs = dict(kwargs)
                build_kwargs.pop('custom_model', None)
                meta_encoder = create_metaformer_stacked_cnn(custom_model, **build_kwargs)
                if hasattr(meta_encoder, 'backbone'):
                    self.backbone = meta_encoder.backbone
                if hasattr(meta_encoder, 'fc_layers'):
                    self.fc_layers = meta_encoder.fc_layers
                if hasattr(meta_encoder, 'feature_dim'):
                    self.feature_dim = meta_encoder.feature_dim
                if hasattr(meta_encoder, 'output_size'):
                    self.output_size = meta_encoder.output_size
                self.forward = meta_encoder.forward
                if hasattr(meta_encoder, 'output_shape'):
                    shape_val = meta_encoder.output_shape
                    if isinstance(shape_val, torch.Size):
                        self._output_shape_override = shape_val
                    elif isinstance(shape_val, (list, tuple)):
                        self._output_shape_override = torch.Size(shape_val)
                return
            original_stacked_cnn_init(self, *args, **kwargs)
        Stacked2DCNN.__init__ = patched_stacked_cnn_init
        try:
            from ludwig.features.image_feature import ImageInputFeature
            original_image_feature_init = ImageInputFeature.__init__
            def patched_image_feature_init(self, *args, **kwargs):
                # Call original init
                original_image_feature_init(self, *args, **kwargs)
                # If ImageInputFeature lacks an input_shape attribute (property without setter)
                # we cannot assign it directly. Instead, override create_sample_input used by the
                # batch size tuner to synthesize a random tensor of appropriate shape.
                if not hasattr(self, "input_shape"):
                    ch = getattr(getattr(self, "encoder_obj", None), "num_channels", 3)
                    h = getattr(getattr(self, "encoder_obj", None), "height", 224)
                    w = getattr(getattr(self, "encoder_obj", None), "width", 224)
                    def _mf_create_sample_input(batch_size=2, sequence_length=None):
                        import torch
                        return torch.rand([batch_size, ch, h, w])
                    # Monkey patch only if framework did not already supply one
                    self.create_sample_input = _mf_create_sample_input
                    try:
                        logger.info(f"[MetaFormer Patch] Injected fallback create_sample_input with shape=({ch},{h},{w})")
                    except Exception:
                        pass
            ImageInputFeature.__init__ = patched_image_feature_init
        except Exception:
            pass
        return True
    except Exception as e:
        logger.error(f"Failed to apply robust patch: {e}")
        return False

def patch_ludwig_direct():
    try:
        from ludwig.encoders.registry import get_encoder_cls
        original_get_encoder_cls = get_encoder_cls
        def patched_get_encoder_cls(*args, **kwargs):
            if args:
                encoder_type = args[-1]
            else:
                encoder_type = kwargs.get("encoder_type")
            if encoder_type == "stacked_cnn":
                return MetaFormerStackedCNN
            return original_get_encoder_cls(*args, **kwargs)
        import ludwig.encoders.registry
        ludwig.encoders.registry.get_encoder_cls = patched_get_encoder_cls
        from ludwig.encoders.image.base import Stacked2DCNN
        original_stacked_cnn_init = Stacked2DCNN.__init__
        def patched_stacked_cnn_init(self, *args, **kwargs):
            custom_model = kwargs.get('custom_model', None)
            if custom_model is None:
                custom_model = sorted(default_cfgs.keys())[0]
            if any(custom_model.startswith(p) for p in META_PREFIXES):
                print(f"DETECTED MetaFormer model: {custom_model}")
                print(f"MetaFormer encoder is being loaded and used.")
                original_stacked_cnn_init(self, *args, **kwargs)
                meta_encoder = create_metaformer_stacked_cnn(custom_model, **kwargs)
                self.forward = meta_encoder.forward
                if hasattr(meta_encoder, 'backbone'):
                    self.backbone = meta_encoder.backbone
                if hasattr(meta_encoder, 'fc_layers'):
                    self.fc_layers = meta_encoder.fc_layers
                if hasattr(meta_encoder, 'custom_model'):
                    self.custom_model = meta_encoder.custom_model
            else:
                original_stacked_cnn_init(self, *args, **kwargs)
        Stacked2DCNN.__init__ = patched_stacked_cnn_init
        return True
    except Exception as e:
        logger.error(f"Failed to apply direct patch: {e}")
        return False

def patch_ludwig_schema_validation():
    print(f" PATCH_LUDWIG_SCHEMA_VALIDATION function called ")
    try:
        from ludwig.schema.features.image import ImageInputFeatureConfig
        original_validate = ImageInputFeatureConfig.validate
        def patched_validate(self, data, **kwargs):
            print(f" PATCHED SCHEMA VALIDATION called ")
            print(f"  data: {data}")
            if 'encoder' in data and 'custom_model' in data['encoder']:
                custom_model = data['encoder']['custom_model']
                print(f" DETECTED custom_model in schema validation: {custom_model} ")
            return original_validate(self, data, **kwargs)
        ImageInputFeatureConfig.validate = patched_validate
        print(f" Successfully patched schema validation ")
        return True
    except Exception as e:
        print(f" Could not patch schema validation: {e} ")
        return False

def patch_ludwig_comprehensive():
    global _PATCHED_LUDWIG_META
    if _PATCHED_LUDWIG_META:
        print(" PATCH_LUDWIG_COMPREHENSIVE already applied (skipping)")
        return True
    print(" PATCH_LUDWIG_COMPREHENSIVE function called ")
    patch_robust = patch_ludwig_robust()
    patch_schema = patch_ludwig_schema_validation()
    _PATCHED_LUDWIG_META = patch_robust or patch_schema
    print(f" Patch results: robust={patch_robust}, schema={patch_schema} ")
    return _PATCHED_LUDWIG_META

def _quick_metaformer_test():
    if not METAFORMER_AVAILABLE:
        print("MetaFormer models not available, skipping test")
        return
    try:
        encoder = MetaFormerStackedCNN(
            custom_model=sorted(default_cfgs.keys())[0],
            height=224,
            width=224,
            num_channels=3,
            output_size=128,
            use_pretrained=False,
            trainable=False,
        )
        dummy = torch.randn(1, 3, 224, 224)
        out = encoder(dummy)
        print("MetaFormer quick test OK:", out['encoder_output'].shape)
    except Exception as e:
        print(f"MetaFormer quick test failed: {e}")

if __name__ == "__main__":
    _quick_metaformer_test()
