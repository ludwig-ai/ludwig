import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

logger = logging.getLogger(__name__)

try:
    from caformer_models import caformer_s18, caformer_b36, caformer_s36, caformer_m36
    CAFORMER_AVAILABLE = True
    logger.info(" CAFormer models imported successfully")
except ImportError as e:
    logger.warning(f" CAFormer models not available: {e}")
    CAFORMER_AVAILABLE = False

class CAFormerStackedCNN(nn.Module):
    def __init__(self,
                 height: int = 224,
                 width: int = 224,
                 num_channels: int = 3,
                 output_size: int = 128,
                 custom_model: str = "caformer_s18",
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
        print(f" CAFormerStackedCNN encoder instantiated! ")
        print(f" Using CAFormer model: {custom_model} ")
        super().__init__()
        
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.output_size = output_size
        self.custom_model = custom_model
        self.use_pretrained = use_pretrained
        self.trainable = trainable
        
        logger.info(f"Initializing CAFormerStackedCNN with model: {custom_model}")
        logger.info(f"Input: {num_channels}x{height}x{width} -> Output: {output_size}")
        
        self.channel_adapter = None
        if num_channels != 3:
            self.channel_adapter = nn.Conv2d(num_channels, 3, kernel_size=1, stride=1, padding=0)
            logger.info(f"Added channel adapter: {num_channels} -> 3 channels")
        
        self.size_adapter = None
        if height != 224 or width != 224:
            self.size_adapter = nn.AdaptiveAvgPool2d((224, 224))
            logger.info(f"Added size adapter: {height}x{width} -> 224x224")
        
        self.backbone = self._load_caformer_backbone()
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
            logger.info("CAFormer backbone frozen (trainable=False)")
        
        logger.info(f"CAFormerStackedCNN initialized successfully")

    def _load_caformer_backbone(self):
        print(f" Loading CAFormer backbone: {self.custom_model} ")
        if not CAFORMER_AVAILABLE:
            raise ImportError("CAFormer models are not available")
        model_map = {
            'caformer_s18': caformer_s18,
            'caformer_s36': caformer_s36,
            'caformer_m36': caformer_m36,
            'caformer_b36': caformer_b36,
        }
        if self.custom_model not in model_map:
            raise ValueError(f"Unknown CAFormer model: {self.custom_model}")
        model = model_map[self.custom_model](
            pretrained=self.use_pretrained, 
            num_classes=1000
        )
        print(f"Successfully loaded weights for {self.custom_model}")
        logger.info(f"Loaded CAFormer backbone: {self.custom_model} (pretrained={self.use_pretrained})")
        return model

    def _get_feature_dim(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone.forward_features(dummy_input)
            feature_dim = features.shape[-1]
        
        logger.info(f"CAFormer feature dimension: {feature_dim}")
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
        
        if x.shape[2] != 224 or x.shape[3] != 224:
            if self.size_adapter is None:
                self.size_adapter = nn.AdaptiveAvgPool2d((224, 224)).to(x.device)
                logger.info(f"Created dynamic size adapter: {x.shape[2]}x{x.shape[3]} -> 224x224")
            x = self.size_adapter(x)
        
        features = self.backbone.forward_features(x)
        output = self.fc_layers(features)
        
        return {'encoder_output': output}

    @property
    def output_shape(self):
        return [self.output_size]

def create_caformer_stacked_cnn(model_name: str, **kwargs) -> CAFormerStackedCNN:
    print(f" CREATE_CAFORMER_STACKED_CNN called with model_name: {model_name} ")
    print(f"Creating CAFormer stacked CNN encoder: {model_name}")
    
    encoder = CAFormerStackedCNN(custom_model=model_name, **kwargs)
    print(f" CAFormer encoder created successfully: {type(encoder)} ")
    return encoder

def patch_ludwig_stacked_cnn():
    # No unconditional print here
    return patch_ludwig_direct()

def patch_ludwig_robust():
    # No unconditional print here
    try:
        from ludwig.encoders.registry import get_encoder_cls
        original_get_encoder_cls = get_encoder_cls
        def patched_get_encoder_cls(encoder_type):
            if encoder_type == "stacked_cnn":
                return CAFormerStackedCNN
            return original_get_encoder_cls(encoder_type)
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
                if isinstance(enc_cfg, dict) and 'custom_model' in enc_cfg:
                    custom_model = enc_cfg['custom_model']
            if custom_model and str(custom_model).startswith('caformer_'):
                print(f"DETECTED CAFormer model: {custom_model}")
                print(f"CAFormer encoder is being loaded and used.")
                caformer_encoder = create_caformer_stacked_cnn(custom_model, **kwargs)
                for attr_name in dir(caformer_encoder):
                    if not attr_name.startswith('_'):
                        setattr(self, attr_name, getattr(caformer_encoder, attr_name))
                self.forward = caformer_encoder.forward
                self.output_shape = caformer_encoder.output_shape
                return
            else:
                # No CAFormer messages for non-CAFormer models
                original_stacked_cnn_init(self, *args, **kwargs)
        Stacked2DCNN.__init__ = patched_stacked_cnn_init
        try:
            from ludwig.features.image_feature import ImageInputFeature
            original_image_feature_init = ImageInputFeature.__init__
            def patched_image_feature_init(self, *args, **kwargs):
                original_image_feature_init(self, *args, **kwargs)
            ImageInputFeature.__init__ = patched_image_feature_init
        except Exception as e:
            pass
        return True
    except Exception as e:
        logger.error(f"Failed to apply robust patch: {e}")
        return False

def patch_ludwig_direct():
    # No unconditional print here
    try:
        from ludwig.encoders.registry import get_encoder_cls
        original_get_encoder_cls = get_encoder_cls
        def patched_get_encoder_cls(encoder_type):
            if encoder_type == "stacked_cnn":
                return CAFormerStackedCNN
            return original_get_encoder_cls(encoder_type)
        import ludwig.encoders.registry
        ludwig.encoders.registry.get_encoder_cls = patched_get_encoder_cls
        from ludwig.encoders.image.base import Stacked2DCNN
        original_stacked_cnn_init = Stacked2DCNN.__init__
        def patched_stacked_cnn_init(self, *args, **kwargs):
            custom_model = kwargs.get('custom_model', 'caformer_s18')
            if custom_model.startswith('caformer_'):
                print(f"DETECTED CAFormer model: {custom_model}")
                print(f"CAFormer encoder is being loaded and used.")
                # call parent __init__ first to properly initialize the module
                original_stacked_cnn_init(self, *args, **kwargs)
                # create and assign caformer attributes
                caformer_encoder = create_caformer_stacked_cnn(custom_model, **kwargs)
                self.forward = caformer_encoder.forward
                if hasattr(caformer_encoder, 'backbone'):
                    self.backbone = caformer_encoder.backbone
                if hasattr(caformer_encoder, 'fc_layers'):
                    self.fc_layers = caformer_encoder.fc_layers
                if hasattr(caformer_encoder, 'custom_model'):
                    self.custom_model = caformer_encoder.custom_model
            else:
                # No CAFormer messages for non-CAFormer models
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
    print(f" PATCH_LUDWIG_COMPREHENSIVE function called ")
    
    patch1 = patch_ludwig_robust()
    patch2 = patch_ludwig_direct()
    patch3 = patch_ludwig_schema_validation()
    
    print(f" Patch results: robust={patch1}, direct={patch2}, schema={patch3} ")
    
    return patch1 or patch2 or patch3

def test_caformer_stacked_cnn():
    if not CAFORMER_AVAILABLE:
        print("CAFormer models not available, skipping test")
        return
    
    try:
        encoder = CAFormerStackedCNN(
            custom_model="caformer_s18",
            height=224,
            width=224,
            num_channels=3,
            output_size=128,
            use_pretrained=False
        )
        
        dummy_input = torch.randn(2, 3, 224, 224)
        output = encoder(dummy_input)
        
        print(f"  CAFormerStackedCNN test passed")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output['encoder_output'].shape}")
        
    except Exception as e:
        print(f" CAFormerStackedCNN test failed: {e}")

if __name__ == "__main__":
    test_caformer_stacked_cnn() 