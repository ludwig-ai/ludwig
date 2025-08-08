# MetaFormer unified models module (migrated from caformer_setup_backup/caformer_models.py)
# Provides IdentityFormer, RandFormer, PoolFormerV2, ConvFormer, CAFormer minimal builders.

from functools import partial
import math
from typing import Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
_model_registry: Dict[str, Callable] = {}

def register_model(fn):
    _model_registry[fn.__name__] = fn
    return fn

def get_registered_model(name: str, pretrained=False, **kwargs):
    if name not in _model_registry:
        raise ValueError(f"Model '{name}' not found. Available: {list(_model_registry.keys())}")
    return _model_registry[name](pretrained=pretrained, **kwargs)

def _cfg(url: str = "", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }

default_cfgs = {
    # IdentityFormer
    "identityformer_s12": _cfg(url="https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s12.pth"),
    "identityformer_s24": _cfg(url="https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s24.pth"),
    "identityformer_s36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s36.pth"),
    "identityformer_m36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m36.pth"),
    "identityformer_m48": _cfg(url="https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m48.pth"),
    # RandFormer
    "randformer_s12": _cfg(url="https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s12.pth"),
    "randformer_s24": _cfg(url="https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s24.pth"),
    "randformer_s36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s36.pth"),
    "randformer_m36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m36.pth"),
    "randformer_m48": _cfg(url="https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m48.pth"),
    # PoolFormerV2
    "poolformerv2_s12": _cfg(url="https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s12.pth"),
    "poolformerv2_s24": _cfg(url="https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s24.pth"),
    "poolformerv2_s36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s36.pth"),
    "poolformerv2_m36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m36.pth"),
    "poolformerv2_m48": _cfg(url="https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m48.pth"),
    # ConvFormer
    "convformer_s18": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth"),
    "convformer_s18_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384.pth", input_size=(3, 384, 384)),
    "convformer_s18_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21ft1k.pth"),
    "convformer_s18_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "convformer_s18_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21k.pth", num_classes=21841),
    "convformer_s36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36.pth"),
    "convformer_s36_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384.pth", input_size=(3, 384, 384)),
    "convformer_s36_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21ft1k.pth"),
    "convformer_s36_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "convformer_s36_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21k.pth", num_classes=21841),
    "convformer_m36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36.pth"),
    "convformer_m36_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384.pth", input_size=(3, 384, 384)),
    "convformer_m36_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21ft1k.pth"),
    "convformer_m36_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "convformer_m36_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21k.pth", num_classes=21841),
    "convformer_b36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36.pth"),
    "convformer_b36_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384.pth", input_size=(3, 384, 384)),
    "convformer_b36_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21ft1k.pth"),
    "convformer_b36_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "convformer_b36_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21k.pth", num_classes=21841),
    # CAFormer
    "caformer_s18": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth"),
    "caformer_s18_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384.pth", input_size=(3, 384, 384)),
    "caformer_s18_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21ft1k.pth"),
    "caformer_s18_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "caformer_s18_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21k.pth", num_classes=21841),
    "caformer_s36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth"),
    "caformer_s36_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384.pth", input_size=(3, 384, 384)),
    "caformer_s36_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21ft1k.pth"),
    "caformer_s36_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "caformer_s36_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21k.pth", num_classes=21841),
    "caformer_m36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth"),
    "caformer_m36_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384.pth", input_size=(3, 384, 384)),
    "caformer_m36_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21ft1k.pth"),
    "caformer_m36_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "caformer_m36_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21k.pth", num_classes=21841),
    "caformer_b36": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth"),
    "caformer_b36_384": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384.pth", input_size=(3, 384, 384)),
    "caformer_b36_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21ft1k.pth"),
    "caformer_b36_384_in21ft1k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384_in21ft1k.pth", input_size=(3, 384, 384)),
    "caformer_b36_in21k": _cfg(url="https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21k.pth", num_classes=21841),
}

# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------
class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)
    def forward(self, x):
        if x.dim() == 4:
            return x * self.scale.view(1, -1, 1, 1)
        return x * self.scale

class SquaredReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads if num_heads else max(1, dim // head_dim)
        self.attention_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.register_buffer("random_matrix",
                             torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1),
                             persistent=False)
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        if self.random_matrix.shape[0] != N:
            rm = torch.eye(N, device=x.device, dtype=x.dtype)
        else:
            rm = self.random_matrix
        x = x.reshape(B, N, C)
        x = torch.einsum("mn,bnc->bmc", rm, x)
        x = x.reshape(B, H, W, C)
        return x

class Pooling(nn.Module):
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x

class LayerNormGeneral(nn.Module):
    def __init__(self, affine_shape=None, normalized_dim=(-1,), scale=True, bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.eps = eps
        if scale and affine_shape is not None:
            self.weight = nn.Parameter(torch.ones(affine_shape))
        else:
            self.weight = None
        if bias and affine_shape is not None:
            self.bias = nn.Parameter(torch.zeros(affine_shape))
        else:
            self.bias = None
    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        v = (c ** 2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(v + self.eps)
        if self.use_scale and self.weight is not None:
            w = self.weight
            if x.dim() == 4 and w.dim() == 1:
                w = w.view(1, -1, 1, 1)
            x = x * w
        if self.use_bias and self.bias is not None:
            b = self.bias
            if x.dim() == 4 and b.dim() == 1:
                b = b.view(1, -1, 1, 1)
            x = x + b
        return x

class LayerNormWithoutBias(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape
    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)
            x = x.permute(0, 3, 1, 2)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)
        return x

class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2, act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7, padding=3, **kwargs):
        super().__init__()
        hidden = int(expansion_ratio * dim)
        self.pw1 = nn.Linear(dim, hidden, bias=bias)
        self.act1 = act1_layer()
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=padding, groups=hidden, bias=bias)
        self.act2 = act2_layer()
        self.pw2 = nn.Linear(hidden, dim, bias=bias)
    def forward(self, x):
        identity = x
        x = self.pw1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pw2(x)
        if x.shape == identity.shape:
            x = x + identity
        return x

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        out_features = out_features or dim
        hidden = int(mlp_ratio * dim)
        drops = to_2tuple(drop)
        self.fc1 = nn.Linear(dim, hidden, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drops[0])
        self.fc2 = nn.Linear(hidden, out_features, bias=bias)
        self.drop2 = nn.Dropout(drops[1])
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MlpHead(nn.Module):
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
                 norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden)
        self.drop = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(hidden, num_classes, bias=bias)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class MetaFormerBlock(nn.Module):
    def __init__(self, dim, token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm, drop=0.0, drop_path=0.0,
                 layer_scale_init_value=None, res_scale_init_value=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop) if token_mixer not in (nn.Identity, None) else token_mixer()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale1 = Scale(dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_scale2 = Scale(dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
    def forward(self, x):
        shortcut = x
        x1 = self.norm1(x)
        x1p = x1.permute(0, 2, 3, 1)
        tm = self.token_mixer(x1p) if not isinstance(self.token_mixer, nn.Identity) else x1p
        tm = tm.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path1(self.layer_scale1(tm))
        x = self.res_scale1(x)
        shortcut2 = x
        x2 = self.norm2(x)
        x2p = x2.permute(0, 2, 3, 1)
        mlp_out = self.mlp(x2p).permute(0, 3, 1, 2)
        x = shortcut2 + self.drop_path2(self.layer_scale2(mlp_out))
        x = self.res_scale2(x)
        return x

class MetaFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=(2, 2, 6, 2), dims=(64, 128, 320, 512),
                 token_mixers=nn.Identity, mlps=Mlp,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
                 drop_path_rate=0.0, head_dropout=0.0,
                 layer_scale_init_values=None,
                 res_scale_init_values=(None, None, 1.0, 1.0),
                 output_norm=partial(nn.LayerNorm, eps=1e-6),
                 head_fn=nn.Linear, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dims[-1]
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layers(dims[0]) if not isinstance(norm_layers, (list, tuple)) else norm_layers[0](dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            norm_mod = norm_layers if not isinstance(norm_layers, (list, tuple)) else norm_layers[i + 1]
            layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                norm_mod(dims[i + 1])
            )
            self.downsample_layers.append(layer)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * len(depths)
        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * len(depths)
        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * len(depths)
        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * len(depths)
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * len(depths)
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i, depth in enumerate(depths):
            blocks = []
            for j in range(depth):
                blocks.append(
                    MetaFormerBlock(
                        dim=dims[i],
                        token_mixer=token_mixers[i],
                        mlp=mlps[i],
                        norm_layer=norm_layers[i] if not isinstance(norm_layers[i], partial) else norm_layers[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_values[i],
                        res_scale_init_value=res_scale_init_values[i],
                    )
                )
            cur += depth
            self.stages.append(nn.Sequential(*blocks))
        self.norm = output_norm(dims[-1]) if not isinstance(output_norm, (list, tuple)) else output_norm[-1]
        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
    def forward_features(self, x):
        for i, down in enumerate(self.downsample_layers):
            x = down(x)
            x = self.stages[i](x)
        x = x.mean(dim=[2, 3])
        return x
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
@register_model
def identityformer_s12(pretrained=False, **kwargs):
    model = MetaFormer(depths=(2, 2, 6, 2), dims=(64,128,320,512),
                       token_mixers=nn.Identity,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["identityformer_s12"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def identityformer_s24(pretrained=False, **kwargs):
    model = MetaFormer(depths=(4,4,12,4), dims=(64,128,320,512),
                       token_mixers=nn.Identity,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["identityformer_s24"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def identityformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(6,6,18,6), dims=(64,128,320,512),
                       token_mixers=nn.Identity,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["identityformer_s36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def identityformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(6,6,18,6), dims=(96,192,384,768),
                       token_mixers=nn.Identity,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["identityformer_m36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def identityformer_m48(pretrained=False, **kwargs):
    model = MetaFormer(depths=(8,8,24,8), dims=(96,192,384,768),
                       token_mixers=nn.Identity,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["identityformer_m48"]
    if pretrained: _load_pretrained(model)
    return model

def _rand_token_mixers(depths, tokens_last_stage=49):
    return [nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=tokens_last_stage)]

@register_model
def randformer_s12(pretrained=False, **kwargs):
    model = MetaFormer(depths=(2,2,6,2), dims=(64,128,320,512),
                       token_mixers=_rand_token_mixers((2,2,6,2)),
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["randformer_s12"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def randformer_s24(pretrained=False, **kwargs):
    model = MetaFormer(depths=(4,4,12,4), dims=(64,128,320,512),
                       token_mixers=_rand_token_mixers((4,4,12,4)),
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["randformer_s24"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def randformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(6,6,18,6), dims=(64,128,320,512),
                       token_mixers=_rand_token_mixers((6,6,18,6)),
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["randformer_s36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def randformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(6,6,18,6), dims=(96,192,384,768),
                       token_mixers=_rand_token_mixers((6,6,18,6)),
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["randformer_m36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def randformer_m48(pretrained=False, **kwargs):
    model = MetaFormer(depths=(8,8,24,8), dims=(96,192,384,768),
                       token_mixers=_rand_token_mixers((8,8,24,8)),
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["randformer_m48"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def poolformerv2_s12(pretrained=False, **kwargs):
    model = MetaFormer(depths=(2,2,6,2), dims=(64,128,320,512),
                       token_mixers=Pooling,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["poolformerv2_s12"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def poolformerv2_s24(pretrained=False, **kwargs):
    model = MetaFormer(depths=(4,4,12,4), dims=(64,128,320,512),
                       token_mixers=Pooling,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["poolformerv2_s24"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def poolformerv2_s36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(6,6,18,6), dims=(64,128,320,512),
                       token_mixers=Pooling,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["poolformerv2_s36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def poolformerv2_m36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(6,6,18,6), dims=(96,192,384,768),
                       token_mixers=Pooling,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["poolformerv2_m36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def poolformerv2_m48(pretrained=False, **kwargs):
    model = MetaFormer(depths=(8,8,24,8), dims=(96,192,384,768),
                       token_mixers=Pooling,
                       norm_layers=partial(LayerNormGeneral, normalized_dim=(1,2,3), eps=1e-6, bias=False),
                       **kwargs)
    model.default_cfg = default_cfgs["poolformerv2_m48"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s18(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,3,9,3), dims=(64,128,320,512),
                       token_mixers=SepConv, head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["convformer_s18"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s18_384(pretrained=False, **kwargs):
    model = convformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s18_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s18_in21ft1k(pretrained=False, **kwargs):
    model = convformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s18_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s18_384_in21ft1k(pretrained=False, **kwargs):
    model = convformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s18_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s18_in21k(pretrained=False, **kwargs):
    model = convformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s18_in21k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,12,18,3), dims=(64,128,320,512),
                       token_mixers=SepConv, head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["convformer_s36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s36_384(pretrained=False, **kwargs):
    model = convformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s36_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s36_in21ft1k(pretrained=False, **kwargs):
    model = convformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s36_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s36_384_in21ft1k(pretrained=False, **kwargs):
    model = convformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s36_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_s36_in21k(pretrained=False, **kwargs):
    model = convformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_s36_in21k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,12,18,3), dims=(96,192,384,576),
                       token_mixers=SepConv, head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["convformer_m36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_m36_384(pretrained=False, **kwargs):
    model = convformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_m36_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_m36_in21ft1k(pretrained=False, **kwargs):
    model = convformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_m36_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_m36_384_in21ft1k(pretrained=False, **kwargs):
    model = convformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_m36_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_m36_in21k(pretrained=False, **kwargs):
    model = convformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_m36_in21k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_b36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,12,18,3), dims=(128,256,512,768),
                       token_mixers=SepConv, head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["convformer_b36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_b36_384(pretrained=False, **kwargs):
    model = convformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_b36_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_b36_in21ft1k(pretrained=False, **kwargs):
    model = convformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_b36_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_b36_384_in21ft1k(pretrained=False, **kwargs):
    model = convformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_b36_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def convformer_b36_in21k(pretrained=False, **kwargs):
    model = convformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["convformer_b36_in21k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s18(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,3,9,3), dims=(64,128,320,512),
                       token_mixers=[SepConv, SepConv, Attention, Attention],
                       head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["caformer_s18"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s18_384(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s18_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s18_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s18_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s18_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s18_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s18_in21k(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s18_in21k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,12,18,3), dims=(64,128,320,512),
                       token_mixers=[SepConv, SepConv, Attention, Attention],
                       head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["caformer_s36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s36_384(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s36_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s36_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s36_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s36_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s36_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_s36_in21k(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_s36_in21k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,12,18,3), dims=(96,192,384,576),
                       token_mixers=[SepConv, SepConv, Attention, Attention],
                       head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["caformer_m36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_m36_384(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_m36_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_m36_in21ft1k(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_m36_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_m36_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_m36_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model  # (If needed you can correct unused variant naming later.)

@register_model
def caformer_m36_in21k(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_m36_in21k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_b36(pretrained=False, **kwargs):
    model = MetaFormer(depths=(3,12,18,3), dims=(128,256,512,768),
                       token_mixers=[SepConv, SepConv, Attention, Attention],
                       head_fn=MlpHead, **kwargs)
    model.default_cfg = default_cfgs["caformer_b36"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_b36_384(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_b36_384"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_b36_in21ft1k(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_b36_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_b36_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_b36_384_in21ft1k"]
    if pretrained: _load_pretrained(model)
    return model

@register_model
def caformer_b36_in21k(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=False, **kwargs)
    model.default_cfg = default_cfgs["caformer_b36_in21k"]
    if pretrained: _load_pretrained(model)
    return model

def _load_pretrained(model: nn.Module):
    url = getattr(model, "default_cfg", {}).get("url", "")
    if not url:
        return
    try:
        state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[MetaFormer] Missing keys (ignored): {missing[:5]} ...")
        if unexpected:
            print(f"[MetaFormer] Unexpected keys (ignored): {unexpected[:5]} ...")
    except Exception as e:
        print(f"[MetaFormer] Failed to load pretrained weights from {url}: {e}")

def _quick_test():
    names = ["identityformer_s12", "randformer_s12", "poolformerv2_s12", "convformer_s18", "caformer_s18"]
    for n in names:
        try:
            m = get_registered_model(n, pretrained=False, num_classes=10)
            x = torch.randn(1, 3, 224, 224)
            y = m(x)
            print(f"{n}: output {y.shape}")
        except Exception as ex:
            print(f"{n}: FAILED ({ex})")

if __name__ == "__main__":
    _quick_test()
