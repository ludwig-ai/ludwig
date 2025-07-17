from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

_model_registry = {}

def register_model(fn):
    _model_registry[fn.__name__] = fn
    return fn

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'caformer_s18': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth'),
    'caformer_s18_384': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384.pth', input_size=(3, 384, 384)),
    'caformer_s18_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21ft1k.pth'),
    'caformer_s18_384_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384_in21ft1k.pth', input_size=(3, 384, 384)),
    'caformer_s18_in21k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21k.pth', num_classes=21841),

    'caformer_s36': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth'),
    'caformer_s36_384': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384.pth', input_size=(3, 384, 384)),
    'caformer_s36_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21ft1k.pth'),
    'caformer_s36_384_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384_in21ft1k.pth', input_size=(3, 384, 384)),
    'caformer_s36_in21k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21k.pth', num_classes=21841),

    'caformer_m36': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth'),
    'caformer_m36_384': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384.pth', input_size=(3, 384, 384)),
    'caformer_m36_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21ft1k.pth'),
    'caformer_m36_384_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384_in21ft1k.pth', input_size=(3, 384, 384)),
    'caformer_m36_in21k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21k.pth', num_classes=21841),

    'caformer_b36': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth'),
    'caformer_b36_384': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384.pth', input_size=(3, 384, 384)),
    'caformer_b36_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21ft1k.pth'),
    'caformer_b36_384_in21ft1k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384_in21ft1k.pth', input_size=(3, 384, 384)),
    'caformer_b36_in21k': _cfg(url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21k.pth', num_classes=21841),
}

class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.post_norm(x)
        # Ensure output is always [B, C, H, W]
        if x.dim() == 4 and x.shape[-1] != x.shape[1]:
            x = x.permute(0, 3, 1, 2)
        return x

class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        if x.dim() == 4 and self.scale.dim() == 1:
            scale = self.scale.view(1, -1, 1, 1)
            return x * scale
        return x * self.scale

class SquaredReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerNormGeneral(nn.Module):
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        if scale and affine_shape is not None:
            self.scale = nn.Parameter(torch.ones(affine_shape))
        else:
            self.scale = None
        if bias and affine_shape is not None:
            self.bias = nn.Parameter(torch.zeros(affine_shape))
        else:
            self.bias = None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.scale
        if self.use_bias:
            x = x + self.bias
        return x

class LayerNormWithoutBias(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # print(f"LayerNormWithoutBias input shape: {x.shape}")
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
            x = F.layer_norm(x, self.weight.shape, self.weight, None, self.eps)
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
            # print(f"LayerNormWithoutBias output shape (4D): {x.shape}")
        else:
            x = F.layer_norm(x, self.weight.shape, self.weight, None, self.eps)
            # print(f"LayerNormWithoutBias output shape (other): {x.shape}")
        return x

class SepConv(nn.Module):
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        input = x
        x = self.pwconv1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        if x.shape == input.shape:
            x = input + x
        return x

class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

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
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.head_dropout(x)
        return x

class MetaFormerBlock(nn.Module):
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        shortcut = x
        x1 = self.norm1(x)
        # Permute to [B, H, W, C] for token_mixer
        x1p = x1.permute(0, 2, 3, 1)
        x2p = self.token_mixer(x1p)
        x2 = x2p.permute(0, 3, 1, 2)
        x3 = self.layer_scale1(x2)
        x4 = self.drop_path1(x3)
        x5 = shortcut + x4
        x5 = self.res_scale1(x5)
        x6 = self.norm2(x5)
        # Permute to [B, H, W, C] for mlp
        x6p = x6.permute(0, 2, 3, 1)
        x7p = self.mlp(x6p)
        x7 = x7p.permute(0, 3, 1, 2)
        x8 = self.layer_scale2(x7)
        x9 = self.drop_path2(x8)
        x10 = x5 + x9
        x10 = self.res_scale2(x10)
        return x10

DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling, kernel_size=7, stride=4, padding=2,
                    post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)), 
                    partial(Downsampling, kernel_size=3, stride=2, padding=1,
                    pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True),
                    partial(Downsampling, kernel_size=3, stride=2, padding=1,
                    pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True),
                    partial(Downsampling, kernel_size=3, stride=2, padding=1,
                    pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True)]

class MetaFormer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6),
                 drop_path_rate=0.,
                 head_dropout=0.0, 
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=nn.Linear,
                 **kwargs,
                 ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = dims[-1]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layers(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                    norm_layers(dims[i+1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                token_mixer=token_mixers[i] if isinstance(token_mixers, (list, tuple)) else token_mixers,
                mlp=mlps[i] if isinstance(mlps, (list, tuple)) else mlps,
                norm_layer=norm_layers[i] if isinstance(norm_layers, (list, tuple)) else norm_layers,
                drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_values[i] if isinstance(layer_scale_init_values, (list, tuple)) else layer_scale_init_values,
                res_scale_init_value=res_scale_init_values[i] if isinstance(res_scale_init_values, (list, tuple)) else res_scale_init_values,
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm.weight', 'norm.bias'}

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # Permute to [B, H, W, C] for final norm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x.mean(dim=[2, 3])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def caformer_s18(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=[Mlp, Mlp, Mlp, Mlp],
        **kwargs
    )
    model.default_cfg = default_cfgs['caformer_s18']
    return model

@register_model
def caformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=[Mlp, Mlp, Mlp, Mlp],
        **kwargs
    )
    model.default_cfg = default_cfgs['caformer_s36']
    return model

@register_model
def caformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=[Mlp, Mlp, Mlp, Mlp],
        **kwargs
    )
    model.default_cfg = default_cfgs['caformer_m36']
    return model

@register_model
def caformer_b36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        mlps=[Mlp, Mlp, Mlp, Mlp],
        **kwargs
    )
    model.default_cfg = default_cfgs['caformer_b36']
    return model
@register_model
def caformer_s18_384(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_384']
    return model

@register_model
def caformer_s18_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_in21ft1k']
    return model

@register_model
def caformer_s18_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_384_in21ft1k']
    return model

@register_model
def caformer_s18_in21k(pretrained=False, **kwargs):
    model = caformer_s18(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_in21k']
    return model

@register_model
def caformer_s36_384(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_384']
    return model

@register_model
def caformer_s36_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_in21ft1k']
    return model

@register_model
def caformer_s36_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_384_in21ft1k']
    return model

@register_model
def caformer_s36_in21k(pretrained=False, **kwargs):
    model = caformer_s36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_in21k']
    return model

@register_model
def caformer_m36_384(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_384']
    return model

@register_model
def caformer_m36_in21ft1k(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_in21ft1k']
    return model

@register_model
def caformer_m36_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_384_in21ft1k']
    return model

@register_model
def caformer_m36_in21k(pretrained=False, **kwargs):
    model = caformer_m36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_in21k']
    return model

@register_model
def caformer_b36_384(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_384']
    return model

@register_model
def caformer_b36_in21ft1k(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_in21ft1k']
    return model

@register_model
def caformer_b36_384_in21ft1k(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_384_in21ft1k']
    return model

@register_model
def caformer_b36_in21k(pretrained=False, **kwargs):
    model = caformer_b36(pretrained=pretrained, **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_in21k']
    return model

def test_caformer_creation():
    models = {
        'caformer_s18': caformer_s18,
        'caformer_s36': caformer_s36,
        'caformer_m36': caformer_m36,
        'caformer_b36': caformer_b36,
    }
    
    for name, model_fn in models.items():
        try:
            model = model_fn(pretrained=False, num_classes=10)
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            print(f"✓ {name}: {output.shape}")
        except Exception as e:
            print(f"✗ {name}: {e}")

if __name__ == "__main__":
    test_caformer_creation() 