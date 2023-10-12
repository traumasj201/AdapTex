import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, DropPath, Mlp, Attention
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import repeat
from functools import partial
from .adapter import Adapter


class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, is_af=True, norm_layer=None,
                 act_layer=None, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, patch_size=patch_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = patch_size
        if is_af:
            norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
            act_layer = act_layer or nn.GELU
            dpr = [x.item() for x in torch.linspace(0, 0., kwargs['depth'])]  # stochastic depth decay rule

            self.blocks = nn.Sequential(*[
                CustomBlock(dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], drop_path=dpr[i],
                            norm_layer=norm_layer, act_layer=act_layer)
                for i in range(kwargs['depth'])])

    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = h // self.patch_size, w // self.patch_size
        pos_emb_ind = repeat(torch.arange(h) * (self.width // self.patch_size - w), 'h -> (h w)', w=w) + torch.arange(
            h * w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind + 1), dim=0).long()
        x += self.pos_embed[:, pos_emb_ind]
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class CustomBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaptmlp = Adapter(d_model=256, dropout=0.1, bottleneck=64,
                                init_option='lora',
                                adapter_scalar=0.1,
                                adapter_layernorm_option='none',
                                )

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x + self.drop_path(self.attn(self.norm1(x)))
        adapt_x = self.adaptmlp(x, add_residual=False)
        residual = x

        x = self.drop_path(self.mlp(self.norm2(x)))
        x = x + adapt_x
        x = residual + x
        return x


def get_encoder(args):
    backbone = ResNetV2(
        layers=args.backbone_layers, num_classes=0, global_pool='', in_chans=args.channels,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)

    min_patch_size = 2 ** (len(args.backbone_layers) + 1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps // min_patch_size, backbone=backbone)

    encoder = CustomVisionTransformer(img_size=(args.max_height, args.max_width),
                                      patch_size=args.patch_size,
                                      is_af=args.is_af,
                                      in_chans=args.channels,
                                      num_classes=0,
                                      embed_dim=args.dim,
                                      depth=args.encoder_depth,
                                      num_heads=args.heads,
                                      embed_layer=embed_layer
                                      )
    return encoder
