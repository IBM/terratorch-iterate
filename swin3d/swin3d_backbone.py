from typing import Sequence

from timm.models import FeatureInfo
from torch import nn
from torch.nn.modules import LayerNorm

from .swin3d import SwinTransformer3D


class Swin3dBackbone(SwinTransformer3D):
    def __init__(
        self,
        patch_size: Sequence[int] = (4, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: Sequence[int] = [2, 2, 6, 2],
        num_heads: Sequence[int] = [3, 6, 12, 24],
        window_size: Sequence[int] = (2, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: bool | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: type[nn.Module] = LayerNorm,
        patch_norm: bool = False,
    ):
        super().__init__(
            patch_size,
            in_chans,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            patch_norm,
        )
        self.feature_info = FeatureInfo(
            self.feature_info, list(range(len(self.feature_info)))
        )  # instantiate feature info

    def forward(self, x):
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        # store the output of each layer, since we are not using timm. It would otherwise do it for us
        outputs = []
        for layer in self.layers:
            x = layer(x.contiguous())
            outputs.append(x.clone())

        return outputs

    def prepare_features_for_image_model(self, x):
        return [
            layer_output.squeeze(2).contiguous() for layer_output in x
        ]  # removes temporal dimension
