import torch
import funlib.learn.torch as ft
from .model import Model


class UNet(Model):

    def __init__(
            self,
            fmaps_in,
            fmaps,
            fmaps_out,
            fmap_inc_factor,
            downsample_factors,
            padding):

        super(UNet, self).__init__()

        levels = len(downsample_factors) + 1
        dims = len(downsample_factors[0])

        kernel_size_down = [[(3,)*dims, (3,)*dims]]*levels
        kernel_size_up = [[(3,)*dims, (3,)*dims]]*(levels - 1)

        unet = ft.models.UNet(
            in_channels=fmaps_in,
            num_fmaps=fmaps,
            fmap_inc_factor=fmap_inc_factor,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            downsample_factors=downsample_factors,
            constant_upsample=True,
            padding=padding)
        conv = {
            2: torch.nn.Conv2d,
            3: torch.nn.Conv3d
        }[dims]
        self.sequence = torch.nn.Sequential(
            unet,
            conv(fmaps, fmaps_out, (1,)*dims),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequence(x)
