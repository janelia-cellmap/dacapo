import torch
import funlib.learn.torch as ft


class UNet(torch.nn.Module):

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

        kernel_size_down = [[(3, 3), (3, 3)]]*levels
        kernel_size_up = [[(3, 3), (3, 3)]]*(levels - 1)

        unet = ft.models.UNet(
            in_channels=fmaps_in,
            num_fmaps=fmaps,
            fmap_inc_factor=fmap_inc_factor,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            downsample_factors=downsample_factors,
            constant_upsample=True,
            padding=padding)
        self.sequence = torch.nn.Sequential(
            unet,
            torch.nn.Conv2d(fmaps, fmaps_out, (1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.sequence(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
