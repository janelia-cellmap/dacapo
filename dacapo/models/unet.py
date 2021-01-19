import funlib.learn.torch as ft
from .model import Model


class UNet(Model):
    '''Creates a funlib.learn.torch U-Net for the given data from a model
    configuration.'''

    def __init__(self, data, model_config):

        input_shape = model_config.input_shape
        fmaps_in = max(1, data.raw.num_channels)
        fmaps_out = model_config.fmaps

        super(UNet, self).__init__(input_shape, fmaps_in, fmaps_out)

        levels = len(model_config.downsample_factors) + 1
        dims = len(model_config.downsample_factors[0])

        if hasattr(model_config, "kernel_size_down"):
            kernel_size_down = model_config.kernel_size_down
        else:
            kernel_size_down = [[(3,)*dims, (3,)*dims]]*levels
        if hasattr(model_config, "kernel_size_up"):
            kernel_size_up = model_config.kernel_size_up
        else:
            kernel_size_up = [[(3,)*dims, (3,)*dims]]*(levels - 1)

        self.unet = ft.models.UNet(
            in_channels=fmaps_in,
            num_fmaps=model_config.fmaps,
            fmap_inc_factor=model_config.fmap_inc_factor,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            downsample_factors=model_config.downsample_factors,
            constant_upsample=True,
            padding=model_config.padding)

    def forward(self, x):
        return self.unet(x)
