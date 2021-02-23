import funlib.learn.torch as ft
from .model import Model, ModelConfig

from typing import List, Dict, Optional, Tuple
import attr

from enum import Enum


class ConvPaddingOption(Enum):
    VALID = "valid"
    SAME = "same"


@attr.s
class UNetConfig(ModelConfig):
    # standard model attributes
    name: str = attr.ib()
    input_shape: List[int] = attr.ib()
    output_shape: Optional[List[int]] = attr.ib()
    fmaps_out: int = attr.ib()

    # unet attributes
    fmap_inc_factor: int = attr.ib()
    kernel_size_down: Optional[List[Tuple[int]]] = attr.ib()
    kernel_size_up: Optional[List[Tuple[int]]] = attr.ib()
    downsample_factors: List[Tuple[int]] = attr.ib()
    constant_upsample: bool = attr.ib(default=True)
    padding: ConvPaddingOption = attr.ib(default=ConvPaddingOption.VALID)

    # attributes that can be read from other configs:
    fmaps_in: Optional[int] = attr.ib(
        default=None
    )  # can be read from data num_channels

    def model(self, fmaps_in: int):
        assert self.fmaps_in is None or self.fmaps_in == fmaps_in
        self.fmaps_in = fmaps_in
        return UNet(self)


class UNet(Model):
    """Creates a funlib.learn.torch U-Net for the given data from a model
    configuration."""

    def __init__(self, model_config: UNetConfig):

        super(UNet, self).__init__(model_config)

        fmaps_in = model_config.fmaps_in
        levels = len(model_config.downsample_factors) + 1
        dims = len(model_config.downsample_factors[0])

        if hasattr(model_config, "kernel_size_down"):
            kernel_size_down = model_config.kernel_size_down
        else:
            kernel_size_down = [[(3,) * dims, (3,) * dims]] * levels
        if hasattr(model_config, "kernel_size_up"):
            kernel_size_up = model_config.kernel_size_up
        else:
            kernel_size_up = [[(3,) * dims, (3,) * dims]] * (levels - 1)

        self.unet = ft.models.UNet(
            in_channels=fmaps_in,
            num_fmaps=model_config.fmaps,
            fmap_inc_factor=model_config.fmap_inc_factor,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            downsample_factors=model_config.downsample_factors,
            constant_upsample=True,
            padding=model_config.padding,
        )

    def forward(self, x):
        return self.unet(x)
