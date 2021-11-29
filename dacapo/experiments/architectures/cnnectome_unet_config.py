import attr

from .cnnectome_unet import CNNectomeUNet
from .architecture_config import ArchitectureConfig

from funlib.geometry import Coordinate

from typing import List, Optional


@attr.s
class CNNectomeUNetConfig(ArchitectureConfig):
    """This is just a dummy architecture config used for testing. None of the
    attributes have any particular meaning."""

    architecture_type = CNNectomeUNet

    input_shape: Coordinate = attr.ib()
    fmaps_out: int = attr.ib()
    fmaps_in: int = attr.ib()
    num_fmaps: int = attr.ib()
    fmap_inc_factor: int = attr.ib()
    downsample_factors: List[Coordinate] = attr.ib()
    kernel_size_down: Optional[List[Coordinate]] = attr.ib(default=None)
    kernel_size_up: Optional[List[Coordinate]] = attr.ib(default=None)
    _eval_shape_increase: Optional[Coordinate] = attr.ib(default=None)
    upsample_factors: List[Coordinate] = attr.ib(factory=lambda x: list())
    constant_upsample: bool = attr.ib(default=True)
    padding: str = attr.ib(default="valid")
