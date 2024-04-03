import attr

from .architecture_config import ArchitectureConfig
from .cellpose_unet import CellposeUnet

from funlib.geometry import Coordinate

from typing import List, Optional


@attr.s
class CellposUNetConfig(ArchitectureConfig):
    """This class configures the CellPose based on
    https://github.com/MouseLand/cellpose/blob/main/cellpose/resnet_torch.py
    """

    architecture_type = CellposeUnet

    input_shape: Coordinate = attr.ib(
        metadata={
            "help_text": "The shape of the data passed into the network during training."
        }
    )
    nbase: List[int] = attr.ib(
        metadata={
            "help_text": "List of integers representing the number of channels in each layer of the downsample path."
        }
    )
    nout: int = attr.ib(metadata={"help_text": "Number of output channels."})
    mkldnn: Optional[bool] = attr.ib(
        default=False, metadata={"help_text": "Whether to use MKL-DNN acceleration."}
    )
    conv_3D: bool = attr.ib(
        default=False, metadata={"help_text": "Whether to use 3D convolution."}
    )
    max_pool: Optional[bool] = attr.ib(
        default=True, metadata={"help_text": "Whether to use max pooling."}
    )
    diam_mean: Optional[float] = attr.ib(
        default=30.0, metadata={"help_text": "Mean diameter of the cells."}
    )


