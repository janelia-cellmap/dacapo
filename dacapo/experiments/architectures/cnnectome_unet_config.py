import attr

from .cnnectome_unet import CNNectomeUNet
from .architecture_config import ArchitectureConfig

from funlib.geometry import Coordinate

from typing import List, Optional


@attr.s
class CNNectomeUNetConfig(ArchitectureConfig):
    """
    This class configures the CNNectomeUNet based on
    https://github.com/saalfeldlab/CNNectome/blob/master/CNNectome/networks/unet_class.py

    Includes support for super resolution via the upsampling factors.

    Attributes:
        input_shape: Coordinate
            The shape of the data passed into the network during training.
        fmaps_out: int
            The number of channels produced by your architecture.
        fmaps_in: int
            The number of channels expected from the raw data.
        num_fmaps: int
            The number of feature maps in the top level of the UNet.
        fmap_inc_factor: int
            The multiplication factor for the number of feature maps for each level of the UNet.
        downsample_factors: List[Coordinate]
            The factors to downsample the feature maps along each axis per layer.
        kernel_size_down: Optional[List[Coordinate]]
            The size of the convolutional kernels used before downsampling in each layer.
        kernel_size_up: Optional[List[Coordinate]]
            The size of the convolutional kernels used before upsampling in each layer.
        _eval_shape_increase: Optional[Coordinate]
            The amount by which to increase the input size when just prediction rather than training.
            It is generally possible to significantly increase the input size since we don't have the memory
            constraints of the gradients, the optimizer and the batch size.
        upsample_factors: Optional[List[Coordinate]]
            The amount by which to upsample the output of the UNet.
        constant_upsample: bool
            Whether to use a transpose convolution or simply copy voxels to upsample.
        padding: str
            The padding to use in convolution operations.
        use_attention: bool
            Whether to use attention blocks in the UNet. This is supported for 2D and  3D.
    Methods:
        architecture_type()
            Returns the architecture type.
    Note:
        The architecture_type attribute is set to CNNectomeUNet.
    References:
        Saalfeld, S., Fetter, R., Cardona, A., & Tomancak, P. (2012).

    """

    architecture_type = CNNectomeUNet

    input_shape: Coordinate = attr.ib(
        metadata={
            "help_text": "The shape of the data passed into the network during training."
        }
    )
    fmaps_out: int = attr.ib(
        metadata={"help_text": "The number of channels produced by your architecture."}
    )
    fmaps_in: int = attr.ib(
        metadata={"help_text": "The number of channels expected from the raw data."}
    )
    num_fmaps: int = attr.ib(
        metadata={
            "help_text": "The number of feature maps in the top level of the UNet."
        }
    )
    fmap_inc_factor: int = attr.ib(
        metadata={
            "help_text": "The multiplication factor for the number of feature maps for each "
            "level of the UNet."
        }
    )
    downsample_factors: List[Coordinate] = attr.ib(
        metadata={
            "help_text": "The factors to downsample the feature maps along each axis per layer."
        }
    )
    kernel_size_down: Optional[List[List[Coordinate]]] = attr.ib(
        default=None,
        metadata={
            "help_text": "The size of the convolutional kernels used before downsampling in each layer."
        },
    )
    kernel_size_up: Optional[List[List[Coordinate]]] = attr.ib(
        default=None,
        metadata={
            "help_text": "The size of the convolutional kernels used before upsampling in each layer."
        },
    )
    _eval_shape_increase: Optional[Coordinate] = attr.ib(
        default=None,
        metadata={
            "help_text": "The amount by which to increase the input size when just "
            "prediction rather than training. It is generally possible to significantly "
            "increase the input size since we don't have the memory constraints of the "
            "gradients, the optimizer and the batch size."
        },
    )
    upsample_factors: Optional[List[Coordinate]] = attr.ib(
        default=None,
        metadata={
            "help_text": "The amount by which to upsample the output of the UNet."
        },
    )
    constant_upsample: bool = attr.ib(
        default=True,
        metadata={
            "help_text": "Whether to use a transpose convolution or simply copy voxels to upsample."
        },
    )
    padding: str = attr.ib(
        default="valid",
        metadata={"help_text": "The padding to use in convolution operations."},
    )
    use_attention: bool = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether to use attention blocks in the UNet. This is supported for 2D and  3D."
        },
    )
    batch_norm: bool = attr.ib(
        default=True,
        metadata={"help_text": "Whether to use batch normalization."},
    )
