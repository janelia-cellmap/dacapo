import attr

from .cnnectome_unet_impl import CNNectomeUNetModule, Upsample, ConvPass
from .architecture import ArchitectureConfig

from funlib.geometry import Coordinate
import torch

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

    _input_shape: Coordinate = attr.ib(
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
        default=False,
        metadata={"help_text": "Whether to use batch normalization."},
    )

    @property
    def input_shape(self) -> Coordinate:
        return self._input_shape

    @property
    def input_shape(self) -> Coordinate:
        return Coordinate(self._input_shape)

    @input_shape.setter
    def input_shape(self, value: Coordinate):
        self._input_shape = value

    def module(self) -> torch.nn.Module:
        fmaps_in = self.fmaps_in
        levels = len(self.downsample_factors) + 1

        if self.kernel_size_down is not None:
            kernel_size_down = self.kernel_size_down
        else:
            kernel_size_down = [[(3,) * self.dims, (3,) * self.dims]] * levels
        if self.kernel_size_up is not None:
            kernel_size_up = self.kernel_size_up
        else:
            kernel_size_up = [[(3,) * self.dims, (3,) * self.dims]] * (levels - 1)

        # downsample factors has to be a list of tuples
        downsample_factors = [tuple(x) for x in self.downsample_factors]

        unet = CNNectomeUNetModule(
            in_channels=fmaps_in,
            num_fmaps=self.num_fmaps,
            num_fmaps_out=self.fmaps_out,
            fmap_inc_factor=self.fmap_inc_factor,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            downsample_factors=downsample_factors,
            constant_upsample=self.constant_upsample,
            padding=self.padding,
            activation_on_upsample=True,
            upsample_channel_contraction=[False]
            + [True] * (len(downsample_factors) - 1),
            use_attention=self.use_attention,
            batch_norm=self.batch_norm,
        )
        if self.upsample_factors is not None and len(self.upsample_factors) > 0:
            layers = [unet]

            for upsample_factor in self.upsample_factors:
                up = Upsample(
                    upsample_factor,
                    mode="nearest",
                    in_channels=self.fmaps_out,
                    out_channels=self.fmaps_out,
                    activation="ReLU",
                )
                layers.append(up)
                conv = ConvPass(
                    self.fmaps_out,
                    self.fmaps_out,
                    kernel_size_up[-1],
                    activation="ReLU",
                    batch_norm=self.batch_norm,
                )
                layers.append(conv)
            unet = torch.nn.Sequential(*layers)

        return unet

    @property
    def input_shape(self) -> Coordinate:
        return Coordinate(self._input_shape)

    @property
    def num_in_channels(self) -> int:
        return self.fmaps_in

    @property
    def num_out_channels(self) -> int:
        return self.fmaps_out

    def scale(self, voxel_size):
        """
        Scale the voxel size according to the upsampling factors.
        """
        if self.upsample_factors is not None:
            for upsample_factor in self.upsample_factors:
                voxel_size = voxel_size / Coordinate(upsample_factor)
        return voxel_size
