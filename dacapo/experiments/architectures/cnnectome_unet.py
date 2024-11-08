from .architecture import Architecture

import torch
import torch.nn as nn

import math


class CNNectomeUNet(Architecture):
    """
    A U-Net architecture for 3D or 4D data. The U-Net expects 3D or 4D tensors
    shaped like::

        ``(batch=1, channels, [length,] depth, height, width)``.

    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution. It will perfrom 4D convolutions as
    long as ``length`` is greater than 1. As soon as ``length`` is 1 due to a
    valid convolution, the time dimension will be dropped and tensors with
    ``(b, c, z, y, x)`` will be use (and returned) from there on.

    Attributes:
            fmaps_in:
                The number of input channels.
            fmaps_out:
                The number of feature maps in the output layer. This is also the
                number of output feature maps. Stored in the ``channels`` dimension.
            num_fmaps:
                The number of feature maps in the first layer. This is also the
                number of output feature maps. Stored in the ``channels`` dimension.
            fmap_inc_factor:
                By how much to multiply the number of feature maps between layers.
                If layer 0 has ``k`` feature maps, layer ``l`` will have
                ``k*fmap_inc_factor**l``.
            downsample_factors:
                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.
            kernel_size_down (optional):
                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the corresponding
                level of the build on the left side. Kernel sizes can be given as
                tuples or integer. If not given, each convolutional pass will
                consist of two 3x3x3 convolutions.
            kernel_size_up (optional):
                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the corresponding
                level of the build on the right side. Within one of the lists going
                from left to right. Kernel sizes can be given as tuples or integer.
                If not given, each convolutional pass will consist of two 3x3x3
                convolutions.
            activation
                Which activation to use after a convolution. Accepts the name of
                any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).
            fov (optional):
                Initial field of view in physical units
            voxel_size (optional):
                Size of a voxel in the input data, in physical units
            num_heads (optional):
                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a multi-task
                learning context.
            constant_upsample (optional):
                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid' (default).
            upsample_channel_contraction:
                When performing the ConvTranspose, whether to reduce the number
                of channels by the fmap_increment_factor. can be either bool or
                list of bools to apply independently per layer.
            activation_on_upsample:
                Whether or not to add an activation after the upsample operation.
            use_attention:
                Whether or not to use an attention block in the U-Net.
        Methods:
            forward(x):
                Forward pass of the U-Net.
            scale(voxel_size):
                Scale the voxel size according to the upsampling factors.
            input_shape:
                Return the input shape of the U-Net.
            num_in_channels:
                Return the number of input channels.
            num_out_channels:
                Return the number of output channels.
            eval_shape_increase:
                Return the increase in shape due to the U-Net.
        Note:
            This class is a wrapper around the ``CNNectomeUNetModule`` class.
            The ``CNNectomeUNetModule`` class is the actual implementation of the
            U-Net architecture.
    """

    def __init__(self, architecture_config):
        """
        Initialize the U-Net architecture.

        Args:
            architecture_config (dict): A dictionary containing the configuration
            of the U-Net architecture. The dictionary should contain the following
            keys:
                - input_shape: The shape of the input data.
                - fmaps_out: The number of output feature maps.
                - fmaps_in: The number of input feature maps.
                - num_fmaps: The number of feature maps in the first layer.
                - fmap_inc_factor: The factor by which the number of feature maps
                increases between layers.
                - downsample_factors: List of tuples ``(z, y, x)`` to use to down-
                and up-sample the feature maps between layers.
                - kernel_size_down (optional): List of lists of kernel sizes. The
                number of sizes in a list determines the number of convolutional
                layers in the corresponding level of the build on the left side.
                Kernel sizes can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.
                - kernel_size_up (optional): List of lists of kernel sizes. The
                number of sizes in a list determines the number of convolutional
                layers in the corresponding level of the build on the right side.
                Within one of the lists going from left to right. Kernel sizes can
                be given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.
                - constant_upsample (optional): If set to true, perform a constant
                upsampling instead of a transposed convolution in the upsampling
                layers.
                - padding (optional): How to pad convolutions. Either 'same' or
                'valid' (default).
                - upsample_factors (optional): List of tuples ``(z, y, x)`` to use
                to upsample the feature maps between layers.
                - activation_on_upsample (optional): Whether or not to add an
                activation after the upsample operation.
                - use_attention (optional): Whether or not to use an attention block
                in the U-Net.
                - batch_norm (optional): Whether to use batch normalization.
        Raises:
            ValueError: If the input shape is not given.
        Examples:
            >>> architecture_config = {
            ...     "input_shape": (1, 1, 128, 128, 128),
            ...     "fmaps_out": 1,
            ...     "fmaps_in": 1,
            ...     "num_fmaps": 24,
            ...     "fmap_inc_factor": 2,
            ...     "downsample_factors": [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            ...     "kernel_size_down": [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     "kernel_size_up": [[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     "constant_upsample": False,
            ...     "padding": "valid",
            ...     "upsample_factors": [(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            ...     "activation_on_upsample": True,
            ...     "use_attention": False
            ... }
            >>> unet = CNNectomeUNet(architecture_config)
        Note:
            The input shape should be given as a tuple ``(batch, channels, [length,] depth, height, width)``.
        """
        super().__init__()

        self._input_shape = architecture_config.input_shape
        self._eval_shape_increase = architecture_config._eval_shape_increase
        self.fmaps_out = architecture_config.fmaps_out
        self.fmaps_in = architecture_config.fmaps_in
        self.num_fmaps = architecture_config.num_fmaps
        self.fmap_inc_factor = architecture_config.fmap_inc_factor
        self.downsample_factors = architecture_config.downsample_factors
        self.kernel_size_down = architecture_config.kernel_size_down
        self.kernel_size_up = architecture_config.kernel_size_up
        self.constant_upsample = architecture_config.constant_upsample
        self.padding = architecture_config.padding
        self.upsample_factors = architecture_config.upsample_factors
        self.upsample_factors = (
            self.upsample_factors if self.upsample_factors is not None else []
        )
        self.use_attention = architecture_config.use_attention
        self.batch_norm = architecture_config.batch_norm

        self.unet = self.module()

    @property
    def eval_shape_increase(self):
        """
        The increase in shape due to the U-Net.

        Returns:
            The increase in shape due to the U-Net.
        Raises:
            AttributeError: If the increase in shape is not given.
        Examples:
            >>> unet.eval_shape_increase
            (1, 1, 128, 128, 128)
        Note:
            The increase in shape should be given as a tuple ``(batch, channels, [length,] depth, height, width)``.
        """
        if self._eval_shape_increase is None:
            return super().eval_shape_increase
        return self._eval_shape_increase

    def module(self):
        """
        Create the U-Net module.

        Returns:
            The U-Net module.
        Raises:
            AttributeError: If the number of input channels is not given.
            AttributeError: If the number of output channels is not given.
            AttributeError: If the number of feature maps in the first layer is not given.
            AttributeError: If the factor by which the number of feature maps increases between layers is not given.
            AttributeError: If the downsample factors are not given.
            AttributeError: If the kernel sizes for the down pass are not given.
            AttributeError: If the kernel sizes for the up pass are not given.
            AttributeError: If the constant upsample flag is not given.
            AttributeError: If the padding is not given.
            AttributeError: If the upsample factors are not given.
            AttributeError: If the activation on upsample flag is not given.
            AttributeError: If the use attention flag is not given.
        Examples:
            >>> unet.module()
            CNNectomeUNetModule(
                in_channels=1,
                num_fmaps=24,
                num_fmaps_out=1,
                fmap_inc_factor=2,
                kernel_size_down=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                kernel_size_up=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
                downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
                constant_upsample=False,
                padding='valid',
                activation_on_upsample=True,
                upsample_channel_contraction=[False, True, True],
                use_attention=False
            )
        Note:
            The U-Net module is an instance of the ``CNNectomeUNetModule`` class.

        """
        fmaps_in = self.fmaps_in
        levels = len(self.downsample_factors) + 1
        dims = len(self.downsample_factors[0])

        if hasattr(self, "kernel_size_down"):
            kernel_size_down = self.kernel_size_down
        else:
            kernel_size_down = [[(3,) * dims, (3,) * dims]] * levels
        if hasattr(self, "kernel_size_up"):
            kernel_size_up = self.kernel_size_up
        else:
            kernel_size_up = [[(3,) * dims, (3,) * dims]] * (levels - 1)

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
        if len(self.upsample_factors) > 0:
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
                    [(3,) * len(upsample_factor)] * 2,
                    activation="ReLU",
                    batch_norm=self.batch_norm,
                )
                layers.append(conv)
            unet = torch.nn.Sequential(*layers)

        return unet

    def scale(self, voxel_size):
        """
        Scale the voxel size according to the upsampling factors.

        Args:
            voxel_size (tuple): The size of a voxel in the input data.
        Returns:
            The scaled voxel size.
        Raises:
            ValueError: If the voxel size is not given.
        Examples:
            >>> unet.scale((1, 1, 1))
            (1, 1, 1)
        Note:
            The voxel size should be given as a tuple ``(z, y, x)``.
        """
        for upsample_factor in self.upsample_factors:
            voxel_size = voxel_size / upsample_factor
        return voxel_size

    @property
    def input_shape(self):
        """
        Return the input shape of the U-Net.

        Returns:
            The input shape of the U-Net.
        Raises:
            AttributeError: If the input shape is not given.
        Examples:
            >>> unet.input_shape
            (1, 1, 128, 128, 128)
        Note:
            The input shape should be given as a tuple ``(batch, channels, [length,] depth, height, width)``.
        """
        return self._input_shape

    @property
    def num_in_channels(self) -> int:
        """
        Return the number of input channels.

        Returns:
            The number of input channels.
        Raises:
            AttributeError: If the number of input channels is not given.
        Examples:
            >>> unet.num_in_channels
            1
        Note:
            The number of input channels should be given as an integer.
        """
        return self.fmaps_in

    @property
    def num_out_channels(self) -> int:
        """
        Return the number of output channels.

        Returns:
            The number of output channels.
        Raises:
            AttributeError: If the number of output channels is not given.
        Examples:
            >>> unet.num_out_channels
            1
        Note:
            The number of output channels should be given as an integer.
        """
        return self.fmaps_out

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (Tensor): The input tensor.
        Returns:
            The output tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> unet = CNNectomeUNet(architecture_config)
            >>> x = torch.randn(1, 1, 64, 64, 64)
            >>> unet(x)
        Note:
            The input tensor should be given as a 5D tensor.
        """
        return self.unet(x)


class CNNectomeUNetModule(torch.nn.Module):
    """
    A U-Net module for 3D or 4D data. The U-Net expects 3D or 4D tensors shaped
    like::

            ``(batch=1, channels, [length,] depth, height, width)``.

    This U-Net performs only "valid" convolutions, i.e., sizes of the feature maps
    decrease after each convolution. It will perfrom 4D convolutions as long as
    ``length`` is greater than 1. As soon as ``length`` is 1 due to a valid
    convolution, the time dimension will be dropped and tensors with ``(b, c, z, y, x)``
    will be use (and returned) from there on.

    Attributes:
        num_levels:
            The number of levels in the U-Net.
        num_heads:
            The number of decoders.
        in_channels:
            The number of input channels.
        out_channels:
            The number of output channels.
        dims:
            The number of dimensions.
        use_attention:
            Whether or not to use an attention block in the U-Net.
        l_conv:
            The left convolutional passes.
        l_down:
            The left downsample layers.
        r_up:
            The right up/crop/concatenate layers.
        r_conv:
            The right convolutional passes.
        kernel_size_down:
            The kernel sizes for the down pass.
        kernel_size_up:
            The kernel sizes for the up pass.
        fmap_inc_factor:
            The factor by which the number of feature maps increases between layers.
        downsample_factors:
            The downsample factors.
        constant_upsample:
            Whether to perform a constant upsampling instead of a transposed convolution.
        padding:
            How to pad convolutions.
        upsample_channel_contraction:
            Whether to reduce the number of channels by the fmap_increment_factor.
        activation_on_upsample:
            Whether or not to add an activation after the upsample operation.
        use_attention:
            Whether or not to use an attention block in the U-Net.
        attention:
            The attention blocks.
    Methods:
        rec_forward(level, f_in):
            Recursive forward pass of the U-Net.
        forward(x):
            Forward pass of the U-Net.
    Note:
        The input tensor should be given as a 5D tensor.
    """

    def __init__(
        self,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down=None,
        kernel_size_up=None,
        activation="ReLU",
        num_fmaps_out=None,
        num_heads=1,
        constant_upsample=False,
        padding="valid",
        upsample_channel_contraction=False,
        activation_on_upsample=False,
        use_attention=False,
        batch_norm=True,
    ):
        """
        Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        This U-Net performs only "valid" convolutions, i.e., sizes of the
        feature maps decrease after each convolution. It will perfrom 4D
        convolutions as long as ``length`` is greater than 1. As soon as
        ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:
            in_channels:
                The number of input channels.
            num_fmaps:
                The number of feature maps in the first layer. This is also the
                number of output feature maps. Stored in the ``channels``
                dimension.
            fmap_inc_factor:
                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.
            downsample_factors:
                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.
            kernel_size_down (optional):
                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.
            kernel_size_up (optional):
                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.
            activation:
                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).
            fov (optional):
                Initial field of view in physical units
            voxel_size (optional):
                Size of a voxel in the input data, in physical units
            num_heads (optional):
                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.
            constant_upsample (optional):
                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.
            padding (optional):
                How to pad convolutions. Either 'same' or 'valid' (default).
            upsample_channel_contraction:
                When performing the ConvTranspose, whether to reduce the number
                of channels by the fmap_increment_factor. can be either bool
                or list of bools to apply independently per layer.
            activation_on_upsample:
                Whether or not to add an activation after the upsample operation.
            use_attention:
                Whether or not to use an attention block in the U-Net.
            attention:
                The attention blocks.
        Returns:
            The U-Net module.
        Raises:
            ValueError: If the number of input channels is not given.
        Examples:
            >>> unet = CNNectomeUNetModule(
            ...     in_channels=1,
            ...     num_fmaps=24,
            ...     num_fmaps_out=1,
            ...     fmap_inc_factor=2,
            ...     kernel_size_down=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     kernel_size_up=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            ...     constant_upsample=False,
            ...     padding='valid',
            ...     activation_on_upsample=True,
            ...     upsample_channel_contraction=[False, True, True],
            ...     use_attention=False
            ... )
        Note:
            The input tensor should be given as a 5D tensor.
        """

        super().__init__()

        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = num_fmaps_out if num_fmaps_out else num_fmaps
        upsample_channel_contraction = (
            [upsample_channel_contraction] * self.num_levels
            if type(upsample_channel_contraction) == bool
            else upsample_channel_contraction
        )

        self.dims = len(downsample_factors[0])
        self.use_attention = use_attention
        self.batch_norm = batch_norm

        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [[(3,) * self.dims, (3,) * self.dims]] * self.num_levels
        self.kernel_size_down = kernel_size_down
        if kernel_size_up is None:
            kernel_size_up = [[(3,) * self.dims, (3,) * self.dims]] * (
                self.num_levels - 1
            )
        self.kernel_size_up = kernel_size_up

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(f * ff for f, ff in zip(factor, factor_product))
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList(
            [
                ConvPass(
                    (
                        in_channels
                        if level == 0
                        else num_fmaps * fmap_inc_factor ** (level - 1)
                    ),
                    num_fmaps * fmap_inc_factor**level,
                    kernel_size_down[level],
                    activation=activation,
                    padding=padding,
                    batch_norm=self.batch_norm,
                )
                for level in range(self.num_levels)
            ]
        )
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList(
            [
                Downsample(downsample_factors[level])
                for level in range(self.num_levels - 1)
            ]
        )

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Upsample(
                            downsample_factors[level],
                            mode="nearest" if constant_upsample else "transposed_conv",
                            in_channels=num_fmaps * fmap_inc_factor ** (level + 1),
                            out_channels=num_fmaps
                            * fmap_inc_factor
                            ** (level + (1 - upsample_channel_contraction[level])),
                            crop_factor=crop_factors[level],
                            next_conv_kernel_sizes=kernel_size_up[level],
                            activation=activation if activation_on_upsample else None,
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )
        #  if num_fmaps_out is None or level != self.num_levels-1 else num_fmaps_out
        if self.use_attention:
            self.attention = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            AttentionBlockModule(
                                F_g=num_fmaps * fmap_inc_factor ** (level + 1),
                                F_l=num_fmaps * fmap_inc_factor**level,
                                F_int=(
                                    num_fmaps
                                    * fmap_inc_factor
                                    ** (
                                        level
                                        + (1 - upsample_channel_contraction[level])
                                    )
                                    if num_fmaps_out is None or level != 0
                                    else num_fmaps_out
                                ),
                                dims=self.dims,
                                upsample_factor=downsample_factors[level],
                                batch_norm=self.batch_norm,
                            )
                            for level in range(self.num_levels - 1)
                        ]
                    )
                    for _ in range(num_heads)
                ]
            )

        # right convolutional passes
        self.r_conv = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvPass(
                            num_fmaps * fmap_inc_factor**level
                            + num_fmaps
                            * fmap_inc_factor
                            ** (level + (1 - upsample_channel_contraction[level])),
                            (
                                num_fmaps * fmap_inc_factor**level
                                if num_fmaps_out is None or level != 0
                                else num_fmaps_out
                            ),
                            kernel_size_up[level],
                            activation=activation,
                            padding=padding,
                            batch_norm=self.batch_norm,
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

    def rec_forward(self, level, f_in):
        """
        Recursive forward pass of the U-Net.

        Args:
            level (int): The level of the U-Net.
            f_in (Tensor): The input tensor.
        Returns:
            The output tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> unet = CNNectomeUNetModule(
            ...     in_channels=1,
            ...     num_fmaps=24,
            ...     num_fmaps_out=1,
            ...     fmap_inc_factor=2,
            ...     kernel_size_down=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     kernel_size_up=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            ...     constant_upsample=False,
            ...     padding='valid',
            ...     activation_on_upsample=True,
            ...     upsample_channel_contraction=[False, True, True],
            ...     use_attention=False
            ... )
            >>> x = torch.randn(1, 1, 64, 64, 64)
            >>> unet.rec_forward(2, x)
        Note:
            The input tensor should be given as a 5D tensor.
        """
        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:
            fs_out = [f_left] * self.num_heads

        else:
            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            if self.use_attention:
                f_left_attented = [
                    self.attention[h][i](gs_out[h], f_left)
                    for h in range(self.num_heads)
                ]
                fs_right = [
                    self.r_up[h][i](gs_out[h], f_left_attented[h])
                    for h in range(self.num_heads)
                ]
            else:  # up, concat, and crop
                fs_right = [
                    self.r_up[h][i](gs_out[h], f_left) for h in range(self.num_heads)
                ]

            # convolve
            fs_out = [self.r_conv[h][i](fs_right[h]) for h in range(self.num_heads)]

        return fs_out

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (Tensor): The input tensor.
        Returns:
            The output tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> unet = CNNectomeUNetModule(
            ...     in_channels=1,
            ...     num_fmaps=24,
            ...     num_fmaps_out=1,
            ...     fmap_inc_factor=2,
            ...     kernel_size_down=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     kernel_size_up=[[(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)], [(3, 3, 3), (3, 3, 3)]],
            ...     downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
            ...     constant_upsample=False,
            ...     padding='valid',
            ...     activation_on_upsample=True,
            ...     upsample_channel_contraction=[False, True, True],
            ...     use_attention=False
            ... )
            >>> x = torch.randn(1, 1, 64, 64, 64)
            >>> unet(x)
        Note:
            The input tensor should be given as a 5D tensor.
        """
        y = self.rec_forward(self.num_levels - 1, x)

        if self.num_heads == 1:
            return y[0]

        return y


class ConvPass(torch.nn.Module):
    """
    Convolutional pass module. This module performs a series of convolutional
    layers followed by an activation function. The module can also pad the
    feature maps to ensure translation equivariance. The module can perform
    2D or 3D convolutions.

    Attributes:
        dims:
            The number of dimensions.
        conv_pass:
            The convolutional pass module.
    Methods:
        forward(x):
            Forward pass of the Conv
    Note:
        The input tensor should be given as a 5D tensor.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        activation,
        padding="valid",
        batch_norm=True,
    ):
        """
        Convolutional pass module. This module performs a series of
        convolutional layers followed by an activation function.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_sizes (list): The kernel sizes for the convolutional layers.
            activation (str): The activation function to use.
            padding (optional): How to pad convolutions. Either 'same' or 'valid'.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> conv_pass = ConvPass(1, 1, [(3, 3, 3), (3, 3, 3)], "ReLU")
        Note:
            The input tensor should be given as a 5D tensor.

        """
        super(ConvPass, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)

        layers = []

        for kernel_size in kernel_sizes:
            self.dims = len(kernel_size)

            conv = {
                2: torch.nn.Conv2d,
                3: torch.nn.Conv3d,
            }[self.dims]

            if padding == "same":
                pad = tuple(k // 2 for k in kernel_size)
            else:
                pad = 0

            try:
                layers.append(conv(in_channels, out_channels, kernel_size, padding=pad))
                if batch_norm:
                    layers.append(
                        {
                            2: torch.nn.BatchNorm2d,
                            3: torch.nn.BatchNorm3d,
                        }[
                            self.dims
                        ](out_channels)
                    )
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

            in_channels = out_channels

            if activation is not None:
                layers.append(activation())

        self.conv_pass = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ConvPass module.

        Args:
            x (Tensor): The input tensor.
        Returns:
            The output tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> conv_pass = ConvPass(1, 1, [(3, 3, 3), (3, 3, 3)], "ReLU")
            >>> x = torch.randn(1, 1, 64, 64, 64)
            >>> conv_pass(x)
        Note:
            The input tensor should be given as a 5D tensor.
        """
        return self.conv_pass(x)


class Downsample(torch.nn.Module):
    """
    Downsample module. This module performs downsampling of the input tensor
    using either max-pooling or average pooling. The module can also crop the
    feature maps to ensure translation equivariance with a stride of the
    downsampling factor.

    Attributes:
        dims:
            The number of dimensions.
        downsample_factor:
            The downsampling factor.
        down:
            The downsampling layer.
    Methods:
        forward(x):
            Downsample the input tensor.
    Note:
        The input tensor should be given as a 5D tensor.

    """

    def __init__(self, downsample_factor):
        """
        Downsample module. This module performs downsampling of the input tensor
        using either max-pooling or average pooling.

        Args:
            downsample_factor (tuple): The downsampling factor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> downsample = Downsample((2, 2, 2))
        Note:
            The input tensor should be given as a 5D tensor.
        """
        super(Downsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d,  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(downsample_factor, stride=downsample_factor)

    def forward(self, x):
        """
        Downsample the input tensor.

        Args:
            x (Tensor): The input tensor.
        Returns:
            The downsampled tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> downsample = Downsample((2, 2, 2))
            >>> x = torch.randn(1, 1, 64, 64, 64)
            >>> downsample(x)
        Note:
            The input tensor should be given as a 5D tensor.
        """
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (x.size(), self.downsample_factor, self.dims - d)
                )

        return self.down(x)


class Upsample(torch.nn.Module):
    """
    Upsample module. This module performs upsampling of the input tensor using
    either transposed convolutions or nearest neighbor interpolation. The
    module can also crop the feature maps to ensure translation equivariance
    with a stride of the upsampling factor.

    Attributes:
        crop_factor:
            The crop factor.
        next_conv_kernel_sizes:
            The kernel sizes for the convolutional layers.
        dims:
            The number of dimensions.
        up:
            The upsampling layer.
    Methods:
        crop_to_factor(x, factor, kernel_sizes):
            Crop feature maps to ensure translation equivariance with stride of
            upsampling factor.
        crop(x, shape):
            Center-crop x to match spatial dimensions given by shape.
        forward(g_out, f_left=None):
            Forward pass of the Upsample module.
    Note:
        The input tensor should be given as a 5D tensor.

    """

    def __init__(
        self,
        scale_factor,
        mode="transposed_conv",
        in_channels=None,
        out_channels=None,
        crop_factor=None,
        next_conv_kernel_sizes=None,
        activation=None,
    ):
        """
        Upsample module. This module performs upsampling of the input tensor

        Args:
            scale_factor (tuple): The upsampling factor.
            mode (optional): The upsampling mode. Either 'transposed_conv' or
            'nearest
            in_channels (optional): The number of input channels.
            out_channels (optional): The number of output channels.
            crop_factor (optional): The crop factor.
            next_conv_kernel_sizes (optional): The kernel sizes for the convolutional layers.
            activation (optional): The activation function to use.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> upsample = Upsample(scale_factor=(2, 2, 2), in_channels=1, out_channels=1)
            >>> upsample = Upsample(scale_factor=(2, 2, 2), in_channels=1, out_channels=1, activation="ReLU")
            >>> upsample = Upsample(scale_factor=(2, 2, 2), in_channels=1, out_channels=1, crop_factor=(2, 2, 2), next_conv_kernel_sizes=[(3, 3, 3), (3, 3, 3)])
        Note:
            The input tensor should be given as a 5D tensor.
        """
        super(Upsample, self).__init__()

        if activation is not None:
            activation = getattr(torch.nn, activation)
        assert (crop_factor is None) == (
            next_conv_kernel_sizes is None
        ), "crop_factor and next_conv_kernel_sizes have to be given together"

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes

        self.dims = len(scale_factor)

        layers = []

        if mode == "transposed_conv":
            up = {2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}[self.dims]

            layers.append(
                up(
                    in_channels,
                    out_channels,
                    kernel_size=scale_factor,
                    stride=scale_factor,
                )
            )

        else:
            layers.append(torch.nn.Upsample(scale_factor=scale_factor, mode=mode))
            conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[self.dims]
            layers.append(
                conv(
                    in_channels,
                    out_channels,
                    kernel_size=(1,) * self.dims,
                    stride=(1,) * self.dims,
                ),
            )
        if activation is not None:
            layers.append(activation())

        if len(layers) > 1:
            self.up = torch.nn.Sequential(*layers)
        else:
            self.up = layers[0]

    def crop_to_factor(self, x, factor, kernel_sizes):
        """
        Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).

        We need to ensure that the feature map is large enough to ensure that
        the translation equivariance is maintained. This is done by cropping
        the feature map to the largest size that is a multiple of the factor
        and that is large enough to ensure that the translation equivariance
        is maintained.

        We need (spatial_shape - convolution_crop) to be a multiple of factor,
        i.e.:
        (s - c) = n*k

        where s is the spatial size of the feature map, c is the crop due to
        the convolutions, n is the number of strides of the upsampling factor,
        and k is the upsampling factor.

        We want to find the largest n for which s' = n*k + c <= s

        n = floor((s - c)/k)

        This gives us the target shape s'

        s' = n*k + c

        Args:
            x (Tensor): The input tensor.
            factor (tuple): The upsampling factor.
            kernel_sizes (list): The kernel sizes for the convolutional layers.
        Returns:
            The cropped tensor.
        Raises:
            RuntimeError: If the feature map is too small to ensure translation equivariance.
        Examples:
            >>> upsample = Upsample(scale_factor=(2, 2, 2), in_channels=1, out_channels=1)
            >>> x = torch.randn(1, 1, 64, 64, 64)
            >>> upsample.crop_to_factor(x, (2, 2, 2), [(3, 3, 3), (3, 3, 3)])
        Note:
            The input tensor should be given as a 5D tensor.
        """

        shape = x.size()
        spatial_shape = shape[-self.dims :]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes) for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c) / f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n * f + c for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:
            assert all(
                ((t > c) for t, c in zip(target_spatial_shape, convolution_crop))
            ), (
                "Feature map with shape %s is too small to ensure "
                "translation equivariance with factor %s and following "
                "convolutions %s" % (shape, factor, kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """
        Center-crop x to match spatial dimensions given by shape.

        Args:
            x (Tensor): The input tensor.
            shape (tuple): The target shape.
        Returns:
            The center-cropped tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> upsample = Upsample(scale_factor=(2, 2, 2), in_channels=1, out_channels=1)
            >>> x = torch.randn(1, 1, 64, 64, 64)
            >>> upsample.crop(x, (32, 32, 32))
        Note:
            The input tensor should be given as a 5D tensor.
        """

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, g_out, f_left=None):
        """
        Forward pass of the Upsample module.

        Args:
            g_out (Tensor): The gating signal tensor.
            f_left (Tensor): The input feature tensor.
        Returns:
            The output feature tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> upsample = Upsample(scale_factor=(2, 2, 2), in_channels=1, out_channels=1)
            >>> g_out = torch.randn(1, 1, 64, 64, 64)
            >>> f_left = torch.randn(1, 1, 32, 32, 32)
            >>> upsample(g_out, f_left)
        Note:
            The gating signal and input feature tensors should be given as 5D tensors.
        """
        g_up = self.up(g_out)

        if self.next_conv_kernel_sizes is not None:
            g_cropped = self.crop_to_factor(
                g_up, self.crop_factor, self.next_conv_kernel_sizes
            )
        else:
            g_cropped = g_up

        if f_left is not None:
            f_cropped = self.crop(f_left, g_cropped.size()[-self.dims :])

            return torch.cat([f_cropped, g_cropped], dim=1)
        else:
            return g_cropped


class AttentionBlockModule(nn.Module):
    """
    Attention Block Module:

       The AttentionBlock uses two separate pathways to process 'g' and 'x',
       combines them, and applies a sigmoid activation to generate an attention map.
       This map is then used to scale the input features 'x', resulting in an output
       that focuses on important features as dictated by the gating signal 'g'.

       The attention block takes two inputs: 'g' (gating signal) and 'x' (input features).

               [g] --> W_g --\                 /--> psi --> * --> [output]
                               \               /
               [x] --> W_x --> [+] --> relu --

       Where:
           - W_g and W_x are 1x1 Convolution followed by Batch Normalization
           - [+] indicates element-wise addition
           - relu is the Rectified Linear Unit activation function
           - psi is a sequence of 1x1 Convolution, Batch Normalization, and Sigmoid activation
           - * indicates element-wise multiplication between the output of psi and input feature 'x'
           - [output] has the same dimensions as input 'x', selectively emphasized by attention weights

       Attributes:
           dims:
               The number of dimensions of the input tensors.
           kernel_sizes:
               The kernel sizes for the convolutional layers.
           upsample_factor:
               The factor by which to upsample the attention map.
           W_g:
               The 1x1 Convolutional layer for the gating signal.
           W_x:
               The 1x1 Convolutional layer for the input features.
           psi:
               The 1x1 Convolutional layer followed by Sigmoid activation.
           up:
               The upsampling layer to match the dimensions of the input features.
           relu:
               The Rectified Linear Unit activation function.
       Methods:
           calculate_and_apply_padding(smaller_tensor, larger_tensor):
               Calculate and apply symmetric padding to the smaller tensor to match the dimensions of the larger tensor.
           forward(g, x):
               Forward pass of the Attention Block.
       Note:
           The AttentionBlockModule is an instance of the ``torch.nn.Module`` class.
    """

    def __init__(self, F_g, F_l, F_int, dims, upsample_factor=None, batch_norm=True):
        """
        Initialize the Attention Block Module.

        Args:
            F_g (int): The number of feature maps in the gating signal tensor.
            F_l (int): The number of feature maps in the input feature tensor.
            F_int (int): The number of feature maps in the intermediate tensor.
            dims (int): The number of dimensions of the input tensors.
            upsample_factor (optional): The factor by which to upsample the attention map.
        Returns:
            The Attention Block Module.
        Raises:
            RuntimeError: If the gating signal and input feature tensors have different dimensions.
        Examples:
            >>> attention_block = AttentionBlockModule(F_g=1, F_l=1, F_int=1, dims=3)
        Note:
            The number of feature maps should be given as an integer.
        """

        super(AttentionBlockModule, self).__init__()
        self.dims = dims
        self.kernel_sizes = [(1,) * self.dims, (1,) * self.dims]
        self.batch_norm = batch_norm
        if upsample_factor is not None:
            self.upsample_factor = upsample_factor
        else:
            self.upsample_factor = (2,) * self.dims

        self.W_g = ConvPass(
            F_g,
            F_int,
            kernel_sizes=self.kernel_sizes,
            activation=None,
            padding="same",
            batch_norm=self.batch_norm,
        )

        self.W_x = nn.Sequential(
            ConvPass(
                F_l,
                F_int,
                kernel_sizes=self.kernel_sizes,
                activation=None,
                padding="same",
                batch_norm=self.batch_norm,
            ),
            Downsample(upsample_factor),
        )

        self.psi = ConvPass(
            F_int,
            1,
            kernel_sizes=self.kernel_sizes,
            activation="Sigmoid",
            padding="same",
            batch_norm=self.batch_norm,
        )

        up_mode = {2: "bilinear", 3: "trilinear"}[self.dims]

        self.up = nn.Upsample(
            scale_factor=upsample_factor, mode=up_mode, align_corners=True
        )

        self.relu = nn.ReLU(inplace=True)

    def calculate_and_apply_padding(self, smaller_tensor, larger_tensor):
        """
        Calculate and apply symmetric padding to the smaller tensor to match the dimensions of the larger tensor.

        Args:
            smaller_tensor (Tensor): The tensor to be padded.
            larger_tensor (Tensor): The tensor whose dimensions the smaller tensor needs to match.
        Returns:
            Tensor: The padded smaller tensor with the same dimensions as the larger tensor.
        Raises:
            RuntimeError: If the tensors have different dimensions.
        Examples:
            >>> larger_tensor = torch.randn(1, 1, 128, 128, 128)
            >>> smaller_tensor = torch.randn(1, 1, 64, 64, 64)
            >>> attention_block = AttentionBlockModule(F_g=1, F_l=1, F_int=1, dims=3)
            >>> padded_tensor = attention_block.calculate_and_apply_padding(smaller_tensor, larger_tensor)
        Note:
            The tensors should have the same dimensions.
        """
        padding = []
        for i in range(2, 2 + self.dims):
            diff = larger_tensor.size(i) - smaller_tensor.size(i)
            padding.extend([diff // 2, diff - diff // 2])

        # Reverse padding to match the 'pad' function's expectation
        padding = padding[::-1]

        # Apply symmetric padding
        return nn.functional.pad(smaller_tensor, padding, mode="constant", value=0)

    def forward(self, g, x):
        """
        Forward pass of the Attention Block.

        Args:
            g (Tensor): The gating signal tensor.
            x (Tensor): The input feature tensor.
        Returns:
            Tensor: The output tensor with the same dimensions as the input feature tensor.
        Raises:
            RuntimeError: If the gating signal and input feature tensors have different dimensions.
        Examples:
            >>> g = torch.randn(1, 1, 128, 128, 128)
            >>> x = torch.randn(1, 1, 128, 128, 128)
            >>> attention_block = AttentionBlockModule(F_g=1, F_l=1, F_int=1, dims=3)
            >>> output = attention_block(g, x)
        Note:
            The gating signal and input feature tensors should have the same dimensions.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1 = self.calculate_and_apply_padding(g1, x1)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.up(psi)
        return x * psi
