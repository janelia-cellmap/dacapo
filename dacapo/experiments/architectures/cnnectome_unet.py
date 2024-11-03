from .architecture import Architecture

import torch
import torch.nn as nn

import math


class CNNectomeUNet(Architecture):
    

    def __init__(self, architecture_config):
        
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
        
        if self._eval_shape_increase is None:
            return super().eval_shape_increase
        return self._eval_shape_increase

    def module(self):
        
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
        
        for upsample_factor in self.upsample_factors:
            voxel_size = voxel_size / upsample_factor
        return voxel_size

    @property
    def input_shape(self):
        
        return self._input_shape

    @property
    def num_in_channels(self) -> int:
        
        return self.fmaps_in

    @property
    def num_out_channels(self) -> int:
        
        return self.fmaps_out

    def forward(self, x):
        
        return self.unet(x)


class CNNectomeUNetModule(torch.nn.Module):
    

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
        
        y = self.rec_forward(self.num_levels - 1, x)

        if self.num_heads == 1:
            return y[0]

        return y


class ConvPass(torch.nn.Module):
    

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        activation,
        padding="valid",
        batch_norm=True,
    ):
        
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
        
        return self.conv_pass(x)


class Downsample(torch.nn.Module):
    

    def __init__(self, downsample_factor):
        
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
        
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (x.size(), self.downsample_factor, self.dims - d)
                )

        return self.down(x)


class Upsample(torch.nn.Module):
    

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
        

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, g_out, f_left=None):
        
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
    

    def __init__(self, F_g, F_l, F_int, dims, upsample_factor=None, batch_norm=True):
        

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
        
        padding = []
        for i in range(2, 2 + self.dims):
            diff = larger_tensor.size(i) - smaller_tensor.size(i)
            padding.extend([diff // 2, diff - diff // 2])

        # Reverse padding to match the 'pad' function's expectation
        padding = padding[::-1]

        # Apply symmetric padding
        return nn.functional.pad(smaller_tensor, padding, mode="constant", value=0)

    def forward(self, g, x):
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1 = self.calculate_and_apply_padding(g1, x1)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.up(psi)
        return x * psi
