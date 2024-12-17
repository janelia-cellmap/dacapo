UNet Models
===========

This section explains how to configure and use UNet models in DaCapo. Several configurations for different types of UNet architectures are demonstrated below.

Overview
--------

UNet is a popular architecture for image segmentation tasks, particularly in biomedical imaging. DaCapo provides support for configuring various types of UNet models with customizable parameters.

Examples
--------

Here are some examples of UNet configurations:

1. **Upsample UNet**

.. code-block:: python

    from dacapo.experiments.architectures import CNNectomeUNetConfig
    from funlib.geometry import Coordinate

    architecture_config = CNNectomeUNetConfig(
        name="upsample_unet",
        input_shape=Coordinate(216, 216, 216),
        eval_shape_increase=Coordinate(72, 72, 72),
        fmaps_in=1,
        num_fmaps=12,
        fmaps_out=72,
        fmap_inc_factor=6,
        downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
        constant_upsample=True,
        upsample_factors=[(2, 2, 2)],
    )

2. **Yoshi UNet**

.. code-block:: python

    yoshi_unet_config = CNNectomeUNetConfig(
        name="yoshi-unet",
        input_shape=Coordinate(188, 188, 188),
        eval_shape_increase=Coordinate(72, 72, 72),
        fmaps_in=1,
        num_fmaps=12,
        fmaps_out=72,
        fmap_inc_factor=6,
        downsample_factors=[(2, 2, 2), (2, 2, 2), (2, 2, 2)],
        constant_upsample=True,
        upsample_factors=[],
    )

3. **Attention Upsample UNet**

.. code-block:: python

    attention_upsample_config = CNNectomeUNetConfig(
        name="attention-upsample-unet",
        input_shape=Coordinate(216, 216, 216),
        eval_shape_increase=Coordinate(72, 72, 72),
        fmaps_in=1,
        num_fmaps=12,
        fmaps_out=72,
        fmap_inc_factor=6,
        downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
        constant_upsample=True,
        upsample_factors=[(2, 2, 2)],
        use_attention=True,
    )

4. **2D UNet**

.. code-block:: python

    architecture_config = CNNectomeUNetConfig(
        name="2d_unet",
        input_shape=(2, 132, 132),
        eval_shape_increase=(8, 32, 32),
        fmaps_in=2,
        num_fmaps=8,
        fmaps_out=8,
        fmap_inc_factor=2,
        downsample_factors=[(1, 4, 4), (1, 4, 4)],
        kernel_size_down=[[(1, 3, 3)] * 2] * 3,
        kernel_size_up=[[(1, 3, 3)] * 2] * 2,
        constant_upsample=True,
        padding="valid",
    )

5. **UNet with Batch Normalization**

.. code-block:: python

    architecture_config = CNNectomeUNetConfig(
        name="unet_norm",
        input_shape=Coordinate(216, 216, 216),
        eval_shape_increase=Coordinate(72, 72, 72),
        fmaps_in=1,
        num_fmaps=2,
        fmaps_out=2,
        fmap_inc_factor=2,
        downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
        constant_upsample=True,
        upsample_factors=[],
        batch_norm=False,
    )

Configuration Parameters
------------------------

- **name**: A unique identifier for the configuration.
- **input_shape**: The shape of the input data.
- **eval_shape_increase**: Increase in shape during evaluation.
- **fmaps_in**: Number of input feature maps.
- **num_fmaps**: Number of feature maps in the first layer.
- **fmaps_out**: Number of output feature maps.
- **fmap_inc_factor**: Factor by which feature maps increase in each layer.
- **downsample_factors**: Factors by which the input is downsampled at each layer.
- **upsample_factors**: Factors by which the input is upsampled at each layer.
- **constant_upsample**: Whether to use constant upsampling.
- **use_attention**: Whether to use attention mechanisms.
- **batch_norm**: Whether to use batch normalization.
- **padding**: Padding mode for convolutional layers.

This page should serve as a reference for configuring UNet models in DaCapo. Adjust the parameters as per your dataset and task requirements.
