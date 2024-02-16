The provided python code already contains descriptive comments and does not need any further docstrings. However, if you specifically want to add docstrings, here's an example for CNNectomeUNetConfig class:

```python
@attr.s
class CNNectomeUNetConfig(ArchitectureConfig):
    """
    Class responsible for configuring the CNNectomeUNet based on
    https://github.com/saalfeldlab/CNNectome/blob/master/CNNectome/networks/unet_class.py

    Includes support for super resolution via the upsampling factors.

    Args:
        input_shape (Coordinate): The shape of the data passed into the network during training.
        
        fmaps_out (int): The number of channels produced by your architecture.

        fmaps_in (int): The number of channels expected from the raw data.

        num_fmaps (int): The number of feature maps in the top level of the UNet.
        
        fmap_inc_factor (int): The multiplication factor for the number of feature maps for each 
        level of the UNet.
        
        downsample_factors (List[Coordinate]): The factors to downsample the feature maps along each axis per layer.
        
        kernel_size_down (Optional[List[Coordinate]]): The size of the convolutional kernels used before downsampling in each layer.
        
        kernel_size_up (Optional[List[Coordinate]]): The size of the convolutional kernels used before upsampling in each layer.
        
        _eval_shape_increase (Optional[Coordinate]): The amount by which to increase the input size when just 
        prediction rather than training. It is generally possible to significantly 
        increase the input size since we don't have the memory constraints of the 
        gradients, the optimizer and the batch size.

        upsample_factors (Optional[List[Coordinate]]): The amount by which to upsample the output of the UNet.

        constant_upsample (bool): Whether to use a transpose convolution or simply copy voxels to upsample.

        padding (str): The padding to use in convolution operations.

        use_attention (bool): Whether to use attention blocks in the UNet. This is supported for 2D and  3D.
    """
```