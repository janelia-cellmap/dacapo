```python
"""Implementation of CNNectome U-Net architecture modules.

This script defines the main classes that make up our CNNectome U-Net architecture.
It contains three classes: CNNectomeUNet, CNNectomeUNetModule, AttentionBlockModule

Attributes:
     CNNectomeUNet: implements the general architecture of the model
     CNNectomeUNetModule: implements the individual modules that make up the network
     AttentionBlockModule: implements the attention mechanism applied in the model
     
Classes:
    CNNectomeUNet: Defines the high level structure of the CNNectome U-Net model. 
        It includes techniques such as convolution, pooling and upscaling for its 
        operation. It extends the functionality of the "Architecture" PyTorch Module.

    CNNectomeUNetModule: Corresponds to the individual modules that make up the 
        network. It defines the relevant operations that the network undergoes including
        convolutions, activation functions and upsampling.

    ConvPass: Represents a single convolution pass within the network. A ConvPass 
        consists of a convolution operation, followed by an activation function.  

    Downsample: Module used to apply a max-pooling operation for down-sampling the input.

    Upsample: A module that upsamples an input by a given factor using a specified mode (either "transposed_conv" or "nearest").

    AttentionBlockModule: Implements the attention mechanism. It consists of convolutional, 
        up-sampling, activation, and padding operations to compute and apply the attention 
        mechanism to the input tensor.
"""
```