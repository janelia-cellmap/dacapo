
import torch
import torch.nn as nn
from .cnnectome_unet import ConvPass,Downsample,Upsample

class AttentionBlockModule(nn.Module):
    def __init__(self, F_g, F_l, F_int, dims):
        """Attention Block Module::

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

    Args:
    F_g (int): The number of feature channels in the gating signal (g). 
               This is the input channel dimension for the W_g convolutional layer.

    F_l (int): The number of feature channels in the input features (x). 
               This is the input channel dimension for the W_x convolutional layer.

    F_int (int): The number of intermediate feature channels. 
                 This represents the output channel dimension of the W_g and W_x convolutional layers 
                 and the input channel dimension for the psi layer. Typically, F_int is smaller 
                 than F_g and F_l, as it serves to compress the feature representations before 
                 applying the attention mechanism.

    The AttentionBlock uses two separate pathways to process 'g' and 'x', combines them,
    and applies a sigmoid activation to generate an attention map. This map is then used 
    to scale the input features 'x', resulting in an output that focuses on important 
    features as dictated by the gating signal 'g'.
           
           """
        
        
        super(AttentionBlockModule, self).__init__()
        self.dims = dims
        self.kernel_sizes = [(1,) * self.dims, (1,) * self.dims]
        print("kernel_sizes:",self.kernel_sizes)

        self.W_g = ConvPass(F_g, F_int, kernel_sizes=self.kernel_sizes, activation=None,padding="same")
        
        self.W_x = nn.Sequential(
            ConvPass(F_l, F_int, kernel_sizes=self.kernel_sizes, activation=None,padding="same"),
            Downsample((2,)*self.dims)
        )

        self.psi = ConvPass(F_int, 1, kernel_sizes=self.kernel_sizes, activation="Sigmoid",padding="same")

        up_mode = {2: 'bilinear', 3: 'trilinear'}[self.dims]

        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.up(psi)
        return x * psi