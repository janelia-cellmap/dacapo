from cellpose.resnet_torch import CPnet
from .architecture import Architecture
from funlib.geometry import Coordinate

    # example
    # nout = 4
    # sz = 3
    # self.net = CPnet(
    #     nbase, nout, sz, mkldnn=False, conv_3D=True, max_pool=True, diam_mean=30.0
    # )
# currently the input channels are embedded in nbdase, but they should be passed as a separate parameternbase = [in_chan, 32, 64, 128, 256]
class CellposeUnet(Architecture):
    def __init__(self, architecture_config):
        super().__init__()
        self._input_shape = Coordinate(architecture_config.input_shape)
        self._nbase = architecture_config.nbase
        self._sz = self._input_shape.dims
        self._eval_shape_increase = Coordinate((0,) * self._sz)
        self._nout = architecture_config.nout
        print("conv_3D:",architecture_config.conv_3D)
        self.unet = CPnet(
            architecture_config.nbase,
            architecture_config.nout,
            self._sz,
            architecture_config.mkldnn,
            architecture_config.conv_3D,
            architecture_config.max_pool,
            architecture_config.diam_mean,
        )
        print(self.unet)

    def forward(self, data):
        """
        Forward pass of the CPnet model.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        if self.unet.mkldnn:
            data = data.to_mkldnn()
        T0 = self.unet.downsample(data)
        if self.unet.mkldnn:
            style = self.unet.make_style(T0[-1].to_dense())
        else:
            style = self.unet.make_style(T0[-1])
        # style0 = style
        if not self.unet.style_on:
            style = style * 0
        T1 = self.unet.upsample(style, T0, self.unet.mkldnn)
        # head layer 
        # T1 = self.unet.output(T1)
        if self.unet.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()
        return T1
    
    @property
    def input_shape(self):
        return self._input_shape

    @property
    def num_in_channels(self) -> int:
        return self._nbase[0]

    @property
    def num_out_channels(self) -> int:
        return self._nout
    
    @property
    def eval_shape_increase(self):
        return self._eval_shape_increase


