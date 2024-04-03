from cellpose.resnet_torch import CPnet
from .architecture import Architecture

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
        self._input_shape = architecture_config.input_shape
        self.unet = CPnet(
            architecture_config.nbase,
            architecture_config.nout,
            architecture_config.sz,
            architecture_config.mkldnn,
            architecture_config.conv_3D,
            architecture_config.max_pool,
            architecture_config.diam_mean,
        )

    def forward(self, x):
        return self.unet(x)


