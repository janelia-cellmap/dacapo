from cellpose.resnet_torch import CPnet
# TODO 
class CellposeUnet(Architecture):
    nbase = [in_chan, 32, 64, 128, 256]
    nout = 4
    sz  = 3
    self.net = CPnet(nbase, nout, sz, mkldnn=False,
           conv_3D=True, max_pool=True,
            diam_mean=30.)
    