from .predictor import Predictor
from dacapo.experiments import Model
from dacapo.experiments.arraytypes import IntensitiesArray
from dacapo.experiments.datasplits.datasets.arrays import NumpyArray, ZarrArray
from dacapo.experiments.architecutres import CNNectomeUNet, NLayerDiscriminator

from funlib.geometry import Coordinate

import numpy as np
import torch


class Pix2PixPredictor(Predictor):
    def __init__(self, num_channels, dims):
        self.num_channels = num_channels
        self.dims = dims

    def create_model(self, g_architecture, d_architecture=None, is_train=True):
        generator: CNNectomeUNet = CNNectomeUNet(
                                  input_shape=g_architecture.input_shape
                                  fmaps_out=g_architecture.fmaps_out
                                  fmaps_in=g_architecture.fmaps_in,
                                  num_fmaps=g_architecture.num_fmaps,
                                  fmap_inc_factor=g_architecture.fmap_inc_factor,
                                  downsample_factors=g_architecture.downsample_factors,
                                  constant_upsample=g_architecture.constant_upsample,
                                  padding=g_architecture.padding
        )   
        
        if self.dims == 2:
            head = torch.nn.Conv2d(
                g_architecture.num_out_channels, self.num_channels, kernel_size=1
            )
        elif self.dims == 3:
            head = torch.nn.Conv3d(
                g_architecture.num_out_channels, self.num_channels, kernel_size=1
            )
        else:
            raise NotImplementedError(
                f"CAREPredictor not implemented for {self.dims} dimensions"
            )

        if is_train:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            try:
                discriminator: NLayerDiscriminator = NLayerDiscriminator(d_architecture.input_nc, 
                                                    d_architecture.ngf,
                                                    d_architecture.n_layers,
                                                    d_architecture.norm_layer)
            except Exception as e:
                return Model(g_architecture, head), Model(d_architecture, head)

        return Model(g_architecture, head)

    def create_target(self, gt):
        return gt

    def create_weight(self, gt, target=None, mask=None):
        if mask is None:
            # array of ones
            return NumpyArray.from_np_array(
                np.ones(gt.data.shape),
                gt.roi,
                gt.voxel_size,
                gt.axes,
            )
        else:
            return mask

    @property
    def output_array_type(self):
        return IntensitiesArray({"channels": {n: str(n) for n in range(self.num_channels)}}, min=0., max=1.)


