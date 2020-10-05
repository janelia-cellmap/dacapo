from dacapo.evaluate import evaluate_affs
from dacapo.models import Model
from dacapo.tasks.post_processors import Watershed
import gunpowder as gp
import torch

import lsd

import time


class LSD(Model):
    def __init__(self, data, model, post_processor=None):

        self.dims = data.raw.spatial_dims

        super(LSD, self).__init__(model.input_shape, model.fmaps_in, self.dims)

        conv = torch.nn.Conv3d
        num_shape_descriptors = 10
        lsd = [
            model,
            conv(model.fmaps_out, num_shape_descriptors, (1,) * num_shape_descriptors),
            torch.nn.Sigmoid(),  # Maybe
        ]

        self.lsd = torch.nn.Sequential(*lsd)
        self.prediction_channels = num_shape_descriptors
        self.target_channels = num_shape_descriptors

    def add_target(self, gt: gp.ArrayKey, target: gp.ArrayKey):

        return (
            lsd.gp.AddLocalShapeDescriptor(gt, target)
            # ensure affs are float
            # gp.Normalize(target, factor=1.0)
        )

    def forward(self, x):
        lsd = self.lsd(x)
        return lsd
