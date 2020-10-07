from dacapo.models import Model
import gunpowder as gp
import torch

import lsd


class LSD(Model):
    def __init__(self, data, model, post_processor=None):

        self.dims = data.raw.spatial_dims

        super(LSD, self).__init__(model.input_shape, model.fmaps_in, self.dims)

        conv = torch.nn.Conv3d
        num_shape_descriptors = 10
        lsd = [
            model,
            conv(model.fmaps_out, num_shape_descriptors, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.lsd = torch.nn.Sequential(*lsd)

    def add_target(self, gt: gp.ArrayKey, target: gp.ArrayKey):

        return lsd.gp.AddLocalShapeDescriptor(gt, target)

    def forward(self, x):
        lsd = self.lsd(x)
        return lsd
