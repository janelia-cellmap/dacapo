from dacapo.models import Model
import gunpowder as gp
import torch

import lsd


class LSD(Model):
    def __init__(self, data, model, post_processor=None):

        self.dims = data.raw.spatial_dims

        num_shape_descriptors = 10
        super(LSD, self).__init__(model.output_shape, model.fmaps_out, num_shape_descriptors)

        conv = torch.nn.Conv3d
        lsd = [
            conv(model.fmaps_out, num_shape_descriptors, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.lsd = torch.nn.Sequential(*lsd)

    def add_target(self, gt: gp.ArrayKey, target: gp.ArrayKey):

        return lsd.gp.AddLocalShapeDescriptor(gt, target, sigma=40)

    def forward(self, x):
        lsd = self.lsd(x)
        return lsd
