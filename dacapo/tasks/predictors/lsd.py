from dacapo.models import Model
import gunpowder as gp
import torch

import lsd


class LSD(Model):
    def __init__(self, data, model, post_processor=None, sigma=1):

        self.dims = data.raw.spatial_dims
        self.sigma = sigma

        num_shape_descriptors = 10
        super(LSD, self).__init__(
            model.output_shape, model.fmaps_out, num_shape_descriptors
        )

        conv = torch.nn.Conv3d
        lsd = [
            conv(model.fmaps_out, num_shape_descriptors, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.lsd = torch.nn.Sequential(*lsd)

        self.prediction_channels = num_shape_descriptors
        self.target_channels = num_shape_descriptors

        self.output_channels = 10

    def add_target(self, gt: gp.ArrayKey, target: gp.ArrayKey):

        extra_context = gp.Coordinate(tuple(s * 3 for s in (self.sigma,) * 3))
        return (
            lsd.gp.AddLocalShapeDescriptor(gt, target, sigma=self.sigma),
            extra_context,
        )

    def forward(self, x):
        lsd = self.lsd(x)
        return lsd
