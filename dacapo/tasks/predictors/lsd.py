from .predictor_abc import PredictorABC

import gunpowder as gp
import torch
import lsd
import attr

from typing import Optional
from enum import Enum


# Define conv layers for different dimension counts
CONV_LAYERS = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


@attr.s
class LSD(PredictorABC):
    name: str = attr.ib(
        metadata={"help_text": "This name is used to differentiate between predictors."}
    )

    sigma: float = attr.ib(
        metadata={
            "help_text": "The size of the gaussian (in world units) "
            "used to calculate shape descriptors."
        }
    )
    num_shape_descriptors: int = attr.ib(
        default=10,
        metadata={
            "help_text": "The number of shape descriptors. Currently 10 is the only option."
        },
    )

    # attributes that can be read from other configurable classes
    fmaps_in: Optional[int] = attr.ib(
        default=None,
        metadata={"help_text": "The number of feature maps provided by the Model."},
    )
    dims: Optional[int] = attr.ib(
        default=None,
        metadata={"help_text": "The dimensionality of your data."},
    )

    def head(self, fmaps_in: int):
        conv = torch.nn.Conv3d
        lsd = [
            conv(fmaps_in, self.num_shape_descriptors, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.lsd = torch.nn.Sequential(*lsd)

    def add_target(self, gt, target, weights=None, mask=None):

        target_node = lsd.gp.AddLocalShapeDescriptor(
            gt, target, mask=weights, sigma=self.sigma
        )
        weights_node = None

        return (target_node, weights_node)


# PREFACTOR
"""
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

    def add_target(self, gt, target, weights=None, mask=None):

        extra_context = gp.Coordinate(tuple(s * 3 for s in (self.sigma,) * 3))
        return (
            lsd.gp.AddLocalShapeDescriptor(gt, target, mask=weights, sigma=self.sigma),
            True,
            extra_context,
        )

    def forward(self, x):
        lsd = self.lsd(x)
        return lsd
"""
