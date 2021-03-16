from dacapo.gp import AddDistance
from .predictor_abc import PredictorABC

import gunpowder as gp
import torch

from typing import List, Optional
from enum import Enum

import attr

# Define conv layers for different dimension counts
CONV_LAYERS = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


class WeightingOption(Enum):
    BALANCE_LABELS = "balance_labels"
    DISTANCE = "distance"


@attr.s
class Affinities(PredictorABC):
    name: str = attr.ib(
        metadata={
            "help_text": "This name is used to differentiate between predictors."
        }
    )

    neighborhood: List[List[int]] = attr.ib()
    weighting_type: WeightingOption = attr.ib()

    # attributes that can be read from other configurable classes
    fmaps_in: Optional[int] = attr.ib(default=None)  # from model
    dims: Optional[int] = attr.ib(default=None)  # from data

    def head(self, fmaps_in: int):

        assert self.fmaps_in is None or self.fmaps_in == fmaps_in
        self.fmaps_in = fmaps_in

        conv = CONV_LAYERS[self.dims]
        affs = [
            conv(fmaps_in, len(self.neighborhood), (1,) * len(self.neighborhood)),
            torch.nn.Sigmoid(),
        ]
        return torch.nn.Sequential(*affs)

    def add_target(self, gt, target, weights=None, mask=None, target_voxel_size=None):
        # Get a node that take gt and generates target
        # If adding targets requires padding, make sure that is also returned.
        target_node = gp.AddAffinities(
            affinity_neighborhood=self.neighborhood, labels=gt, affinities=target
        )
        if weights is not None:
            if self.weighting_type == WeightingOption.BALANCE_LABELS:
                weights_node = gp.BalanceLabels(target, weights, mask=mask)
            elif self.weighting_type == WeightingOption.DISTANCE:
                weights_node = AddDistance(gt, weights, mask)
            else:
                raise Exception(f"{self.weighting_type} not a valid option")
        else:
            weights_node = None
        padding = (
            gp.Coordinate(
                max([0] + [a[d] for a in self.neighborhood])
                for d in range(len(self.neighborhood[0]))
            )
            * target_voxel_size
        )

        return (
            target_node,
            weights_node,
            padding
            # TODO: Fix Error: Found dtype Byte but expected Float
            # This can occur when backpropogating through MSE where
            # the predictions are floats but the targets are uint8
        )


# PREFACTOR
"""
class Affinities(Model):
    def __init__(self, data, model, post_processor=None, weighting_type=None):

        assert data.gt.num_classes == 0, (
            f"Your GT has {data.gt.num_classes} classes, don't know how "
            "to get affinities out of that."
        )
        if weighting_type is not None:
            self.weighting_type = weighting_type
        else:
            self.weighting_type = "balance_labels"

        self.voxel_size = data.raw.voxel_size

        self.dims = data.raw.spatial_dims

        super(Affinities, self).__init__(model.output_shape, model.fmaps_out, self.dims)

        if self.dims == 2:
            self.neighborhood = [(0, 1, 0), (0, 0, 1)]
        elif self.dims == 3:
            self.neighborhood = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        else:
            raise RuntimeError("Affinities other than 2D/3D not implemented")

        conv = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}[self.dims]
        affs = [
            conv(model.fmaps_out, self.dims, (1,) * self.dims),
            torch.nn.Sigmoid(),
        ]

        self.affs = torch.nn.Sequential(*affs)
        self.prediction_channels = self.dims
        self.target_channels = self.dims
        if post_processor is None:
            self.post_processor = Watershed()
        else:
            self.post_processor = post_processor

        self.output_channels = self.dims

    def add_target(self, gt, target, weights=None, mask=None):
        target_node = gp.AddAffinities(
            affinity_neighborhood=self.neighborhood, labels=gt, affinities=target
        )
        if weights is not None:
            if self.weighting_type == "balance_labels":
                weights_node = gp.BalanceLabels(target, weights, mask=mask)
            elif self.weighting_type == "distance":
                weights_node = AddDistance(gt, weights, mask)
            else:
                raise Exception(f"{self.weighting_type} not a valid option")
        else:
            weights_node = None
        padding = (
            gp.Coordinate(
                max([0] + [a[d] for a in self.neighborhood])
                for d in range(len(self.neighborhood[0]))
            )
            * self.voxel_size
        )

        return (
            target_node,
            weights_node,
            padding
            # TODO: Fix Error: Found dtype Byte but expected Float
            # This can occur when backpropogating through MSE where
            # the predictions are floats but the targets are uint8
        )

    def forward(self, x):
        affs = self.affs(x)
        return affs

    def evaluate(self, predictions, gt, targets, return_results):
        reconstructions = self.post_processor.enumerate(predictions)

        for parameters, reconstruction in reconstructions:

            # This could be factored out.
            # keep evaulate as a super class method
            # over-write evaluate_reconstruction
            ret = evaluate_detection(reconstruction, gt, return_results=return_results)

            if return_results:
                scores, results = ret
                yield parameters, scores, results
            else:
                yield parameters, ret
"""