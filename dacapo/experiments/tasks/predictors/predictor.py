from funlib.geometry import Coordinate

import torch

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dacapo.experiments.architectures.architecture import Architecture
    from dacapo.experiments.model import Model
    from dacapo.experiments.datasplits.datasets.arrays import Array


class Predictor(ABC):
    @abstractmethod
    def create_model(self, architecture: "Architecture") -> "Model":
        """Given a training architecture, create a model for this predictor.
        This is usually done by appending extra layers to the output of the
        architecture to get the output tensor of the architecture into the
        right shape for this predictor."""
        pass

    @abstractmethod
    def create_target(self, gt: "Array") -> "Array":
        """Create the target array for training, given a ground-truth array.

        In general, the target is different from the ground-truth.

        The target is the array that is passed to the loss, and hence directly
        compared to the prediction (i.e., the output of the model). Depending
        on the predictor, the target can therefore be different from the
        ground-truth (e.g., an instance segmentation ground-truth would have to
        be converted into boundaries, if the model is predicting boundaries).

        By default, it is assumed that the spatial dimensions of ground-truth
        and target are the same.

        If your predictor needs more ground-truth context to create a target
        (e.g., because it predicts the distance to a boundary, up to a certain
        threshold), you can request a larger ground-truth region. See method
        ``gt_region_for_roi``.
        """
        pass

    @abstractmethod
    def create_weight(
        self,
        gt: "Array",
        target: "Array",
        mask: "Array",
        moving_class_counts: Any,
    ) -> tuple["Array", Any]:
        """Create the weight array for training, given a ground-truth and
        associated target array.
        """
        pass

    @property
    @abstractmethod
    def output_array_type(self):
        pass

    def gt_region_for_roi(self, target_spec):
        """Report how much spatial context this predictor needs to generate a
        target for the given ROI. By default, uses the same ROI.

        Overwrite this method to request ground-truth in a larger ROI, as
        needed."""
        return target_spec

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
        return Coordinate((0,) * gt_voxel_size.dims)
