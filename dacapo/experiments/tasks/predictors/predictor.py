from funlib.geometry import Coordinate

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from dacapo.experiments.architectures.architecture import Architecture
    from dacapo.experiments.model import Model
    from dacapo.experiments.datasplits.datasets.arrays import Array


class Predictor(ABC):
    @abstractmethod
    def create_model(self, architecture: "Architecture") -> "Model":
        pass

    @abstractmethod
    def create_target(self, gt: "Array") -> "Array":
        pass

    @abstractmethod
    def create_weight(
        self,
        gt: "Array",
        target: "Array",
        mask: "Array",
        moving_class_counts: Any,
    ) -> Tuple["Array", Any]:
        pass

    @property
    @abstractmethod
    def output_array_type(self):
        pass

    def gt_region_for_roi(self, target_spec):
        return target_spec

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
        return Coordinate((0,) * gt_voxel_size.dims)
