from funlib.geometry import Coordinate

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from dacapo.experiments.architectures.architecture import Architecture
    from dacapo.experiments.model import Model
    from dacapo.experiments.datasplits.datasets.arrays import Array


class Predictor(ABC):
    """
    An abstract class that serves as a blueprint for all the specific predictors.

    Attributes:
        output_array_type: A property which is expected to be implemented in subclasses.
    """

    @abstractmethod
    def create_model(self, architecture: "Architecture") -> "Model":
        """
        To create a model with the given training architecture.

        Args:
            architecture: An instance of class Architecture, to define training architecture for the model.

        Returns:
             An instance of class Model with the designed architecture.
        """
        pass

    @abstractmethod
    def create_target(self, gt: "Array") -> "Array":
        """
        Creates target for training based on ground-truth array.

        Args:
            gt: An instance of class Array, representing ground-truth values.

        Returns:
            Instance of Array class, representing target for training.
        """
        pass

    @abstractmethod
    def create_weight(
        self,
        gt: "Array",
        target: "Array",
        mask: "Array",
        moving_class_counts: Any,
    ) -> Tuple["Array", Any]:
        """
        Creates a weight array, using a ground-truth and an associated target array.

        Args:
            gt: Ground Truth array.
            target: Target array.
            mask: Associated mask array.
            moving_class_counts: Counts of moving classes.

        Returns:
            Tuple containing Array instance with weight array and any additional returned value.
        """
        pass

    @property
    @abstractmethod
    def output_array_type(self):
        """
        Subclasses should implement this method to define the type of array output by the predictor.
        """
        pass

    def gt_region_for_roi(self, target_spec):
        """
        Method to report the required spatial context to generate a target for the given ROI.

        Args:
            target_spec: Target specifications for which ground truth region is needed.

        Returns:
            Returns the same ROI by default, unless overridden.
        """

        return target_spec

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
        """
        Calculates and returns the padding size for an array.

        Args:
            gt_voxel_size: Ground Truth voxel size of type Coordinate.

        Returns:
            Coordinate having padding size.
        """

        return Coordinate((0,) * gt_voxel_size.dims)
