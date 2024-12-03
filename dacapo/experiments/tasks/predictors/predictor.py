from funlib.geometry import Coordinate
from funlib.persistence import Array

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from dacapo.experiments.architectures.architecture import Architecture
    from dacapo.experiments.model import Model


class Predictor(ABC):
    """
    A predictor is a class that defines how to train a model to predict a
    certain output from a certain input.

    A predictor is responsible for creating the model, the target, the weight,
    and the output array type for a given training architecture.

    Methods:
        create_model(self, architecture: "Architecture") -> "Model": Given a training architecture, create a model for this predictor.
        create_target(self, gt: Array) -> Array: Create the target array for training, given a ground-truth array.
        create_weight(self, gt: Array, target: Array, mask: Array, moving_class_counts: Any) -> Tuple[Array, Any]: Create the weight array for training, given a ground-truth and associated target array.
        gt_region_for_roi(self, target_spec): Report how much spatial context this predictor needs to generate a target for the given ROI.
        padding(self, gt_voxel_size: Coordinate) -> Coordinate: Return the padding needed for the ground-truth array.
    Notes:
        This is a subclass of ABC.
    """

    @abstractmethod
    def create_model(self, architecture: "Architecture") -> "Model":
        """
        Given a training architecture, create a model for this predictor.
        This is usually done by appending extra layers to the output of the
        architecture to get the output tensor of the architecture into the
        right shape for this predictor.

        Args:
            architecture: The training architecture.
        Returns:
            The model.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_model(architecture)

        """
        pass

    @abstractmethod
    def create_target(self, gt: Array) -> Array:
        """
        Create the target array for training, given a ground-truth array.

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

        Args:
            gt: The ground-truth array.
        Returns:
            The target array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_target(gt)

        """
        pass

    @abstractmethod
    def create_weight(
        self,
        gt: Array,
        target: Array,
        mask: Array,
        moving_class_counts: Any,
    ) -> Tuple[Array, Any]:
        """
        Create the weight array for training, given a ground-truth and
        associated target array.

        Args:
            gt: The ground-truth array.
            target: The target array.
            mask: The mask array.
            moving_class_counts: The moving class counts.
        Returns:
            The weight array and the moving class counts.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.create_weight(gt, target, mask, moving_class_counts)

        """
        pass

    @property
    @abstractmethod
    def output_array_type(self):
        pass

    def gt_region_for_roi(self, target_spec):
        """
        Report how much spatial context this predictor needs to generate a
        target for the given ROI. By default, uses the same ROI.

        Overwrite this method to request ground-truth in a larger ROI, as
        needed.

        Args:
            target_spec: The ROI for which the target is requested.
        Returns:
            The ROI for which the ground-truth is requested.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.gt_region_for_roi(target_spec)


        """
        return target_spec

    def padding(self, gt_voxel_size: Coordinate) -> Coordinate:
        """
        Return the padding needed for the ground-truth array.

        Args:
            gt_voxel_size: The voxel size of the ground-truth array.
        Returns:
            The padding needed for the ground-truth array.
        Raises:
            NotImplementedError: This method is not implemented.
        Examples:
            >>> predictor.padding(gt_voxel_size)
        """
        return Coordinate((0,) * gt_voxel_size.dims)
