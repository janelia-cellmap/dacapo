from abc import ABC, abstractmethod
from funlib.geometry import Coordinate
from funlib.persistence import Array

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
        PostProcessorParameters,
    )
    from dacapo.store.local_array_store import LocalArrayIdentifier


class PostProcessor(ABC):
    """
    Base class of all post-processors.

    A post-processor takes a model's prediction and converts it into the final
    output (e.g., per-voxel class probabilities into a semantic segmentation). A
    post-processor can have multiple parameters, which can be enumerated using
    the `enumerate_parameters` method. The `process` method takes a set of
    parameters and applies the post-processing to the prediction.

    Attributes:
        prediction_array_identifier: The identifier of the array containing the
            model's prediction.
    Methods:
        enumerate_parameters: Enumerate all possible parameters of this
            post-processor.
        set_prediction: Set the prediction array identifier.
        process: Convert predictions into the final output.
    Note:
        This class is abstract. Subclasses must implement the abstract methods. Once
        created, the values of its attributes cannot be changed.
    """

    @abstractmethod
    def enumerate_parameters(self) -> Iterable["PostProcessorParameters"]:
        """
        Enumerate all possible parameters of this post-processor.

        Returns:
            An iterable of `PostProcessorParameters` instances.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = MyPostProcessor()
            >>> for parameters in post_processor.enumerate_parameters():
            ...     print(parameters)
            MyPostProcessorParameters(param1=0.0, param2=0.0)
            MyPostProcessorParameters(param1=0.0, param2=1.0)
            MyPostProcessorParameters(param1=1.0, param2=0.0)
            MyPostProcessorParameters(param1=1.0, param2=1.0)
        Note:
            This method must be implemented in the subclass. It should return an
            iterable of `PostProcessorParameters` instances.

        """
        pass

    @abstractmethod
    def set_prediction(
        self, prediction_array_identifier: "LocalArrayIdentifier"
    ) -> None:
        """
        Set the prediction array identifier.

        Args:
            prediction_array_identifier: The identifier of the array containing
                the model's prediction.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = MyPostProcessor()
            >>> post_processor.set_prediction("prediction")
        Note:
            This method must be implemented in the subclass. It should set the
            `prediction_array_identifier` attribute.
        """
        pass

    @abstractmethod
    def process(
        self,
        parameters: "PostProcessorParameters",
        output_array_identifier: "LocalArrayIdentifier",
        num_workers: int = 16,
        chunk_size: Coordinate = Coordinate((64, 64, 64)),
    ) -> Array:
        """
        Convert predictions into the final output.

        Args:
            parameters: The parameters of the post-processor.
            output_array_identifier: The identifier of the array to store the
                output.
            num_workers: The number of workers to use.
            chunk_size: The size of the chunks to process.
        Returns:
            The output array.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = MyPostProcessor()
            >>> post_processor.set_prediction("prediction")
            >>> parameters = MyPostProcessorParameters(param1=0.0, param2=0.0)
            >>> output = post_processor.process(parameters, "output")
        Note:
            This method must be implemented in the subclass. It should convert the
            model's prediction into the final output.
        """
        pass
