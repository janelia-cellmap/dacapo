"""
This module provides an abstract base class for all post-processors in Dacapo Python Library. 

The process involves taking a model's prediction and converting it into the final
output (example, per-voxel class probabilities into a semantic segmentation).

Attributes:
    ABC (class): This is a helper class that has ABCMeta as its metaclass.
                  With this class, an abstract base class can be created by
                  deriving from ABC avoiding sometimes confusing meta-class usage.
    abstractmethod :A decorator indicating abstract methods.          
"""

from abc import ABC, abstractmethod
from dacapo.compute_context import ComputeContext, LocalTorch
from funlib.geometry import Coordinate

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
        PostProcessorParameters,
    )
    from dacapo.experiments.datasplits.datasets.arrays import Array
    from dacapo.store.local_array_store import LocalArrayIdentifier


class PostProcessor(ABC):
    """
    This is an abstract base class from which all other specific 
    post-processors should inherit.
    """

    @abstractmethod
    def enumerate_parameters(self) -> Iterable["PostProcessorParameters"]:
        """
        Abstract method for enumerating all possible parameters of post-processor. 
        """
        pass

    @abstractmethod
    def set_prediction(
        self, prediction_array_identifier: "LocalArrayIdentifier"
    ) -> None:
        """
        Abstract method for setting predictions.
        
        Args:
            prediction_array_identifier (LocalArrayIdentifier): Prediction array's identifier.
        """
        pass

    @abstractmethod
    def process(
        self,
        parameters: "PostProcessorParameters",
        output_array_identifier: "LocalArrayIdentifier",
        compute_context: ComputeContext | str = LocalTorch(),
        num_workers: int = 16,
        chunk_size: Coordinate = Coordinate((64, 64, 64)),
    ) -> "Array":
        """
        Abstract method for converting predictions into the final output.
        
        Args:
            parameters (PostProcessorParameters): Parameters for post processing.
            output_array_identifier (LocalArrayIdentifier): Output array's identifier.
            compute_context (ComputeContext or str): The context which the computations are to be done. Defaults to LocalTorch.
            num_workers (int, optional): Number of workers for the processing. Defaults to 16.
            chunk_size (Coordinate, optional): Size of the chunk for processing. Defaults to (64, 64, 64).
            
        Returns:
            Array: The processed array.
        """
        pass
