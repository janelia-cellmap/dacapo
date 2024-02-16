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
    """Base class of all post-processors.

    A post-processor takes a model's prediction and converts it into the final
    output (e.g., per-voxel class probabilities into a semantic segmentation).
    """

    @abstractmethod
    def enumerate_parameters(self) -> Iterable["PostProcessorParameters"]:
        """Enumerate all possible parameters of this post-processor."""

    @abstractmethod
    def set_prediction(
        self, prediction_array_identifier: "LocalArrayIdentifier"
    ) -> None:
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
        """Convert predictions into the final output."""
