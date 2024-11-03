from abc import ABC, abstractmethod
from funlib.geometry import Coordinate

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
        PostProcessorParameters,
    )
    from dacapo.experiments.datasplits.datasets.arrays import Array
    from dacapo.store.local_array_store import LocalArrayIdentifier


class PostProcessor(ABC):
    

    @abstractmethod
    def enumerate_parameters(self) -> Iterable["PostProcessorParameters"]:
        
        pass

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
        num_workers: int = 16,
        chunk_size: Coordinate = Coordinate((64, 64, 64)),
    ) -> "Array":
        
        pass
