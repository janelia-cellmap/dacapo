from typing import Iterable
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from .CARE_post_processor_parameters import CAREPostProcessorParameters
from .post_processor import PostProcessor
import numpy as np
import zarr

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from dacapo.store.local_array_store import LocalArrayIdentifier
    from dacapo.experiments.tasks.post_processors import PostProcessorParameters


class CAREPostProcessor(PostProcessor):
    def __init__(self) -> None:
        super().__init__()

    def enumerate_parameters(self) -> Iterable[CAREPostProcessorParameters]:
        """Enumerate all possible parameters of this post-processor."""

        yield CAREPostProcessorParameters(id=1)

    def set_prediction(
        self, prediction_array_identifier: "LocalArrayIdentifier"
    ):  # TODO
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(
        self,
        parameters: "PostProcessorParameters",
        output_array_identifier: "LocalArrayIdentifier",
    ) -> ZarrArray:

        output_array: ZarrArray = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            self.prediction_array.axes,
            self.prediction_array.roi,
            self.prediction_array.num_channels,
            self.prediction_array.voxel_size,
            np.uint8,
        )

        return output_array
