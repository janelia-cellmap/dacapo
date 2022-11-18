from typing import Iterable
from dacapo.experiments.datasplits.datasets.arrays import IntensitiesArray

from .CARE_post_processor_parameters import CAREPostProcessorParameters
from .post_processor import PostProcessor
import numpy as np
import zarr

from typing import Iterable


class CAREPostProcessor(PostProcessor):
    def __init__(self, detection_threshold: float):
        self.detection_threshold = detection_threshold

    def enumerate_parameters(self) -> Iterable[CAREPostProcessorParameters]:
        """Enumerate all possible parameters of this post-processor."""

        yield CAREPostProcessorParameters(id=1)

    def set_prediction(self, prediction_array_identifier: "LocalArrayIdentifier"):
        self.prediction_array = IntensitiesArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(
        self,
        parameters: "PostProcessorParameters",
        output_array_identifier: "IntensitiesArrayIdentifier",
        ) -> IntensitiesArray:
        # TODO: Investigate Liskov substitution princple and whether it is a problem here
        # OOP theory states the super class should always be replaceable with its subclasses
        # meaning the input arguments to methods on the subclass can only be more loosely
        # constrained and the outputs can only be more highly constrained. In this case
        # we know our parameters will be a `ThresholdPostProcessorParameters` class,
        # which is more specific than the `PostProcessorParameters` parent class.
        # Seems unrelated to me since just because all `PostProcessors` use some
        # `PostProcessorParameters` doesn't mean they can use any `PostProcessorParameters`
        # so our subclasses aren't directly replaceable anyway.
        # Might be missing something since I only did a quick google, leaving this here
        # for me or someone else to investigate further in the future.
        output_array = IntensitiesArray.create_from_array_identifier(
            output_array_identifier,
            self.prediction_array.axes,
            self.prediction_array.roi,
            self.prediction_array.num_channels,
            self.prediction_array.voxel_size,
            np.uint8,
        )

        output_array[self.prediction_array.roi] = (
            self.prediction_array[self.prediction_array.roi] > 0
        ).astype(np.uint8)

        return output_array
