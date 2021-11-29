from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from .threshold_post_processor_parameters import ThresholdPostProcessorParameters
from .post_processor import PostProcessor
import numpy as np
import zarr


class ThresholdPostProcessor(PostProcessor):
    def __init__(self):
        pass

    def enumerate_parameters(self):
        """Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``."""

        yield ThresholdPostProcessorParameters(id=1)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(self, parameters, output_array_identifier):
        output_array = ZarrArray.create_from_array_identifier(
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