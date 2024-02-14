from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from .argmax_post_processor_parameters import ArgmaxPostProcessorParameters
from .post_processor import PostProcessor
import numpy as np


class ArgmaxPostProcessor(PostProcessor):
    def __init__(self):
        pass

    def enumerate_parameters(self):
        """Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``."""

        yield ArgmaxPostProcessorParameters(id=1)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(self, parameters, output_array_identifier):
        output_array = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            [dim for dim in self.prediction_array.axes if dim != "c"],
            self.prediction_array.roi,
            None,
            self.prediction_array.voxel_size,
            np.uint8,
        )

        output_array[self.prediction_array.roi] = np.argmax(
            self.prediction_array[self.prediction_array.roi],
            axis=self.prediction_array.axes.index("c"),
        )
        return output_array
