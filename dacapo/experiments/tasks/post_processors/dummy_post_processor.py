from .dummy_post_processor_parameters import DummyPostProcessorParameters
from .post_processor import PostProcessor
import numpy as np
import zarr


class DummyPostProcessor(PostProcessor):

    def __init__(self, detection_threshold):
        self.detection_threshold = detection_threshold

    def enumerate_parameters(self):
        """Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``."""

        for i, min_size in enumerate(range(1, 11)):
            yield DummyPostProcessorParameters(id=i, min_size=min_size)

    def set_prediction(self, prediction_array):
        pass

    def process(
            self,
            parameters,
            output_array_identifier):

        # store some dummy data
        f = zarr.open(str(output_array_identifier.container), 'a')
        f[output_array_identifier.dataset] = np.ones((10, 10, 10)) * parameters.min_size
