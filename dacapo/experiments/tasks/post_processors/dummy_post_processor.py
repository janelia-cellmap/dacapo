from .dummy_post_processor_parameters import DummyPostProcessorParameters
from .post_processor import PostProcessor

import numpy as np
import zarr

from typing import Iterable


class DummyPostProcessor(PostProcessor):
    

    def __init__(self, detection_threshold: float):
        
        self.detection_threshold = detection_threshold

    def enumerate_parameters(self) -> Iterable[DummyPostProcessorParameters]:
        

        for i, min_size in enumerate(range(1, 11)):
            yield DummyPostProcessorParameters(id=i, min_size=min_size)

    def set_prediction(self, prediction_array_identifier):
        
        pass

    def process(self, parameters, output_array_identifier, *args, **kwargs):
        
        # store some dummy data
        f = zarr.open(str(output_array_identifier.container), "a")
        f[output_array_identifier.dataset] = np.ones((10, 10, 10)) * parameters.min_size
