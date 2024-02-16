from .cellpose_post_processor_parameters import CellposePostProcessorParameters
from .post_processor import PostProcessor
from dacapo.store import LocalArrayIdentifier, ZarrArray
import numpy as np
import zarr

from typing import Iterable

from cellpose.dynamics import compute_masks

# https://github.com/MouseLand/cellpose/blob/54b14fe567d885db293280b9b8d68dc50703d219/cellpose/models.py#L608

class CellposePostProcessor(PostProcessor):
    def __init__(self, detection_threshold: float):
        self.detection_threshold = detection_threshold

    def enumerate_parameters(self) -> Iterable[CellposePostProcessorParameters]:
        """Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``."""

        for i, min_size in enumerate(range(1, 11)):
            yield CellposePostProcessorParameters(id=i, min_size=min_size)

    def set_prediction(self, prediction_array_identifier: LocalArrayIdentifier):
        self.prediction_array = ZarrArray.open_from_identifier(prediction_array_identifier)

    def process(self, parameters, output_array_identifier):
        # store some dummy data
        f = zarr.open(str(output_array_identifier.container), "a")
        f[output_array_identifier.dataset] = compute_masks(self.prediction_array.data[:-1]/5., self.prediction_array.data[-1], 
                      niter=200, cellprob_threshold=self.detection_threshold, do_3D=True, 
                      min_size=parameters.min_size)[0]
