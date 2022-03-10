from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from .watershed_post_processor_parameters import WatershedPostProcessorParameters
from .post_processor import PostProcessor

from funlib.geometry import Coordinate

try:
    from affogato.segmentation import (
        compute_mws_segmentation_from_affinities,
    )
except ImportError:

    def compute_mws_segmentation_from_affinities(*args, **kwargs):
        raise ImportError(
            "Affogato is not installed. Please install via "
            "`conda install -c conda-forge affogato`"
        )


import numpy as np
import zarr

from typing import List


class WatershedPostProcessor(PostProcessor):
    def __init__(self, offsets: List[Coordinate]):
        self.offsets = offsets

    def enumerate_parameters(self):
        """Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``."""

        for i, bias in enumerate([0.1, 0.5, 0.9]):
            yield WatershedPostProcessorParameters(id=i, bias=bias)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(self, parameters, output_array_identifier):
        output_array = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            [axis for axis in self.prediction_array.axes if axis != "c"],
            self.prediction_array.roi,
            None,
            self.prediction_array.voxel_size,
            np.uint64,
        )
        # if a previous segmentation is provided, it must have a "grid graph"
        # in its metadata.

        segmentation = compute_mws_segmentation_from_affinities(
            self.prediction_array[self.prediction_array.roi][: len(self.offsets)],
            self.offsets,
            beta_parameter=parameters.bias,
        )

        output_array[self.prediction_array.roi] = segmentation

        return output_array
