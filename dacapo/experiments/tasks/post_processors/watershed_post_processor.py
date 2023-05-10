from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from .watershed_post_processor_parameters import WatershedPostProcessorParameters
from .post_processor import PostProcessor

from funlib.geometry import Coordinate
from funlib.segment.arrays import replace_values

import mwatershed as mws

from scipy.ndimage import measurements


import numpy as np

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
        pred_data = self.prediction_array[self.prediction_array.roi]
        affs = pred_data[: len(self.offsets)]
        segmentation = mws.agglom(
            affs - 0.5,
            self.offsets,
        )
        # filter fragments
        average_affs = np.mean(affs, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(segmentation)

        for fragment, mean in zip(
            fragment_ids, measurements.mean(average_affs, segmentation, fragment_ids)
        ):
            if mean < 0.5:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(filtered_fragments, dtype=segmentation.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(segmentation, filtered_fragments, replace, inplace=True)

        output_array[self.prediction_array.roi] = segmentation

        return output_array
