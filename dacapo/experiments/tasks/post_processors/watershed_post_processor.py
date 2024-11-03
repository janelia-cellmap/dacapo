from upath import UPath as Path
import dacapo.blockwise
from dacapo.blockwise.scheduler import segment_blockwise
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.utils.array_utils import to_ndarray, save_ndarray
from funlib.persistence import open_ds
import daisy
import mwatershed as mws

from .watershed_post_processor_parameters import WatershedPostProcessorParameters
from .post_processor import PostProcessor

from funlib.geometry import Coordinate, Roi


import numpy as np

from typing import List


class WatershedPostProcessor(PostProcessor):
    

    def __init__(self, offsets: List[Coordinate]):
        
        self.offsets = offsets

    def enumerate_parameters(self):
        

        for i, bias in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
            yield WatershedPostProcessorParameters(id=i, bias=bias)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array_identifier = prediction_array_identifier
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )
        

    def process(
        self,
        parameters: WatershedPostProcessorParameters,  # type: ignore[override]
        output_array_identifier: "LocalArrayIdentifier",
        num_workers: int = 16,
        block_size: Coordinate = Coordinate((256, 256, 256)),
    ):
        
        if self.prediction_array._daisy_array.chunk_shape is not None:
            block_size = Coordinate(
                self.prediction_array._daisy_array.chunk_shape[
                    -self.prediction_array.dims :
                ]
            )

        output_array = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            [axis for axis in self.prediction_array.axes if axis != "c"],
            self.prediction_array.roi,
            None,
            self.prediction_array.voxel_size,
            np.uint64,
            block_size * self.prediction_array.voxel_size,
        )
        input_array = open_ds(
            self.prediction_array_identifier.container.path,
            self.prediction_array_identifier.dataset,
        )

        data = to_ndarray(input_array, output_array.roi).astype(float)
        segmentation = mws.agglom(
            data - parameters.bias, offsets=self.offsets, randomized_strides=True
        )
        save_ndarray(segmentation, self.prediction_array.roi, output_array)

        return output_array_identifier
