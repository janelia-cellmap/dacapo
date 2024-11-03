from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from .threshold_post_processor_parameters import ThresholdPostProcessorParameters
from dacapo.store.array_store import LocalArrayIdentifier
from .post_processor import PostProcessor
import numpy as np
import daisy
from daisy import Roi, Coordinate
from dacapo.utils.array_utils import to_ndarray, save_ndarray
from funlib.persistence import open_ds

from typing import Iterable


class ThresholdPostProcessor(PostProcessor):
    

    def __init__(self):
        pass

    def enumerate_parameters(self) -> Iterable["ThresholdPostProcessorParameters"]:
        
        for i, threshold in enumerate([127]):
            yield ThresholdPostProcessorParameters(id=i, threshold=threshold)

    def set_prediction(self, prediction_array_identifier):
        
        self.prediction_array_identifier = prediction_array_identifier
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(
        self,
        parameters: "ThresholdPostProcessorParameters",  # type: ignore[override]
        output_array_identifier: "LocalArrayIdentifier",
        num_workers: int = 12,
        block_size: Coordinate = Coordinate((256, 256, 256)),
    ) -> ZarrArray:
        
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
        if self.prediction_array._daisy_array.chunk_shape is not None:
            block_size = self.prediction_array._daisy_array.chunk_shape

        write_size = [
            b * v
            for b, v in zip(
                block_size[-self.prediction_array.dims :],
                self.prediction_array.voxel_size,
            )
        ]
        output_array = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            self.prediction_array.axes,
            self.prediction_array.roi,
            self.prediction_array.num_channels,
            self.prediction_array.voxel_size,
            np.uint8,
        )

        read_roi = Roi((0, 0, 0), write_size[-self.prediction_array.dims :])
        input_array = open_ds(
            self.prediction_array_identifier.container.path,
            self.prediction_array_identifier.dataset,
        )

        def process_block(block):
            data = to_ndarray(input_array, block.read_roi) > parameters.threshold
            data = data.astype(np.uint8)
            if int(data.max()) == 0:
                print("No data in block", block.read_roi)
                return
            save_ndarray(data, block.write_roi, output_array)

        task = daisy.Task(
            f"threshold_{output_array.dataset}",
            total_roi=self.prediction_array.roi,
            read_roi=read_roi,
            write_roi=read_roi,
            process_function=process_block,
            check_function=None,
            read_write_conflict=False,
            fit="overhang",
            max_retries=0,
            timeout=None,
        )

        return daisy.run_blockwise([task], multiprocessing=False)
