import daisy
from daisy import Roi, Coordinate
from funlib.persistence import open_ds
from dacapo.utils.array_utils import to_ndarray, save_ndarray
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from dacapo.store.array_store import LocalArrayIdentifier
from .argmax_post_processor_parameters import ArgmaxPostProcessorParameters
from .post_processor import PostProcessor
import numpy as np
from daisy import Roi, Coordinate


class ArgmaxPostProcessor(PostProcessor):
    def __init__(self):
        pass

    def enumerate_parameters(self):
        yield ArgmaxPostProcessorParameters(id=1)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array_identifier = prediction_array_identifier
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(
        self,
        parameters,
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

        write_size = [
            b * v
            for b, v in zip(
                block_size[-self.prediction_array.dims :],
                self.prediction_array.voxel_size,
            )
        ]

        output_array = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            [dim for dim in self.prediction_array.axes if dim != "c"],
            self.prediction_array.roi,
            None,
            self.prediction_array.voxel_size,
            np.uint8,
        )

        read_roi = Roi((0, 0, 0), write_size[-self.prediction_array.dims :])
        input_array = open_ds(
            self.prediction_array_identifier.container.path,
            self.prediction_array_identifier.dataset,
        )

        def process_block(block):
            # Apply argmax to each block of data
            data = np.argmax(
                to_ndarray(input_array, block.read_roi),
                axis=self.prediction_array.axes.index("c"),
            ).astype(np.uint8)
            save_ndarray(data, block.write_roi, output_array)

        # Define the task for blockwise processing
        task = daisy.Task(
            f"argmax_{output_array.dataset}",
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

        # Run the task blockwise
        return daisy.run_blockwise([task], multiprocessing=False)
