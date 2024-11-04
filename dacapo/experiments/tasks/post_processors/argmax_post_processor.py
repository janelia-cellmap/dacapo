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
    """
    Post-processor that takes the argmax of the input array along the channel
    axis. The output is a binary array where the value is 1 if the argmax is
    greater than the threshold, and 0 otherwise.

    Attributes:
        prediction_array: The array containing the model's prediction.
    Methods:
        enumerate_parameters: Enumerate all possible parameters of this post-processor.
        set_prediction: Set the prediction array identifier.
        process: Convert predictions into the final output.
    Note:
        This class is abstract. Subclasses must implement the abstract methods. Once
        created, the values of its attributes cannot be changed.
    """

    def __init__(self):
        """
        Initialize the post-processor.

        Args:
            detection_threshold: The detection threshold.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = ArgmaxPostProcessor()
        Note:
            This method must be implemented in the subclass. It should set the
            `detection_threshold` attribute.
        """
        pass

    def enumerate_parameters(self):
        """
        Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``.

        Returns:
            An iterable of `PostProcessorParameters` instances.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = ArgmaxPostProcessor()
            >>> for parameters in post_processor.enumerate_parameters():
            ...     print(parameters)
            ArgmaxPostProcessorParameters(id=0)
        Note:
            This method must be implemented in the subclass. It should return an
            iterable of `PostProcessorParameters` instances.
        """

        yield ArgmaxPostProcessorParameters(id=1)

    def set_prediction(self, prediction_array_identifier):
        """
        Set the prediction array identifier.

        Args:
            prediction_array_identifier: The identifier of the array containing
                the model's prediction.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = ArgmaxPostProcessor()
            >>> post_processor.set_prediction("prediction")
        Note:
            This method must be implemented in the subclass. It should set the
            `prediction_array_identifier` attribute.
        """
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
        """
        Convert predictions into the final output.

        Args:
            parameters: The parameters of the post-processor.
            output_array_identifier: The identifier of the output array.
            num_workers: The number of workers to use.
            block_size: The size of the blocks to process.
        Returns:
            The output array.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = ArgmaxPostProcessor()
            >>> post_processor.set_prediction("prediction")
            >>> post_processor.process(parameters, "output")
        Note:
            This method must be implemented in the subclass. It should process the
            predictions and return the output array.
        """
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
