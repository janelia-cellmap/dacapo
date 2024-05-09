from upath import UPath as Path
from dacapo.blockwise.scheduler import run_blockwise
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from .threshold_post_processor_parameters import ThresholdPostProcessorParameters
from dacapo.store.array_store import LocalArrayIdentifier
from .post_processor import PostProcessor
import dacapo.blockwise
import numpy as np
from daisy import Roi, Coordinate

from typing import Iterable


class ThresholdPostProcessor(PostProcessor):
    """
    A post-processor that applies a threshold to the prediction.

    Attributes:
        prediction_array_identifier: The identifier of the prediction array.
        prediction_array: The prediction array.
    Methods:
        enumerate_parameters: Enumerate all possible parameters of this post-processor.
        set_prediction: Set the prediction array.
        process: Process the prediction with the given parameters.
    Note:
        This post-processor applies a threshold to the prediction. The threshold is used to define the segmentation. The prediction array is set using the `set_prediction` method.
    """

    def __init__(self):
        pass

    def enumerate_parameters(self) -> Iterable["ThresholdPostProcessorParameters"]:
        """
        Enumerate all possible parameters of this post-processor.

        Returns:
            Generator[ThresholdPostProcessorParameters]: A generator of parameters.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> for parameters in post_processor.enumerate_parameters():
            ...     print(parameters)
        Note:
            This method should return a generator of instances of ``ThresholdPostProcessorParameters``.
        """
        for i, threshold in enumerate([100, 127, 150]):
            yield ThresholdPostProcessorParameters(id=i, threshold=threshold)

    def set_prediction(self, prediction_array_identifier):
        """
        Set the prediction array.

        Args:
            prediction_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> post_processor.set_prediction(prediction_array_identifier)
        Note:
            This method should set the prediction array using the given identifier.
        """
        self.prediction_array_identifier = prediction_array_identifier
        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(
        self,
        parameters: "ThresholdPostProcessorParameters",  # type: ignore[override]
        output_array_identifier: "LocalArrayIdentifier",
        num_workers: int = 16,
        block_size: Coordinate = Coordinate((256, 256, 256)),
    ) -> ZarrArray:
        """
        Process the prediction with the given parameters.

        Args:
            parameters (ThresholdPostProcessorParameters): The parameters to use for processing.
            output_array_identifier (LocalArrayIdentifier): The identifier of the output array.
            num_workers (int): The number of workers to use for processing.
            block_size (Coordinate): The block size to use for processing.
        Returns:
            ZarrArray: The output array.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> post_processor.process(parameters, output_array_identifier)
        Note:
            This method should process the prediction with the given parameters and return the output array. The method uses the `run_blockwise` function from the `dacapo.blockwise.scheduler` module to run the blockwise post-processing.
            The output array is created using the `ZarrArray.create_from_array_identifier` function from the `dacapo.experiments.datasplits.datasets.arrays` module.
        """
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
            write_size,
        )

        read_roi = Roi((0, 0, 0), write_size[-self.prediction_array.dims :])
        # run blockwise post-processing
        run_blockwise(
            worker_file=str(
                Path(Path(dacapo.blockwise.__file__).parent, "threshold_worker.py")
            ),
            total_roi=self.prediction_array.roi,
            read_roi=read_roi,
            write_roi=read_roi,
            num_workers=num_workers,
            max_retries=2,  # TODO: make this an option
            timeout=None,  # TODO: make this an option
            ######
            input_array_identifier=self.prediction_array_identifier,
            output_array_identifier=output_array_identifier,
            threshold=parameters.threshold,
        )

        return output_array
