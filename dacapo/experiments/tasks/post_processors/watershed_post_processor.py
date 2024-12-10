from upath import UPath as Path
import dacapo.blockwise
from dacapo.blockwise.scheduler import segment_blockwise

from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.utils.array_utils import to_ndarray, save_ndarray
from funlib.persistence import open_ds
import daisy
import mwatershed as mws

from .watershed_post_processor_parameters import WatershedPostProcessorParameters
from .post_processor import PostProcessor

from funlib.geometry import Coordinate, Roi
from dacapo.tmp import create_from_identifier, open_from_identifier


import numpy as np

from typing import List


class WatershedPostProcessor(PostProcessor):
    """
    A post-processor that applies a watershed transformation to the
    prediction.

    Attributes:
        offsets: List of offsets for the watershed transformation.
    Methods:
        enumerate_parameters: Enumerate all possible parameters of this post-processor.
        set_prediction: Set the prediction array.
        process: Process the prediction with the given parameters.
    Note:
        This post-processor uses the `watershed_function.py` script to apply the watershed transformation. The offsets are used to define the neighborhood for the watershed transformation.

    """

    def __init__(self, offsets: List[Coordinate]):
        """
        A post-processor that applies a watershed transformation to the
        prediction.

        Args:
            offsets (List[Coordinate]): List of offsets for the watershed transformation.
        Examples:
            >>> WatershedPostProcessor(offsets=[(0, 0, 1), (0, 1, 0), (1, 0, 0)])
        Note:
            This post-processor uses the `watershed_function.py` script to apply the watershed transformation. The offsets are used to define the neighborhood for the watershed transformation.
        """
        self.offsets = offsets

    def enumerate_parameters(self):
        """
        Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``.

        Returns:
            Generator[WatershedPostProcessorParameters]: A generator of parameters.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> for parameters in post_processor.enumerate_parameters():
            ...     print(parameters)
        Note:
            This method should be implemented by the subclass. It should return a generator of instances of ``WatershedPostProcessorParameters``.

        """

        for i, bias in enumerate([0.1, 0.25, 0.5, 0.75, 0.9]):
            yield WatershedPostProcessorParameters(id=i, bias=bias)

    def set_prediction(self, prediction_array_identifier):
        self.prediction_array_identifier = prediction_array_identifier
        self.prediction_array = open_from_identifier(prediction_array_identifier)
        """
        Set the prediction array.

        Args:
            prediction_array_identifier (LocalArrayIdentifier): The prediction array identifier.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> post_processor.set_prediction(prediction_array_identifier)
        Note:
            This method should be implemented by the subclass. To set the prediction array, the method uses the `open_from_identifier` function from the `dacapo.experiments.datasplits.datasets.arrays` module.
        """

    def process(
        self,
        parameters: WatershedPostProcessorParameters,  # type: ignore[override]
        output_array_identifier: "LocalArrayIdentifier",
        num_workers: int = 16,
        block_size: Coordinate = Coordinate((256, 256, 256)),
    ):
        """
        Process the prediction with the given parameters.

        Args:
            parameters (WatershedPostProcessorParameters): The parameters to use for processing.
            output_array_identifier (LocalArrayIdentifier): The output array identifier.
            num_workers (int): The number of workers to use for processing.
            block_size (Coordinate): The block size to use for processing.
        Returns:
            LocalArrayIdentifier: The output array identifier.
        Raises:
            NotImplementedError: If the method is not implemented.
        Examples:
            >>> post_processor.process(parameters, output_array_identifier)
        Note:
            This method should be implemented by the subclass. To run the watershed transformation, the method uses the `segment_blockwise` function from the `dacapo.blockwise.scheduler` module.
        """
        if self.prediction_array._source_data.chunks is not None:
            block_size = Coordinate(
                self.prediction_array._source_data.chunks[
                    -self.prediction_array.spatial_dims :
                ]
            )

        output_array = create_from_identifier(
            output_array_identifier,
            [axis for axis in self.prediction_array.axis_names if axis != "c^"],
            self.prediction_array.roi,
            None,
            self.prediction_array.voxel_size,
            np.uint64,
            write_size=block_size * self.prediction_array.voxel_size,
            overwrite=True,
        )
        input_array = open_ds(
            f"{self.prediction_array_identifier.container.path}/{self.prediction_array_identifier.dataset}",
        )

        data = input_array.to_ndarray(output_array.roi).astype(float)
        segmentation = mws.agglom(
            data - parameters.bias, offsets=self.offsets, randomized_strides=True
        )
        output_array[self.prediction_array.roi] = segmentation

        return output_array_identifier
