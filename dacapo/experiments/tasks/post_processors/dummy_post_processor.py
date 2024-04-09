from .dummy_post_processor_parameters import DummyPostProcessorParameters
from .post_processor import PostProcessor

import numpy as np
import zarr

from typing import Iterable


class DummyPostProcessor(PostProcessor):
    """
    Dummy post-processor that stores some dummy data. The dummy data is a 10x10x10
    array filled with the value of the min_size parameter. The min_size parameter
    is specified in the parameters of the post-processor. The post-processor has
    a detection threshold that is used to determine if an object is detected.

    Attributes:
        detection_threshold: The detection threshold.
    Methods:
        enumerate_parameters: Enumerate all possible parameters of this post-processor.
        set_prediction: Set the prediction array identifier.
        process: Convert predictions into the final output.
    Note:
        This class is abstract. Subclasses must implement the abstract methods. Once
        created, the values of its attributes cannot be changed.
    """

    def __init__(self, detection_threshold: float):
        """
        Initialize the post-processor.

        Args:
            detection_threshold: The detection threshold.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = DummyPostProcessor(0.5)
        Note:
            This method must be implemented in the subclass. It should set the
            `detection_threshold` attribute.
        """
        self.detection_threshold = detection_threshold

    def enumerate_parameters(self) -> Iterable[DummyPostProcessorParameters]:
        """
        Enumerate all possible parameters of this post-processor. Should
        return instances of ``PostProcessorParameters``.

        Returns:
            An iterable of `PostProcessorParameters` instances.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = DummyPostProcessor()
            >>> for parameters in post_processor.enumerate_parameters():
            ...     print(parameters)
            DummyPostProcessorParameters(id=0, min_size=1)
            DummyPostProcessorParameters(id=1, min_size=2)
            DummyPostProcessorParameters(id=2, min_size=3)
            DummyPostProcessorParameters(id=3, min_size=4)
            DummyPostProcessorParameters(id=4, min_size=5)
            DummyPostProcessorParameters(id=5, min_size=6)
            DummyPostProcessorParameters(id=6, min_size=7)
            DummyPostProcessorParameters(id=7, min_size=8)
            DummyPostProcessorParameters(id=8, min_size=9)
            DummyPostProcessorParameters(id=9, min_size=10)
        Note:
            This method must be implemented in the subclass. It should return an
            iterable of `PostProcessorParameters` instances.
        """

        for i, min_size in enumerate(range(1, 11)):
            yield DummyPostProcessorParameters(id=i, min_size=min_size)

    def set_prediction(self, prediction_array_identifier):
        """
        Set the prediction array identifier.

        Args:
            prediction_array_identifier: The identifier of the array containing
                the model's prediction.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = DummyPostProcessor()
            >>> post_processor.set_prediction("prediction")
        Note:
            This method must be implemented in the subclass. It should set the
            `prediction_array_identifier` attribute.
        """
        pass

    def process(self, parameters, output_array_identifier, *args, **kwargs):
        """
        Convert predictions into the final output.

        Args:
            parameters: The parameters of the post-processor.
            output_array_identifier: The identifier of the output array.
            num_workers: The number of workers to use.
            chunk_size: The size of the chunks to process.
        Returns:
            The output array.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        Examples:
            >>> post_processor = DummyPostProcessor()
            >>> post_processor.process(parameters, "output")
        Note:
            This method must be implemented in the subclass. It should process the
            predictions and store the output in the output array.
        """
        # store some dummy data
        f = zarr.open(str(output_array_identifier.container), "a")
        f[output_array_identifier.dataset] = np.ones((10, 10, 10)) * parameters.min_size
