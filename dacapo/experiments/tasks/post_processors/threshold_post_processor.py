```python
from pathlib import Path
from dacapo.blockwise.scheduler import run_blockwise
from dacapo.compute_context import ComputeContext, LocalTorch
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from .threshold_post_processor_parameters import ThresholdPostProcessorParameters
from .post_processor import PostProcessor
import numpy as np
from daisy import Roi, Coordinate

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from dacapo.store.local_array_store import LocalArrayIdentifier
    from dacapo.experiments.tasks.post_processors import (
        ThresholdPostProcessorParameters,
    )


class ThresholdPostProcessor(PostProcessor):
    """
    A post-processing class which inherits from the `PostProcessor` parent class.
    Utilizes threshold techniques for post-processing which can be parametrized.
    """

    def __init__(self):
        pass

    def enumerate_parameters(self) -> Iterable[ThresholdPostProcessorParameters]:
        """
        Enumerate all possible parameters of this post-processor.
        
        Yields
        ------
        ThresholdPostProcessorParameters
            post-process parameters.
        """

        for i, threshold in enumerate([-0.1, 0.0, 0.1]):
            yield ThresholdPostProcessorParameters(id=i, threshold=threshold)

    def set_prediction(self, prediction_array_identifier: "LocalArrayIdentifier"):
        """
        Set the prediction array for post-processing.
        
        Parameters
        ----------
        prediction_array_identifier : `LocalArrayIdentifier`
            Identifier for the prediction array.
        """

        self.prediction_array = ZarrArray.open_from_array_identifier(
            prediction_array_identifier
        )

    def process(
        self,
        parameters: "ThresholdPostProcessorParameters", 
        output_array_identifier: "LocalArrayIdentifier",
        compute_context: ComputeContext | str = LocalTorch(),
        num_workers: int = 16,
        chunk_size: Coordinate = Coordinate((64, 64, 64)),
    ) -> ZarrArray:
        """
        Apply the threshold post-processing on the prediction array.

        Parameters
        ----------
        parameters : `ThresholdPostProcessorParameters`
            Parameters for the post-processing.
        output_array_identifier : `LocalArrayIdentifier`
            Identifier for the output array.
        compute_context : `ComputeContext` or `str`, optional
            The context to compute in, by default LocalTorch().
        num_workers : int, optional
            Number of workers to use for parallel processing, by default 16.
        chunk_size : `Coordinate`, optional
            The size of chunk to use for processing, by default Coordinate((64, 64, 64)).

        Returns
        -------
        ZarrArray
            The post-processed prediction array.

        Raises
        ------
        TODO
        """

        output_array = ZarrArray.create_from_array_identifier(
            output_array_identifier,
            self.prediction_array.axes,
            self.prediction_array.roi,
            self.prediction_array.num_channels,
            self.prediction_array.voxel_size,
            np.uint8,
        )

        read_roi = Roi((0, 0, 0), self.prediction_array.voxel_size * chunk_size)
        
        run_blockwise(
            worker_file=str(
                Path(Path(__file__).parent, "blockwise", "predict_worker.py")
            ),
            compute_context=compute_context,
            total_roi=self.prediction_array.roi,
            read_roi=read_roi,
            write_roi=read_roi,
            num_workers=num_workers,
            max_retries=2,  
            timeout=None, 
            input_array_identifier=LocalArrayIdentifier(
                self.prediction_array.file_name, self.prediction_array.dataset
            ),
            output_array_identifier=output_array_identifier,
            threshold=parameters.threshold,
        )

        return output_array
```
