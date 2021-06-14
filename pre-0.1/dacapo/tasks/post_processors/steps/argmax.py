import attr
import numpy as np
import daisy

from dacapo.store import MongoDbStore
from .step_abc import PostProcessingStepABC

import time
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@attr.s
class ArgMaxStep(PostProcessingStepABC):
    step_id: str = attr.ib(default="argmax")

    # blockwise_processing_parameters
    write_shape: Optional[List[int]] = attr.ib(default=None)
    num_workers: int = attr.ib(default=2)

    def tasks(
        self,
        pred_id: str,
        container: str,
        probabilities_dataset: str,
        labels_dataset: str,
        upstream_tasks=None,
    ):
        """
        pred_id: The id of the prediction that you are postprocessing
        container: the path to the zarr container for reading/writing
        probabilities_dataset: the dataset you want to argmax
        labels_dataset: the dataset to write argmaxed values to
        upstream_tasks: optional list of (task, param_dict) pairs
            if not None, apply argmax to each individually.
        """

        tasks, task_parameters = [], []

        if upstream_tasks is None:
            upstream_tasks = [None, {}]

        for upstream_task, upstream_parameters in zip(*upstream_tasks):
            parameters = dict(**upstream_parameters)

            logger.error(
                "The dataset_names should include a unique id for the parameter "
                "set to avoid conflicts"
            )
            probs = daisy.open_ds(container, probabilities_dataset, mode="r")
            labels = daisy.open_ds(container, labels_dataset, mode="r+")
            # input_roi defined by provided dataset
            # TODO: allow for subrois?
            input_roi = probs.roi

            # get write_shape
            if self.write_shape is None:
                # default to 512 per axis or input_roi shape if less than that
                write_shape = daisy.Coordinate(
                    tuple(
                        min(a, b)
                        for a, b in zip(input_roi.shape, [512] * input_roi.dims)
                    )
                )
            else:
                write_shape = self.write_shape

            write_roi = daisy.Roi((0,) * write_shape.dims, write_shape)

            t = daisy.Task(
                task_id=f"{pred_id}_{self.step_id}",
                total_roi=input_roi,
                read_roi=write_roi,
                write_roi=write_roi,
                process_function=self.get_process_function(pred_id, probs, labels),
                check_function=self.get_check_function(pred_id),
                num_workers=self.num_workers,
                fit="overhang",
            )
            tasks.append(t)
            task_parameters.append(parameters)

        return tasks, task_parameters

    def get_process_function(self, pred_id, probs, labels):
        store = MongoDbStore()

        def argmax_block(b):
            start = time.time()

            probabilities = probs.to_ndarray(b.write_roi)

            predicted_labels = np.argmax(probabilities, axis=0)

            labels[b.write_roi] = predicted_labels

            store.mark_block_done(
                pred_id, self.step_id, b.block_id, start, time.time() - start
            )

        return argmax_block
