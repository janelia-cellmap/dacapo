import attr
import numpy as np
import daisy

from dacapo.store import MongoDbStore
from .step_abc import PostProcessingStepABC

import time


@attr.s
class ArgMaxStep(PostProcessingStepABC):
    step_id: str = attr.ib(default="argmax")

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

        probs = daisy.open_ds(container, probabilities_dataset, mode="r")
        labels = daisy.open_ds(container, labels_dataset, mode="r+")

        input_roi = probs.roi
        read_roi = daisy.Roi(
            (0,) * input_roi.dims,
            tuple(min(a, b) for a, b in zip(input_roi.shape, [128] * input_roi.dims)),
        )  # default to 128 voxels in each axis unless input_roi has smaller shape
        write_roi = read_roi
        num_workers = 2  # should be configurable

        store = MongoDbStore()

        def argmax_block(b):
            start = time.time()

            probabilities = probs.to_ndarray(b.write_roi)

            predicted_labels = np.argmax(probabilities, axis=0)

            labels[b.write_roi] = predicted_labels

            store.mark_block_done(
                pred_id, self.step_id, b.block_id, start, time.time() - start
            )

        t = daisy.Task(
            task_id=f"{pred_id}_{self.step_id}",
            total_roi=input_roi,
            read_roi=read_roi,
            write_roi=write_roi,
            process_function=lambda b: argmax_block(b),
            check_function=lambda b: store.check_block(
                pred_id, self.step_id, b.block_id
            ),
            num_workers=num_workers,
            fit="overhang",
        )

        return [t, [{}]]
