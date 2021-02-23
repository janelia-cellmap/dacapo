from .post_processor_abc import PostProcessorABC
from .steps import ArgMaxStep
from dacapo.store import MongoDbStore

import daisy
import numpy as np
import attr

import time
from typing import Tuple
from pathlib import Path


@attr.s
class ArgMax(PostProcessorABC):
    zarr_collection: Path = attr.ib()
    soft_predictions: str = attr.ib()
    predictions: str = attr.ib()
    steps: Tuple[ArgMaxStep] = attr.ib()

    def task(self):
        pass

    def process(self, prediction, parameters):
        return daisy.Array(
            np.argmax(prediction.data, axis=0), prediction.roi, prediction.voxel_size
        )

    def daisy_steps(self):
        yield ("argmax", "shrink", blockwise_argmax, ["volumes/pred_labels"])


def blockwise_argmax(
    run_hash,
    output_dir,
    output_filename,
    probs_dataset,
    labels_dataset,
):

    probs = daisy.open_ds(f"{output_dir}/{output_filename}", probs_dataset, mode="r")
    labels = daisy.open_ds(f"{output_dir}/{output_filename}", labels_dataset, mode="r+")

    pred_id = f"{run_hash}_argmax"
    step_id = "prediction"
    store = MongoDbStore()

    client = daisy.Client()
    while True:

        block = client.acquire_block()

        if block is None:
            break

        start = time.time()

        probabilities = probs.to_ndarray(block.write_roi)

        predicted_labels = np.argmax(probabilities, axis=0)

        labels[block.write_roi] = predicted_labels

        store.mark_block_done(
            pred_id, step_id, block.block_id, start, time.time() - start
        )

        client.release_block(block, ret=0)
