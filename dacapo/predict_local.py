from dacapo.experiments.model import Model
from dacapo.store.local_array_store import LocalArrayIdentifier
from funlib.persistence import open_ds, prepare_ds, Array
from dacapo.utils.array_utils import to_ndarray
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from funlib.geometry import Coordinate, Roi
import numpy as np
from dacapo.compute_context import create_compute_context
from typing import Optional
import logging
import daisy
import torch
import os
from dacapo.utils.array_utils import to_ndarray, save_ndarray

logger = logging.getLogger(__name__)


def predict(
    model: Model,
    raw_array_identifier: LocalArrayIdentifier,
    prediction_array_identifier: LocalArrayIdentifier,
    output_roi: Optional[Roi] = None,
):
    # get the model's input and output size
    if isinstance(raw_array_identifier, LocalArrayIdentifier):
        raw_array = open_ds(
            str(raw_array_identifier.container), raw_array_identifier.dataset
        )
    else:
        raw_array = raw_array_identifier
    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)

    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    context = (input_size - output_size) / 2

    if output_roi is None:
        input_roi = raw_array.roi
        output_roi = input_roi.grow(-context, -context)
    else:
        input_roi = output_roi.grow(context, context)

    read_roi = Roi((0, 0, 0), input_size)
    write_roi = read_roi.grow(-context, -context)

    axes = ["c", "z", "y", "x"]

    num_channels = model.num_out_channels

    result_dataset = ZarrArray.create_from_array_identifier(
        prediction_array_identifier,
        axes,
        output_roi,
        num_channels,
        output_voxel_size,
        np.float32,
    )

    logger.info("Total input ROI: %s, output ROI: %s", input_size, output_roi)
    logger.info("Block read ROI: %s, write ROI: %s", read_roi, write_roi)

    out_container, out_dataset = (
        prediction_array_identifier.container.path,
        prediction_array_identifier.dataset,
    )
    compute_context = create_compute_context()
    device = compute_context.device

    def predict_fn(block):
        raw_input = to_ndarray(raw_array, block.read_roi)
        # expend batch dim
        # no need to normalize, done by datasplit
        raw_input = np.expand_dims(raw_input, (0, 1))
        with torch.no_grad():
            predictions = (
                model.forward(torch.from_numpy(raw_input).float().to(device))
                .detach()
                .cpu()
                .numpy()[0]
            )

            save_ndarray(predictions, block.write_roi, result_dataset)
            # result_dataset[block.write_roi] = predictions

    # fixing the input roi to be a multiple of the output voxel size
    input_roi = input_roi.snap_to_grid(
        np.lcm(input_voxel_size, output_voxel_size), mode="shrink"
    )

    task = daisy.Task(
        f"predict_{out_container}_{out_dataset}",
        total_roi=input_roi,
        read_roi=Roi((0, 0, 0), input_size),
        write_roi=Roi(context, output_size),
        process_function=predict_fn,
        check_function=None,
        read_write_conflict=False,
        fit="overhang",
        max_retries=0,
        timeout=None,
    )

    return daisy.run_blockwise([task], multiprocessing=False)
