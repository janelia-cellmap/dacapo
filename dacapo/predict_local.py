from dacapo.experiments.model import Model
from dacapo.store.local_array_store import LocalArrayIdentifier
from funlib.persistence import open_ds, prepare_ds, Array
from dacapo.utils.array_utils import to_ndarray
from funlib.geometry import Coordinate, Roi
import numpy as np
from dacapo.compute_context import create_compute_context
from typing import Optional
import logging
import daisy
import torch
import os
from dacapo.utils.array_utils import to_ndarray, save_ndarray
from dacapo.tmp import create_from_identifier

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
            f"{raw_array_identifier.container}/{raw_array_identifier.dataset}"
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

    axes = ["c^", "z", "y", "x"]

    num_channels = model.num_out_channels

    result_dataset = create_from_identifier(
        prediction_array_identifier,
        axes,
        output_roi,
        num_channels,
        output_voxel_size,
        np.float32,
        overwrite=True,
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
        raw_input = raw_array.to_ndarray(block.read_roi)

        # expand batch dimension
        # this is done in case models use BatchNorm or similar layers that
        # expect a batch dimension
        raw_input = np.expand_dims(raw_input, 0)

        # raw may or may not have channel dimensions.
        axis_names = raw_array.axis_names
        if raw_array.channel_dims == 0:
            raw_input = np.expand_dims(raw_input, 0)
            axis_names = ["c^"] + axis_names

        with torch.no_grad():
            model.eval()
            predictions = (
                model.forward(torch.from_numpy(raw_input).float().to(device))
                .detach()
                .cpu()
                .numpy()[0]
            )
            model.train()
            predictions = Array(
                predictions,
                block.write_roi.offset,
                raw_array.voxel_size,
                axis_names,
                raw_array.units,
            )

            result_dataset[block.write_roi.intersect(result_dataset.roi)] = predictions[
                block.write_roi.intersect(result_dataset.roi)
            ]

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
