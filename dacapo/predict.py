from pathlib import Path

import click
from dacapo.blockwise import run_blockwise
from dacapo.experiments import Run
from dacapo.gp import DaCapoArraySource
from dacapo.experiments import Model
from dacapo.store import create_config_store
from dacapo.store import create_weights_store
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.compute_context import LocalTorch, ComputeContext
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray, Array

from funlib.geometry import Coordinate, Roi
import gunpowder as gp
import gunpowder.torch as gp_torch
import numpy as np
import zarr

from typing import Optional
import logging

logger = logging.getLogger(__name__)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The name of the run to apply."
)
@click.option(
    "-i",
    "--iteration",
    required=True,
    type=int,
    help="The training iteration of the model to use for prediction.",
)
@click.option(
    "-ic",
    "--input_container",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-id", "--input_dataset", required=True, type=str)
@click.option("-op", "--output_path", required=True, type=click.Path(file_okay=False))
@click.option(
    "-roi",
    "--output_roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option("-w", "--num_workers", type=int, default=30)
@click.option("-dt", "--output_dtype", type=str, default="uint8")
@click.option(
    "-cc",
    "--compute_context",
    type=str,
    default="LocalTorch",
    help="The compute context to use for prediction. Must be the name of a subclass of ComputeContext.",
)
@click.option("-ow", "--overwrite", is_flag=True)
def predict(
    run_name: str,
    iteration: int,
    input_container: Path or str,
    input_dataset: str,
    output_path: Path or str,
    output_roi: Optional[str | Roi] = None,
    num_workers: int = 30,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    compute_context: ComputeContext | str = LocalTorch(),
    overwrite: bool = True,
):
    # retrieving run
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # get arrays
    raw_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    raw_array = ZarrArray.open_from_array_identifier(raw_array_identifier)
    output_container = Path(
        output_path,
        "".join(Path(input_container).name.split(".")[:-1]) + f".zarr",
    )  # TODO: zarr hardcoded
    prediction_array_identifier = LocalArrayIdentifier(
        output_container, f"prediction_{run_name}_{iteration}"
    )

    if isinstance(output_roi, str):
        start, end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in output_roi.strip("[]").split(",")
            ]
        )
        output_roi = Roi(
            Coordinate(start),
            Coordinate(end) - Coordinate(start),
        )

    if output_roi is None:
        output_roi = raw_array.roi
    else:
        output_roi = output_roi.snap_to_grid(
            raw_array.voxel_size, mode="grow"
        ).intersect(raw_array.roi)

    if isinstance(output_dtype, str):
        output_dtype = np.dtype(output_dtype)

    model = run.model.eval()

    # get the model's input and output size

    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    logger.info(
        "Predicting with input size %s, output size %s", input_size, output_size
    )

    # calculate input and output rois

    context = (input_size - output_size) / 2
    if output_roi is None:
        input_roi = raw_array.roi
        output_roi = input_roi.grow(-context, -context)
    else:
        input_roi = output_roi.grow(context, context)

    logger.info("Total input ROI: %s, output ROI: %s", input_roi, output_roi)

    # prepare prediction dataset
    axes = ["c"] + [axis for axis in raw_array.axes if axis != "c"]
    ZarrArray.create_from_array_identifier(
        prediction_array_identifier,
        axes,
        output_roi,
        model.num_out_channels,
        output_voxel_size,
        output_dtype,
        overwrite=overwrite,
    )

    # run blockwise prediction
    run_blockwise(
        worker_file=str(Path(Path(__file__).parent, "blockwise", "predict_worker.py")),
        compute_context=compute_context,
        total_roi=output_roi,
        read_roi=Roi((0, 0, 0), input_size),
        write_roi=Roi((0, 0, 0), output_size),
        num_workers=num_workers,
        max_retries=2,  # TODO: make this an option
        timeout=None,  # TODO: make this an option
        ######
        run_name=run_name,
        iteration=iteration,
        raw_array_identifier=raw_array_identifier,
        prediction_array_identifier=prediction_array_identifier,
    )

    container = zarr.open(str(prediction_array_identifier.container))
    dataset = container[prediction_array_identifier.dataset]
    dataset.attrs["axes"] = (  # type: ignore
        raw_array.axes if "c" in raw_array.axes else ["c"] + raw_array.axes
    )
