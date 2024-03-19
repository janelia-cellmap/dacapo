from pathlib import Path

from dacapo.blockwise import run_blockwise
import dacapo.blockwise
from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray

from funlib.geometry import Coordinate, Roi
import numpy as np
import zarr

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def predict(
    run_name: str,
    iteration: int,
    input_container: Path | str,
    input_dataset: str,
    output_path: LocalArrayIdentifier | Path | str,
    output_roi: Optional[Roi | str] = None,
    num_workers: int = 12,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    overwrite: bool = True,
):
    """Predict with a trained model.

    Args:
        run_name (str): The name of the run to predict with.
        iteration (int): The training iteration of the model to use for prediction.
        input_container (Path | str): The container of the input array.
        input_dataset (str): The dataset name of the input array.
        output_path (LocalArrayIdentifier | str): The path where the prediction array will be stored, or a LocalArryIdentifier for the prediction array.
        output_roi (Optional[Roi | str], optional): The ROI of the output array. If None, the ROI of the input array will be used. Defaults to None.
        num_workers (int, optional): The number of workers to use for blockwise prediction. Defaults to 30.
        output_dtype (np.dtype | str, optional): The dtype of the output array. Defaults to np.uint8.
        overwrite (bool, optional): If True, the output array will be overwritten if it already exists. Defaults to True.
    """
    # retrieving run
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # check to see if we can load the weights
    weights_store = create_weights_store()
    try:
        weights_store.retrieve_weights(run_name, iteration)
    except FileNotFoundError:
        raise ValueError(
            f"No weights found for run {run_name} at iteration {iteration}."
        )

    # get arrays
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    raw_array = ZarrArray.open_from_array_identifier(input_array_identifier)
    if isinstance(output_path, LocalArrayIdentifier):
        output_array_identifier = output_path
    else:
        if ".zarr" in str(output_path) or ".n5" in str(output_path):
            output_container = Path(output_path)
        else:
            output_container = Path(
                output_path,
                Path(input_container).stem + ".zarr",
            )  # TODO: zarr hardcoded
        output_array_identifier = LocalArrayIdentifier(
            output_container, f"prediction_{run_name}_{iteration}"
        )

    # get the model's input and output size
    model = run.model.eval()

    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    # calculate input and output rois

    context = (input_size - output_size) // 2

    if output_roi is None:
        output_roi = raw_array.roi.grow(-context, -context)
    elif isinstance(output_roi, str):
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
        output_roi = output_roi.snap_to_grid(
            raw_array.voxel_size, mode="grow"
        ).intersect(raw_array.roi.grow(-context, -context))
    _input_roi = output_roi.grow(context, context)  # type: ignore

    if isinstance(output_dtype, str):
        output_dtype = np.dtype(output_dtype)

    print(f"Predicting with input size {input_size}, output size {output_size}")

    print(f"Total input ROI: {_input_roi}, output ROI: {output_roi}")

    # prepare prediction dataset
    ZarrArray.create_from_array_identifier(
        output_array_identifier,
        raw_array.axes,
        output_roi,
        model.num_out_channels,
        output_voxel_size,
        output_dtype,
        overwrite=overwrite,
        write_size=output_size,
    )

    # run blockwise prediction
    worker_file = str(Path(Path(dacapo.blockwise.__file__).parent, "predict_worker.py"))
    print("Running blockwise prediction with worker_file: ", worker_file)
    run_blockwise(
        worker_file=worker_file,
        total_roi=_input_roi,
        read_roi=Roi((0, 0, 0), input_size),
        write_roi=Roi(context, output_size),
        num_workers=num_workers,
        max_retries=2,  # TODO: make this an option
        timeout=None,  # TODO: make this an option
        ######
        run_name=run_name,
        iteration=iteration,
        input_array_identifier=input_array_identifier,
        output_array_identifier=output_array_identifier,
    )
    print("Done predicting.")
