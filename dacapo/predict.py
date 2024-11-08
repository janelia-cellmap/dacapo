from upath import UPath as Path
from dacapo.blockwise import run_blockwise
import dacapo.blockwise
from dacapo.experiments import Run
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.store.local_array_store import LocalArrayIdentifier

from dacapo.compute_context import create_compute_context, LocalTorch
from dacapo.tmp import open_from_identifier, create_from_identifier

from funlib.geometry import Coordinate, Roi
import numpy as np

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def predict(
    run_name: str | Run,
    iteration: int | None,
    input_container: Path | str,
    input_dataset: str,
    output_path: LocalArrayIdentifier | Path | str,
    output_roi: Optional[Roi | str] = None,
    num_workers: int = 1,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    overwrite: bool = True,
):
    """Predict with a trained model.

    Args:
        run_name (str or Run): The name of the run to predict with or the Run object.
        iteration (int or None): The training iteration of the model to use for prediction.
        input_container (Path | str): The container of the input array.
        input_dataset (str): The dataset name of the input array.
        output_path (LocalArrayIdentifier | str): The path where the prediction array will be stored, or a LocalArryIdentifier for the prediction array.
        output_roi (Optional[Roi | str], optional): The ROI of the output array. If None, the ROI of the input array will be used. Defaults to None.
        num_workers (int, optional): The number of workers to use for blockwise prediction. Defaults to 1 for local processing, otherwise 12.
        output_dtype (np.dtype | str, optional): The dtype of the output array. Defaults to np.uint8.
        overwrite (bool, optional): If True, the output array will be overwritten if it already exists. Defaults to True.
    Raises:
        ValueError: If run_name is not found in config store
    Examples:
        >>> predict("run_name", 100, "input.zarr", "raw", "output.zarr", output_roi="[0:100,0:100,0:100]")

    """
    # retrieving run
    if isinstance(run_name, Run):
        run = run_name
        run_name = run.name
    else:
        config_store = create_config_store()
        run_config = config_store.retrieve_run_config(run_name)
        run = Run(run_config)

    # get arrays
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    raw_array = open_from_identifier(input_array_identifier)
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

    compute_context = create_compute_context()
    if isinstance(compute_context, LocalTorch):
        num_workers = 1

    model = run.model.eval()

    if iteration is not None and not compute_context.distribute_workers:
        # create weights store
        weights_store = create_weights_store()

        # load weights
        run.model.load_state_dict(
            weights_store.retrieve_weights(run_name, iteration).model
        )

    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]
    num_out_channels = model.num_out_channels
    # del model

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
    if raw_array.channel_dims == 0:
        axis_names = ["c^"] + raw_array.axis_names
    else:
        axis_names = raw_array.axis_names

    if isinstance(output_roi, Roi):
        out_roi: Roi = output_roi
    else:
        raise ValueError("out_roi must be a roi")
    create_from_identifier(
        output_array_identifier,
        axis_names,
        out_roi,
        num_out_channels,
        output_voxel_size,
        output_dtype,
        overwrite=overwrite,
        write_size=output_size,
    )

    # run blockwise prediction
    worker_file = str(Path(Path(dacapo.blockwise.__file__).parent, "predict_worker.py"))
    print("Running blockwise prediction with worker_file: ", worker_file)
    success = run_blockwise(
        worker_file=worker_file,
        total_roi=_input_roi,
        read_roi=Roi((0, 0, 0), input_size),
        write_roi=Roi(context, output_size),
        num_workers=num_workers,
        max_retries=2,  # TODO: make this an option
        timeout=None,  # TODO: make this an option
        ######
        run_name=run.name,
        iteration=iteration,
        input_array_identifier=input_array_identifier,
        output_array_identifier=output_array_identifier,
    )
    print("Done predicting.")
    return success
