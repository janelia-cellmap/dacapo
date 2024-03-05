from pathlib import Path

import torch
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.gp import DaCapoArraySource
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.experiments import Run
from dacapo.compute_context import create_compute_context
import gunpowder as gp
import gunpowder.torch as gp_torch

from funlib.geometry import Coordinate

import numpy as np
import click

import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


read_write_conflict: bool = False
fit: str = "valid"
path = __file__


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


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
@click.option(
    "-oc", "--output_container", required=True, type=click.Path(file_okay=False)
)
@click.option("-od", "--output_dataset", required=True, type=str)
@click.option("-d", "--device", type=str, default="cuda")
def start_worker(
    run_name: str,
    iteration: int,
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
    device: str | torch.device = "cuda",
):
    logger.warning(f"Predicting {run_name}")
    # retrieving run
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # create weights store
    weights_store = create_weights_store()

    # load weights
    weights_store.retrieve_weights(run_name, iteration)

    # get arrays
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    raw_array = ZarrArray.open_from_array_identifier(input_array_identifier)

    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )
    output_array = ZarrArray.open_from_array_identifier(output_array_identifier)

    # set benchmark flag to True for performance
    torch.backends.cudnn.benchmark = True

    # get the model's input and output size
    model = run.model.eval().to(device)
    # raise error if no cuda
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if not model.is_cuda:
        raise ValueError("Model is not on CUDA")
    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    logger.warning(
        "Predicting with input size %s, output size %s", input_size, output_size
    )

    # create gunpowder keys

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICTION")

    # assemble prediction pipeline

    # prepare data source
    pipeline = DaCapoArraySource(raw_array, raw)
    # raw: (c, d, h, w)
    pipeline += gp.Pad(raw, None)
    # raw: (c, d, h, w)
    pipeline += gp.Unsqueeze([raw])
    # raw: (1, c, d, h, w)

    pipeline += gp.Normalize(raw)

    # predict
    pipeline += gp_torch.Predict(
        model=model,
        inputs={"x": raw},
        outputs={0: prediction},
        array_specs={
            prediction: gp.ArraySpec(
                voxel_size=output_voxel_size,
                dtype=np.float32,  # assumes network output is float32
            )
        },
        spawn_subprocess=False,
        device=device,  # type: ignore
    )
    # raw: (1, c, d, h, w)
    # prediction: (1, [c,] d, h, w)

    # prepare writing
    pipeline += gp.Squeeze([raw, prediction])
    # raw: (c, d, h, w)
    # prediction: (c, d, h, w)

    # convert to uint8 if necessary:
    if output_array.dtype == np.uint8:
        pipeline += gp.IntensityScaleShift(
            prediction, scale=255.0, shift=0.0
        )  # assumes float32 is [0,1]
        pipeline += gp.AsType(prediction, output_array.dtype)

    # write to output array
    pipeline += gp.ZarrWrite(
        {
            prediction: output_array_identifier.dataset,
        },
        store=str(output_array_identifier.container),
    )

    # make reference batch request
    request = gp.BatchRequest()
    request.add(raw, input_size, voxel_size=input_voxel_size)
    request.add(
        prediction,
        output_size,
        voxel_size=output_voxel_size,
    )
    # use daisy requests to run pipeline
    pipeline += gp.DaisyRequestBlocks(
        reference=request,
        roi_map={raw: "read_roi", prediction: "write_roi"},
        num_workers=1,
    )

    logger.warning("Running pipeline")

    with gp.build(pipeline):
        batch = pipeline.request_batch(gp.BatchRequest())


def spawn_worker(
    run_name: str,
    iteration: int,
    input_array_identifier: "LocalArrayIdentifier",
    output_array_identifier: "LocalArrayIdentifier",
):
    """Spawn a worker to predict on a given dataset.

    Args:
        run_name (str): The name of the run to apply.
        iteration (int): The training iteration of the model to use for prediction.
        input_array_identifier (LocalArrayIdentifier): The raw data to predict on.
        output_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
    """
    # compute_context = create_compute_context()

    # Make the command for the worker to run
    command = [
        "python",
        path,
        "start-worker",
        "--run-name",
        run_name,
        "--iteration",
        str(iteration),
        "--input_container",
        str(input_array_identifier.container),
        "--input_dataset",
        input_array_identifier.dataset,
        "--output_container",
        str(output_array_identifier.container),
        "--output_dataset",
        output_array_identifier.dataset,
        "--device",
        "cuda",
    ]

    str_command = " ".join(command)
    logger.warning(f"Spawning worker with command: {str_command}")

    def run_worker():
        # Run the worker in the given compute context
        import os
        import subprocess

        folder = f"/groups/cellmap/cellmap/zouinkhim/ml_experiments_ome/validate/logs/{run_name}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        name = f"predict_{run_name}_{iteration}"
        full_command = [
                "bsub",
                "-q", "gpu_tesla",
                "-gpu", "num=1",
                "-n", "14",
                 "-J", name,
                "-o", f"/groups/cellmap/cellmap/zouinkhim/ml_experiments_ome/validate/logs/{run_name}/predict_{iteration}.out",
                "-e", f"/groups/cellmap/cellmap/zouinkhim/ml_experiments_ome/validate/logs/{run_name}/predict_{iteration}.err",
                    "-P",
                    f"cellmap",
                ]+ command
        str_command = " ".join(full_command)
        print(f"Submitting: {str_command}")
        logger.warning(f"Submitting: {str_command}")
        subprocess.run(full_command)
        # compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
