import sys
from pathlib import Path
from typing import Optional

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
import daisy

import numpy as np
import click

import logging

logger = logging.getLogger(__file__)

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
    type=int,
    help="The training iteration of the model to use for prediction.",
    default=None,
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
def start_worker(
    run_name: str,
    iteration: int | None,
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
):
    compute_context = create_compute_context()
    device = compute_context.device

    # retrieving run
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    if iteration is not None:
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
    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    print(f"Predicting with input size {input_size}, output size {output_size}")

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
    model.eval()
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
        device=str(device),
    )

    # make reference batch request
    request = gp.BatchRequest()
    request.add(raw, input_size, voxel_size=input_voxel_size)
    request.add(
        prediction,
        output_size,
        voxel_size=output_voxel_size,
    )

    daisy_client = daisy.Client()

    while True:
        with daisy_client.acquire_block() as block:
            if block is None:
                return

            print(f"Processing block {block}")

            chunk_request = request.copy()
            chunk_request[raw].roi = block.read_roi
            chunk_request[prediction].roi = block.write_roi

            with gp.build(pipeline):
                batch = pipeline.request_batch(chunk_request)
            # prediction: (1, [c,] d, h, w)
            output = batch.arrays[prediction].data.squeeze()

            # convert to uint8 if necessary:
            if output_array.dtype == np.uint8:
                if "sigmoid" not in str(model.eval_activation).lower():
                    # assume output is in [-1, 1]
                    output += 1
                    output /= 2
                output *= 255
                output = output.clip(0, 255)
                output = output.astype(np.uint8)
            output_array[block.write_roi] = output


def spawn_worker(
    run_name: str,
    iteration: int | None,
    input_array_identifier: "LocalArrayIdentifier",
    output_array_identifier: "LocalArrayIdentifier",
):
    """Spawn a worker to predict on a given dataset.

    Args:
        run_name (str): The name of the run to apply.
        iteration (int or None): The training iteration of the model to use for prediction.
        input_array_identifier (LocalArrayIdentifier): The raw data to predict on.
        output_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
    """
    compute_context = create_compute_context()

    # Make the command for the worker to run
    command = [
        # "python",
        sys.executable,
        path,
        "start-worker",
        "--run-name",
        run_name,
        "--input_container",
        input_array_identifier.container,
        "--input_dataset",
        input_array_identifier.dataset,
        "--output_container",
        output_array_identifier.container,
        "--output_dataset",
        output_array_identifier.dataset,
    ]
    if iteration is not None:
        command.extend(["--iteration", str(iteration)])

    print("Defining worker with command: ", compute_context.wrap_command(command))

    def run_worker():
        # Run the worker in the given compute context
        print("Running worker with command: ", command)
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
