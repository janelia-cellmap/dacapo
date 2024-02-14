from pathlib import Path
import subprocess
from typing import Optional
import dacapo
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from dacapo.experiments.model import Model
from dacapo.gp.dacapo_array_source import DaCapoArraySource
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.experiments import Run
from dacapo.compute_context import ComputeContext, LocalTorch, Bsub
import gunpowder as gp
import gunpowder.torch as gp_torch

import daisy
from daisy import Roi, Coordinate
from funlib.persistence import open_ds, Array

from skimage.transform import rescale  # TODO
import numpy as np
import torch
import click

import sys
import logging

logger = logging.getLogger(__file__)

read_write_conflict: bool = False
fit: str = "valid"


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
    input_container: Path or str,
    input_dataset: str,
    output_container: Path or str,
    output_dataset: str,
    device: str = "cuda",
):
    # retrieving run
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # create weights store
    weights_store = create_weights_store()

    # load weights
    weights_store.retrieve_weights(run_name, iteration)

    # get arrays
    raw_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    raw_array = ZarrArray.open_from_array_identifier(raw_array_identifier)

    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )
    output_array = ZarrArray.open_from_array_identifier(output_array_identifier)

    # get the model's input and output size
    model = run.model.eval()
    input_voxel_size = Coordinate(raw_array.voxel_size)
    output_voxel_size = model.scale(input_voxel_size)
    input_shape = Coordinate(model.eval_input_shape)
    input_size = input_voxel_size * input_shape
    output_size = output_voxel_size * model.compute_output_shape(input_shape)[1]

    logger.info(
        "Predicting with input size %s, output size %s", input_size, output_size
    )
    # create gunpowder keys

    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICTION")

    # assemble prediction pipeline

    # prepare data source
    pipeline = DaCapoArraySource(raw_array, raw)
    # raw: (c, d, h, w)
    pipeline += gp.Pad(raw, Coordinate((None,) * input_voxel_size.dims))
    # raw: (c, d, h, w)
    pipeline += gp.Unsqueeze([raw])
    # raw: (1, c, d, h, w)

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

    # wait for blocks to run pipeline
    client = daisy.Client()

    while True:
        print("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            ref_request = gp.BatchRequest()
            ref_request[raw] = gp.ArraySpec(
                roi=block.read_roi, voxel_size=input_voxel_size, dtype=raw_array.dtype
            )
            ref_request[prediction] = gp.ArraySpec(
                roi=block.write_roi,
                voxel_size=output_voxel_size,
                dtype=output_array.dtype,
            )

            with gp.build(pipeline):
                batch = pipeline.request_batch(ref_request)

            # write to output array
            output_array[block.write_roi] = batch.arrays[prediction].data


def spawn_worker(
    run_name: str,
    iteration: int,
    raw_array_identifier: LocalArrayIdentifier,
    prediction_array_identifier: LocalArrayIdentifier,
    compute_context: ComputeContext = LocalTorch(),
):
    """Spawn a worker to predict on a given dataset.

    Args:
        model (Model): The model to use for prediction.
        raw_array (Array): The raw data to predict on.
        prediction_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
        compute_context (ComputeContext, optional): The compute context to use. Defaults to LocalTorch().
    """
    # Make the command for the worker to run
    command = [  # TODO
        "python",
        __file__,
        "start-worker",
        "--run-name",
        run_name,
        "--iteration",
        iteration,
        "--input_container",
        raw_array_identifier.container,
        "--input_dataset",
        raw_array_identifier.dataset,
        "--output_container",
        prediction_array_identifier.container,
        "--output_dataset",
        prediction_array_identifier.dataset,
        "--device",
        str(compute_context.device),
    ]

    def run_worker():
        # Run the worker in the given compute context
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
