from examples.random_source_pipeline import random_source_pipeline
import gunpowder as gp


from pathlib import Path
import sys
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.compute_context import create_compute_context

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
    "-oc", "--output_container", required=True, type=click.Path(file_okay=False)
)
@click.option("-rod", "--raw_output_dataset", required=True, type=str)
@click.option("-lod", "--labels_output_dataset", required=True, type=str)
def start_worker(
    output_container: Path | str,
    raw_output_dataset: str,
    labels_output_dataset: str,
):
    # get arrays
    raw_output_array_identifier = LocalArrayIdentifier(
        Path(output_container), raw_output_dataset
    )
    raw_output_array = ZarrArray.open_from_array_identifier(raw_output_array_identifier)

    labels_output_array_identifier = LocalArrayIdentifier(
        Path(output_container), labels_output_dataset
    )
    labels_output_array = ZarrArray.open_from_array_identifier(
        labels_output_array_identifier
    )

    pipeline, request = random_source_pipeline()

    def batch_generator():
        with gp.build(pipeline):
            while True:
                yield pipeline.request_batch(request)

    batch_gen = batch_generator()

    # wait for blocks to run pipeline
    client = daisy.Client()

    while True:
        print("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            batch = next(batch_gen)
            raw_array = batch.arrays[gp.ArrayKey("RAW")]
            labels_array = batch.arrays[gp.ArrayKey("LABELS")]

            raw_data = raw_array.data
            raw_data -= raw_data.min()
            raw_data /= raw_data.max()
            raw_data *= 255
            raw_data = raw_data.astype(np.uint8)
            labels_data = labels_array.data.astype(np.uint32)

            # write to output array
            raw_output_array[block.write_roi] = raw_data
            labels_output_array[block.write_roi] = labels_data


def spawn_worker(
    raw_output_array_identifier: "LocalArrayIdentifier",
    labels_output_array_identifier: "LocalArrayIdentifier",
):
    """Spawn a worker to generate a synthetic dataset.

    Args:
        raw_output_array_identifier (LocalArrayIdentifier): The identifier of the raw output array.
        labels_output_array_identifier (LocalArrayIdentifier): The identifier of the labels output array.
    """
    compute_context = create_compute_context()

    # Make the command for the worker to run
    command = [
        # "python",
        sys.executable,
        path,
        "start-worker",
        "--output_container",
        raw_output_array_identifier.container,
        "--raw_output_dataset",
        raw_output_array_identifier.dataset,
        "--labels_output_dataset",
        labels_output_array_identifier.dataset,
    ]

    def run_worker():
        # Run the worker in the given compute context
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
