from pathlib import Path
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.compute_context import create_compute_context

import daisy

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
@click.option("-th", "--threshold", type=float, default=0.0)
def start_worker(
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
    threshold: float = 0.0,
):
    # get arrays
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = ZarrArray.open_from_array_identifier(input_array_identifier)

    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )
    output_array = ZarrArray.open_from_array_identifier(output_array_identifier)

    # wait for blocks to run pipeline
    client = daisy.Client()

    while True:
        logger.warning("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            # write to output array
            output_array[block.write_roi] = (
                input_array[block.write_roi] > threshold
            ).astype(np.uint8)
            logger.warning(f"writing to {output_array_identifier} at {block.write_roi}")


def spawn_worker(
    input_array_identifier: "LocalArrayIdentifier",
    output_array_identifier: "LocalArrayIdentifier",
    threshold: float = 0.0,
):
    """Spawn a worker to predict on a given dataset.

    Args:
        model (Model): The model to use for prediction.
        raw_array (Array): The raw data to predict on.
        prediction_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
    """
    # compute_context = create_compute_context()

    # Make the command for the worker to run
    command = [
        "python",
        path,
        "start-worker",
        "--input_container",
        str(input_array_identifier.container),
        "--input_dataset",
        input_array_identifier.dataset,
        "--output_container",
        str(output_array_identifier.container),
        "--output_dataset",
        output_array_identifier.dataset,
        "--threshold",
        str(threshold),
    ]

    def run_worker():
        # Run the worker in the given compute context
        import subprocess
        name = f"threshold_{output_array_identifier.dataset}_{threshold}"
        import os
        if not os.path.exists("validate/logs/threshold"):
            os.makedirs("validate/logs/threshold")
        full_command = [
                "bsub",
                "-n", "14",
                 "-J", name,
                "-o", f"validate/logs/threshold/{name}.out",
                "-e", f"validate/logs/threshold/{name}.err",
                    "-P",
                    f"cellmap",
                ]+ command
        str_command = " ".join(full_command)
        logger.warning(f"Submitting: {str_command}")
        subprocess.run(full_command)
        # compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
