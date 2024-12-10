from upath import UPath as Path
import sys

from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.compute_context import create_compute_context
from dacapo.tmp import open_from_identifier

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
    return_io_loop: bool = False,
):
    return start_worker_fn(
        input_container=input_container,
        input_dataset=input_dataset,
        output_container=output_container,
        output_dataset=output_dataset,
        threshold=threshold,
        return_io_loop=return_io_loop,
    )


def start_worker_fn(
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
    threshold: float = 0.0,
    return_io_loop: bool = False,
):
    """
    Start the threshold worker.

    Args:
        input_container (Path | str): The input container.
        input_dataset (str): The input dataset.
        output_container (Path | str): The output container.
        output_dataset (str): The output dataset.
        threshold (float): The threshold.

    """
    # get arrays
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = open_from_identifier(input_array_identifier)

    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )
    output_array = open_from_identifier(output_array_identifier)

    def io_loop():
        # wait for blocks to run pipeline
        client = daisy.Client()

        while True:
            print("getting block")
            with client.acquire_block() as block:
                if block is None:
                    break

                # write to output array
                output_array[block.write_roi] = (
                    input_array[block.write_roi] > threshold
                ).astype(np.uint8)

    if return_io_loop:
        return io_loop
    else:
        io_loop()


def spawn_worker(
    input_array_identifier: "LocalArrayIdentifier",
    output_array_identifier: "LocalArrayIdentifier",
    threshold: float = 0.0,
):
    """
    Spawn a worker to predict on a given dataset.

    Args:
        input_array_identifier (LocalArrayIdentifier): The raw data to predict on.
        output_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
        threshold (float): The threshold.
    Returns:
        Callable: The function to run the worker.
    """
    compute_context = create_compute_context()
    if not compute_context.distribute_workers:
        return start_worker_fn(
            input_array_identifier.container,
            input_array_identifier.dataset,
            output_array_identifier.container,
            output_array_identifier.dataset,
            return_io_loop=True,
        )

    # Make the command for the worker to run
    command = [
        sys.executable,
        path,
        "start-worker",
        "--input_container",
        input_array_identifier.container,
        "--input_dataset",
        input_array_identifier.dataset,
        "--output_container",
        output_array_identifier.container,
        "--output_dataset",
        output_array_identifier.dataset,
        "--threshold",
        threshold,
    ]

    def run_worker():
        """
        Run the worker in the given compute context.
        """
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
