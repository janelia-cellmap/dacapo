"""This module is a part of dacapo python library used in running prediction using a trained model.
It defines two key functions start_worker and spawn_worker which helps in initializing a worker
which will use the model to predict on given dataset. It utilizes click library for creating 
command line interface. 

Functions: 
    cli() - Entry point for script's command group
    start_worker() - Starts a worker for running prediction on a given dataset. Requires multiple input arguments
                     including input_container, input_dataset, output_container, ouput_dataset.
    spawn_worker() - Creates a command to run worker and execute the command in given compute context.

Example: 
    Command to use start_worker: 
    python <filename> start-worker --input_container <input_containter_path> --input_dataset <input_dataset_path>
                                   --output_container <output_containter_path> --output_dataset <output_dataset_path>
"""

from pathlib import Path
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.compute_context import ComputeContext, LocalTorch

import daisy

import numpy as np
import click

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
    """Base command groups on click CLI.
    
    Args:
        log_level (str): Logging level of the logger. Can be one of ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    """
    
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
def start_worker(
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
):
    """Command to start worker to run prediction on a given dataset.

    Args:
        input_container (Path | str): Path to the input container (i.e., directory path containing the input data).
        input_dataset (str): Name or path of the input dataset.
        output_container (Path | str): Path to the output container (i.e., directory path where output data will be stored).
        output_dataset (str): Name or path for the output dataset.
    """

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
        print("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            # write to output array
            output_array[block.write_roi] = np.argmax(
                input_array[block.write_roi],
                axis=input_array.axes.index("c"),
            )


def spawn_worker(
    input_array_identifier: "LocalArrayIdentifier",
    output_array_identifier: "LocalArrayIdentifier",
    compute_context: ComputeContext = LocalTorch(),
):
    """Spawn a worker to predict on a given dataset.

    Args:
        input_array_identifier (LocalArrayIdentifier): Identifier of the input array (data).
        output_array_identifier (LocalArrayIdentifier): Identifier of the output array (prediction results).
        compute_context (ComputeContext, optional): Computing context where worker executes. Defaults to LocalTorch().
    """
    # Make the command for the worker to run
    command = [
        "python",
        __file__,
        "start-worker",
        "--input_container",
        input_array_identifier.container,
        "--input_dataset",
        input_array_identifier.dataset,
        "--output_container",
        output_array_identifier.container,
        "--output_dataset",
        output_array_identifier.dataset,
    ]

    def run_worker():
        """Internal function to run the worker command."""
        # Run the worker in the given compute context
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
