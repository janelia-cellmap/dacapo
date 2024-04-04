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
    """
    CLI for running the threshold worker.

    Args:
        log_level (str): The log level to use.
    Raises:
        NotImplementedError: If the method is not implemented in the derived class.
    Examples:
        >>> cli(log_level="INFO")
    Note:
        The method is implemented in the class.
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
    """
    Start the threshold worker.

    Args:
        input_container (Path | str): The input container.
        input_dataset (str): The input dataset.
        output_container (Path | str): The output container.
        output_dataset (str): The output dataset.
    Raises:
        NotImplementedError: If the method is not implemented in the derived class.
    Examples:
        >>> start_worker(input_container="input_container", input_dataset="input_dataset", output_container="output_container", output_dataset="output_dataset")
    Note:
        The method is implemented in the class.
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
):
    """
    Spawn a worker to predict on a given dataset.

    Args:
        model (Model): The model to use for prediction.
        raw_array (Array): The raw data to predict on.
        prediction_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
    Returns:    
        The worker to run.
    Raises:
        NotImplementedError: If the method is not implemented in the derived class.
    Examples:
        >>> spawn_worker(model, raw_array, prediction_array_identifier)
    Note:
        The method is implemented in the class.
    """
    compute_context = create_compute_context()

    # Make the command for the worker to run
    command = [
        # "python",
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
    ]

    def run_worker():
        """
        Run the worker in the given compute context.

        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> run_worker()
        Note:
            The method is implemented in the class.
        """
        # Run the worker in the given compute context
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
