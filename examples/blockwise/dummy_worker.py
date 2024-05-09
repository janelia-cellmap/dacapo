from typing import Any, Optional
import sys
from dacapo.compute_context import create_compute_context

import daisy

import click

import logging

logger = logging.getLogger(__file__)

read_write_conflict: bool = False
fit: str = "valid"
path = __file__

# OPTIONALLY DEFINE GLOBALS HERE


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
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option(
    "-a",
    "--arg",
    required=True,
    type=any,
    default=None,
)
# ADD MORE CLICK OPTION ARGUMENTS HERE
def start_worker(
    arg: Any,
    # ADD MORE ARGUMENTS HERE
    return_io_loop: Optional[bool] = False,
):
    """
    Start the worker.

    Args:
        arg (Any): An example argument to use.

    """
    # Do something with the argument
    print(arg)

    def io_loop():
        # wait for blocks to run pipeline
        client = daisy.Client()

        while True:
            print("getting block")
            with client.acquire_block() as block:
                if block is None:
                    break

                # Do the blockwise process
                print(
                    f"processing block: {block.id}, with read_roi: {block.read_roi}, using arg: {arg}"
                )
                # DO SOMETHING WITH THE BLOCK

    if return_io_loop:
        return io_loop
    else:
        io_loop()


def spawn_worker(
    arg: Any,
    # ADD MORE ARGUMENTS HERE
):
    """
    Spawn a worker.

    Args:
        arg (Any): An example argument to use.
    Returns:
        Callable: The function to run the worker.
    """
    compute_context = create_compute_context()
    if not compute_context.distribute_workers:
        return start_worker(
            arg=arg,
            # ADD MORE ARGUMENTS HERE
            return_io_loop=True,
        )

    # Make the command for the worker to run
    command = [
        sys.executable,
        path,
        "start-worker",
        "--arg",
        str(arg),
        # ADD MORE ARGUMENTS HERE, THEY MUST BE STRINGS
    ]

    def run_worker():
        """
        Run the worker in the given compute context.
        """
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
