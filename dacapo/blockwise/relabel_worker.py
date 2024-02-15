from glob import glob
import os
import daisy
from dacapo.compute_context import ComputeContext, LocalTorch
from dacapo.store.array_store import LocalArrayIdentifier
from funlib.segment.arrays.impl import find_components
from funlib.segment.arrays.replace_values import replace_values
from funlib.persistence import open_ds

import numpy as np

import logging
import click


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


fit = "shrink"
read_write_conflict = False


@cli.command()
@click.option("--output_container", type=str, help="Output container")
@click.option("--output_dataset", type=str, help="Output dataset")
@click.option("--tmpdir", type=str, help="Temporary directory")
def start_worker(
    output_container,
    output_dataset,
    tmpdir,
    *args,
    **kwargs,
):
    client = daisy.Client()
    array_out = open_ds(output_container, output_dataset, mode="a")

    nodes, edges = read_cross_block_merges(tmpdir)

    components = find_components(nodes, edges)

    while True:
        with client.acquire_block() as block:
            if block is None:
                break

            relabel_in_block(array_out, nodes, components, block)


def relabel_in_block(array_out, old_values, new_values, block):
    a = array_out.to_ndarray(block.write_roi)
    replace_values(a, old_values, new_values, inplace=True)
    array_out[block.write_roi] = a


def read_cross_block_merges(tmpdir):
    block_files = glob(os.path.join(tmpdir, "block_*.npz"))

    nodes = []
    edges = []
    for block_file in block_files:
        b = np.load(block_file)
        nodes.append(b["nodes"])
        edges.append(b["edges"])

    return np.concatenate(nodes), np.concatenate(edges)


def spawn_worker(
    output_array_identifier: LocalArrayIdentifier,
    tmpdir: str,
    compute_context: ComputeContext = LocalTorch(),
    *args,
    **kwargs,
):
    """Spawn a worker to predict on a given dataset.

    Args:
        output_array_identifier (LocalArrayIdentifier): The output array identifier
        tmpdir (str): The temporary directory
        compute_context (ComputeContext, optional): The compute context. Defaults to LocalTorch().
    """
    # Make the command for the worker to run
    command = [
        "python",
        __file__,
        "start-worker",
        "--output_container",
        output_array_identifier.container,
        "--output_dataset",
        output_array_identifier.dataset,
        "--tmpdir",
        tmpdir,
    ]

    def run_worker():
        # Run the worker in the given compute context
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
