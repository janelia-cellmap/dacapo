from glob import glob
import os
import sys
from time import sleep
import daisy
from dacapo.compute_context import create_compute_context
from dacapo.store.array_store import LocalArrayIdentifier
from scipy.cluster.hierarchy import DisjointSet
from funlib.persistence import open_ds

import numpy as np
import numpy_indexed as npi

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
path = __file__


@cli.command()
@click.option("--output_container", type=str, help="Output container")
@click.option("--output_dataset", type=str, help="Output dataset")
@click.option("--tmpdir", type=str, help="Temporary directory")
def start_worker(
    output_container,
    output_dataset,
    tmpdir,
    return_io_loop=False,
):
    return start_worker_fn(
        output_container=output_container,
        output_dataset=output_dataset,
        tmpdir=tmpdir,
        return_io_loop=return_io_loop,
    )


def start_worker_fn(
    output_container,
    output_dataset,
    tmpdir,
    return_io_loop=False,
):
    
    client = daisy.Client()
    array_out = open_ds(output_container, output_dataset, mode="a")

    nodes, edges = read_cross_block_merges(tmpdir)

    components = find_components(nodes, edges)

    def io_loop():
        client = daisy.Client()
        while True:
            with client.acquire_block() as block:
                if block is None:
                    break

                try:
                    relabel_in_block(array_out, nodes, components, block)
                except OSError as e:
                    logging.error(
                        f"Failed to relabel block {block.write_roi}: {e}. Trying again."
                    )
                    sleep(1)
                    relabel_in_block(array_out, nodes, components, block)

    if return_io_loop:
        return io_loop
    else:
        io_loop()


def relabel_in_block(array_out, old_values, new_values, block):
    
    a = array_out.to_ndarray(block.write_roi)
    # DGA: had to add in flatten and reshape since remap (in particular indices) didn't seem to work with ndarrays for the input
    if old_values.size > 0:
        a = npi.remap(a.flatten(), old_values, new_values).reshape(a.shape)
    array_out[block.write_roi] = a


def find_components(nodes, edges):
    
    # scipy
    disjoint_set = DisjointSet(nodes)
    for edge in edges:
        disjoint_set.merge(edge[0], edge[1])
    return [disjoint_set[n] for n in nodes]


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
):
    
    compute_context = create_compute_context()

    if not compute_context.distribute_workers:
        return start_worker_fn(
            output_array_identifier.container,
            output_array_identifier.dataset,
            tmpdir,
            return_io_loop=True,
        )

    # Make the command for the worker to run
    command = [
        # "python",
        sys.executable,
        path,
        "start-worker",
        "--output_container",
        output_array_identifier.container,
        "--output_dataset",
        output_array_identifier.dataset,
        "--tmpdir",
        tmpdir,
    ]

    def run_worker():
        
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
