from glob import glob
import os
import sys
import daisy
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
def start_worker(output_container, output_dataset, tmpdir):
    client = daisy.Client()
    array_out = open_ds(output_container, output_dataset, mode="a")

    nodes, edges = read_cross_block_merges(tmpdir)

    components = find_components(nodes, edges)

    print(f"Num nodes: {len(nodes)}")
    print(f"Num edges: {len(edges)}")
    print(f"Num components: {len(components)}")

    while True:
        print("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            print(f"Segmenting in block {block}")

            relabel_in_block(array_out, nodes, components, block)
    print("worker finished.")


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


if __name__ == "__main__":
    cli()
