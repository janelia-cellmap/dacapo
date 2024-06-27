from typing import Any, Optional
import sys
from dacapo.compute_context import create_compute_context

import daisy

import click

import logging

import skimage.measure
import skimage.filters
import skimage.morphology
from funlib.persistence import open_ds
import numpy as np
from skimage.segmentation import watershed
from scipy import ndimage as ndi

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
    "-pc",
    "--peroxi-container",
    required=True,
    type=str,
    default=None,
)
@click.option(
    "-pd",
    "--peroxi-dataset",
    required=True,
    type=str,
    default=None,
)
@click.option(
    "-mc",
    "--mito-container",
    required=True,
    type=str,
    default=None,
)
@click.option(
    "-md",
    "--mito-dataset",
    required=True,
    type=str,
    default=None,
)
@click.option(
    "-t",
    "--threshold",
    required=False,
    type=float,
    default=0.5,
)
@click.option(
    "-g",
    "--gaussian-kernel",
    required=False,
    type=int,
    default=2,
)
def start_worker(
    peroxi_container,
    peroxi_dataset,
    mito_container,
    mito_dataset,
    threshold,
    gaussian_kernel,
    return_io_loop: Optional[bool] = False,
):
    """
    Start the worker.

    Args:
        peroxi_container (str): The container of the peroxisome predictions.
        peroxi_dataset (str): The dataset of the peroxisome predictions.
        mito_container (str): The container of the mitochondria predictions.
        mito_dataset (str): The dataset of the mitochondria predictions.
        threshold (float): The threshold to use for the peroxisome predictions.
        gaussian_kernel (int): The kernel size to use for the gaussian filter.

    returns:
        instance_peroxi (np.ndarray): The instance labels of the peroxisome predictions.

    """
    # Do something with the argument
    # print(arg)

    def io_loop():
        # wait for blocks to run pipeline
        client = daisy.Client()
        peroxi_ds = open_ds(peroxi_container, peroxi_dataset)
        mito_ds = open_ds(mito_container, mito_dataset)

        while True:
            print("getting block")
            with client.acquire_block() as block:
                if block is None:
                    break

                # Do the blockwise process
                peroxi = peroxi_ds.to_ndarray(block.read_roi)
                mito = mito_ds.to_ndarray(block.read_roi)

                print(f"processing block: {block.id}, with read_roi: {block.read_roi}")
                peroxi = skimage.filters.gaussian(peroxi, gaussian_kernel)
                # threshold precictions
                binary_peroxi = peroxi > threshold
                # get instance labels
                markers, _ = ndi.label(binary_peroxi)
                # Apply Watershed
                ws_labels = watershed(-peroxi, markers, mask=peroxi)
                instance_peroxi = skimage.measure.label(ws_labels).astype(np.int64)
                # relabel background to 0
                instance_peroxi[mito > 0] = 0
                # make mask of unwanted object class overlaps
                return instance_peroxi.astype(np.uint64)

    if return_io_loop:
        return io_loop
    else:
        io_loop()


def spawn_worker(
    peroxi_container,
    peroxi_dataset,
    mito_container,
    mito_dataset,
    threshold,
    gaussian_kernel,
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
            peroxi_container=peroxi_container,
            peroxi_dataset=peroxi_dataset,
            mito_container=mito_container,
            mito_dataset=mito_dataset,
            threshold=threshold,
            gaussian_kernel=gaussian_kernel,
            return_io_loop=True,
        )

    # Make the command for the worker to run
    command = [
        sys.executable,
        path,
        "start-worker",
        "--peroxi-container",
        peroxi_container,
        "--peroxi-dataset",
        peroxi_dataset,
        "--mito-container",
        mito_container,
        "--mito-dataset",
        mito_dataset,
        "--threshold",
        str(threshold),
        "--gaussian-kernel",
        str(gaussian_kernel),
    ]

    def run_worker():
        """
        Run the worker in the given compute context.
        """
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
