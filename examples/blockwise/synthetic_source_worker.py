from dacapo.utils.pipeline import random_source_pipeline
import gunpowder as gp


from pathlib import Path
import sys

from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.compute_context import create_compute_context
from dacapo.tmp import create_from_identifier, open_from_identifier
import dacapo

import daisy
from funlib.geometry import Coordinate, Roi

import numpy as np
import click

import logging

logger = logging.getLogger(__file__)

read_write_conflict: bool = False
fit: str = "shrink"
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


fit = "valid"


def generate_synthetic_dataset(
    output_container: Path | str,
    raw_output_dataset: str = "raw",
    labels_output_dataset: str = "labels",
    shape: str | Coordinate = Coordinate((512, 512, 512)),
    voxel_size: str | Coordinate = Coordinate((8, 8, 8)),
    write_shape: str | Coordinate = Coordinate((256, 256, 256)),
    num_workers: int = 16,
    overwrite: bool = False,
):
    # get ROI from string
    if isinstance(voxel_size, str):
        _voxel_size = Coordinate([int(v) for v in voxel_size.split(",")])
    else:
        _voxel_size = voxel_size
    if isinstance(shape, str):
        _shape = Coordinate([int(v) for v in shape.split(",")])
    else:
        _shape = shape
    if isinstance(write_shape, str):
        _write_shape = Coordinate([int(v) for v in write_shape.split(",")])
    else:
        _write_shape = write_shape
    roi = Roi((0, 0, 0), _shape * _voxel_size)
    read_roi = write_roi = Roi((0, 0, 0), _write_shape * _voxel_size)

    # get arrays
    raw_output_array_identifier = LocalArrayIdentifier(
        Path(output_container), raw_output_dataset
    )
    raw_output_array = create_from_identifier(
        raw_output_array_identifier,
        roi=roi,
        dtype=np.uint8,
        voxel_size=_voxel_size,
        num_channels=None,
        axis_names=["z", "y", "x"],
        overwrite=overwrite,
        write_size=_write_shape * voxel_size,
    )

    labels_output_array_identifier = LocalArrayIdentifier(
        Path(output_container), labels_output_dataset
    )
    labels_output_array = create_from_identifier(
        labels_output_array_identifier,
        roi=roi,
        dtype=np.uint64,
        voxel_size=_voxel_size,
        num_channels=None,
        axis_names=["z", "y", "x"],
        overwrite=overwrite,
        write_size=_write_shape * voxel_size,
    )

    # make daisy blockwise task
    dacapo.run_blockwise(
        __file__,
        roi,
        read_roi,
        write_roi,
        num_workers=num_workers,
        raw_output_array_identifier=raw_output_array_identifier,
        labels_output_array_identifier=labels_output_array_identifier,
    )


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
    raw_output_array = open_from_identifier(raw_output_array_identifier)

    labels_output_array_identifier = LocalArrayIdentifier(
        Path(output_container), labels_output_dataset
    )
    labels_output_array = open_from_identifier(labels_output_array_identifier)

    # get data generator

    def batch_generator(shape=(128, 128, 128), voxel_size=(8, 8, 8)):
        pipeline, request = random_source_pipeline(
            input_shape=shape, voxel_size=voxel_size
        )
        with gp.build(pipeline):
            while True:
                yield pipeline.request_batch(request)

    batch_gen = None

    id_offset = None

    # wait for blocks to run pipeline
    client = daisy.Client()

    while True:
        print("getting block")
        with client.acquire_block() as block:
            if block is None:
                break

            if batch_gen is None or id_offset is None:
                size = block.write_roi.get_shape()
                voxel_size = raw_output_array.voxel_size
                shape = Coordinate(size / voxel_size)
                batch_gen = batch_generator(
                    shape=shape,
                    voxel_size=voxel_size,
                )
                id_offset = np.prod(shape)  # number of voxels in the block
            batch = next(batch_gen)
            raw_array = batch.arrays[gp.ArrayKey("RAW")]
            labels_array = batch.arrays[gp.ArrayKey("LABELS")]

            raw_data = raw_array.data
            raw_data -= raw_data.min()
            raw_data /= raw_data.max()
            raw_data *= 255
            raw_data = raw_data.astype(np.uint8)
            labels_data = labels_array.data.astype(np.uint64)
            labels_data += np.uint64(id_offset * block.block_id[1])
            labels_data[labels_data == np.uint64(id_offset * block.block_id[1])] = 0

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
