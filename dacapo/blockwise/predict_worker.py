import subprocess
from typing import Optional
import dacapo
from dacapo.experiments.model import Model
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.experiments import Run
from dacapo.compute_context import ComputeContext, LocalTorch, Bsub

import daisy
from daisy import Roi
from funlib.persistence import open_ds, Array

from skimage.transform import rescale  # TODO
import numpy as np
import torch
import click

import sys
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
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option("-n", "--name", type=str)
@click.option("-c", "--criterion", type=str)
@click.option("-cs", "--channels", type=str)
@click.option("-oc", "--out_container", type=click.Path(exists=True, file_okay=False))
@click.option("-od", "--out_dataset", type=str)
@click.option("-ic", "--in_container", type=click.Path(exists=True, file_okay=False))
@click.option("-id", "--in_dataset", type=str)
@click.option("--min-raw", type=float, default=0)
@click.option("--max-raw", type=float, default=255)
@click.option(
    "-mc",
    "--mask-container",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option(
    "-md",
    "--mask-dataset",
    type=click.Path(file_okay=False),
    multiple=True,
    default=list(),
)
@click.option(
    "--instance",
    type=bool,
    default=False,
)
def start_worker(
    name,
    criterion,
    channels,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    min_raw,
    max_raw,
    mask_container,
    mask_dataset,
    instance,
):
    shift = min_raw
    scale = max_raw - min_raw
    parsed_channels = [channel.split(":") for channel in channels.split(",")]

    device = torch.device("cuda")

    client = daisy.Client()

    config_store = create_config_store()
    weights_store = create_weights_store()

    run_config = config_store.retrieve_run_config(name)
    run = Run(run_config)

    model = run.model
    try:
        weights_store._load_best(run, criterion)
    except FileNotFoundError:
        iteration = int(criterion)
        weights = weights_store.retrieve_weights(run, iteration)
        model.load_state_dict(weights.model)

    model = run.model.to(device)

    raw_dataset = open_ds(in_container, in_dataset)
    mask_datasets = [open_ds(mc, md) for mc, md in zip(mask_container, mask_dataset)]

    voxel_size = raw_dataset.voxel_size
    output_voxel_size = model.scale(voxel_size)

    if not instance:
        out_datasets = [
            open_ds(
                out_container,
                f"{out_dataset}/{channel}",
                mode="r+",
            )
            for _, channel in parsed_channels
        ]
    else:
        num_channels = model.num_out_channels
        assert len(parsed_channels) == 1
        indexes, channel = parsed_channels[0]
        out_datasets = [
            open_ds(out_container, f"{out_dataset}/{channel}__{i}", mode="r+")
            for i in range(0, num_channels, 3)
        ]

    while True:
        with client.acquire_block() as block:
            if block is None:
                break
            if len(mask_datasets) > 0:
                mask_data = any(
                    [
                        np.any(
                            mask_dataset.to_ndarray(
                                roi=block.read_roi.snap_to_grid(
                                    mask_dataset.voxel_size
                                ),
                                fill_value=0,
                            )
                        )
                        for mask_dataset in mask_datasets
                    ]
                )
            else:
                mask_data = 1
            if not np.any(mask_data):
                # avoid predicting if mask is empty
                continue

            if not instance:
                raw_input = (
                    2.0
                    * (
                        raw_dataset.to_ndarray(
                            roi=block.read_roi, fill_value=shift + scale
                        ).astype(np.float32)
                        - shift
                    )
                    / scale
                ) - 1.0
            else:
                raw_input = (
                    raw_dataset.to_ndarray(
                        roi=block.read_roi, fill_value=shift + scale
                    ).astype(np.float32)
                    - shift
                ) / scale
            raw_input = np.expand_dims(raw_input, (0, 1))
            write_roi = block.write_roi.intersect(out_datasets[0].roi)

            if out_datasets[0].to_ndarray(write_roi).any():
                # block has already been processed
                continue

            with torch.no_grad():
                predictions = Array(
                    model.forward(torch.from_numpy(raw_input).float().to(device))
                    .detach()
                    .cpu()
                    .numpy()[0],
                    block.write_roi,
                    output_voxel_size,
                )

                write_data = predictions.to_ndarray(write_roi).clip(-1, 1)
                if not instance:
                    write_data = (write_data + 1) * 255.0 / 2.0
                    for (i, _), out_dataset in zip(parsed_channels, out_datasets):
                        indexes = []
                        if "-" in i:
                            indexes = [int(j) for j in i.split("-")]
                        else:
                            indexes = [int(i)]
                        if len(indexes) > 1:
                            out_dataset[write_roi] = np.stack(
                                [write_data[j] for j in indexes], axis=0
                            ).astype(np.uint8)
                        else:
                            out_dataset[write_roi] = write_data[indexes[0]].astype(
                                np.uint8
                            )
                else:
                    for i, out_dataset in zip(range(0, num_channels, 3), out_datasets):
                        out_dataset[write_roi] = write_data[i : i + 3].astype(
                            np.float32
                        )

            block.status = daisy.BlockStatus.SUCCESS


def spawn_worker(
    model: Model,
    raw_array: Array,
    prediction_array_identifier: LocalArrayIdentifier,
    num_cpu_workers: int = 4,
    compute_context: ComputeContext = LocalTorch(),
    output_roi: Optional[Roi] = None,
    output_dtype: np.dtype = np.float32,  # type: ignore
    overwrite: bool = False,
):
    """Spawn a worker to predict on a given dataset.

    Args:
        model (Model): The model to use for prediction.
        raw_array (Array): The raw data to predict on.
        prediction_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
        num_cpu_workers (int, optional): The number of CPU workers to use. Defaults to 4.
        compute_context (ComputeContext, optional): The compute context to use. Defaults to LocalTorch().
        output_roi (Optional[Roi], optional): The ROI to predict on. Defaults to None.
        output_dtype (np.dtype, optional): The dtype of the output. Defaults to np.float32.
        overwrite (bool, optional): Whether to overwrite the prediction array. Defaults to False.
    """
    # Make the command for the worker to run
    command = [  # TODO
        # "dacapo",
        # "predict",
        # "--name",
        # model.name,
        # "--criterion",
        # "best",
        # "--channels",
        # "0",
        # "--out_container",
        # prediction_array_identifier.container,
        # "--out_dataset",
        # prediction_array_identifier.dataset,
        # "--in_container",
        # raw_array.container,
        # "--in_dataset",
        # raw_array.dataset,
    ]

    def run_worker():
        # Run the worker in the given compute context
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
