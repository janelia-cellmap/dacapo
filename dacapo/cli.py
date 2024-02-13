from pathlib import Path
from typing import Optional

import numpy as np

import dacapo
import click
import logging
from daisy import Roi, Coordinate
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from dacapo.experiments.datasplits.datasets.dataset import Dataset
from dacapo.experiments.run import Run
from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo import compute_context
from dacapo.compute_context import ComputeContext, LocalTorch


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
    "-r", "--run-name", required=True, type=str, help="The NAME of the run to train."
)
def train(run_name):
    dacapo.train(run_name)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The name of the run to validate."
)
@click.option(
    "-i",
    "--iteration",
    required=True,
    type=int,
    help="The iteration at which to validate the run.",
)
def validate(run_name, iteration):
    dacapo.validate(run_name, iteration)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The name of the run to apply."
)
@click.option(
    "-ic",
    "--input_container",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-id", "--input_dataset", required=True, type=str)
@click.option("-op", "--output_path", required=True, type=click.Path(file_okay=False))
@click.option("-vd", "--validation_dataset", type=str, default=None)
@click.option("-c", "--criterion", default="voi")
@click.option("-i", "--iteration", type=int, default=None)
@click.option("-p", "--parameters", type=str, default=None)
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option("-w", "--num_cpu_workers", type=int, default=30)
@click.option("-dt", "--output_dtype", type=str, default="uint8")
@click.option("-ow", "--overwrite", is_flag=True)
@click.option("-cc", "--compute_context", type=str, default="LocalTorch")
def apply(
    run_name: str,
    input_container: Path or str,
    input_dataset: str,
    output_path: Path or str,
    validation_dataset: Optional[Dataset or str] = None,
    criterion: Optional[str] = "voi",
    iteration: Optional[int] = None,
    parameters: Optional[PostProcessorParameters or str] = None,
    roi: Optional[Roi or str] = None,
    num_cpu_workers: int = 30,
    output_dtype: Optional[np.dtype | str] = "uint8",
    overwrite: bool = True,
    compute_context: Optional[ComputeContext | str] = LocalTorch(),
):
    if isinstance(compute_context, str):
        compute_context = getattr(compute_context, compute_context)()

    dacapo.apply(
        run_name,
        input_container,
        input_dataset,
        output_path,
        validation_dataset,
        criterion,
        iteration,
        parameters,
        roi,
        num_cpu_workers,
        output_dtype,
        overwrite=overwrite,
        compute_context=compute_context,  # type: ignore
    )


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The name of the run to apply."
)
@click.option(
    "-i",
    "--iteration",
    required=True,
    type=int,
    help="The training iteration of the model to use for prediction.",
)
@click.option(
    "-ic",
    "--input_container",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-id", "--input_dataset", required=True, type=str)
@click.option("-op", "--output_path", required=True, type=click.Path(file_okay=False))
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option("-w", "--num_cpu_workers", type=int, default=30)
@click.option("-dt", "--output_dtype", type=str, default="uint8")
@click.option(
    "-cc",
    "--compute_context",
    type=str,
    default="LocalTorch",
    help="The compute context to use for prediction. Must be the name of a subclass of ComputeContext.",
)
@click.option("-ow", "--overwrite", is_flag=True)
def predict(
    run_name: str,
    iteration: int,
    input_container: Path or str,
    input_dataset: str,
    output_path: Path or str,
    roi: Optional[str | Roi] = None,
    num_cpu_workers: int = 30,
    output_dtype: Optional[np.dtype | str] = np.uint8,  # type: ignore
    compute_context: Optional[ComputeContext | str] = LocalTorch(),
    overwrite: bool = True,
):
    # retrieving run
    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(run_name)
    run = Run(run_config)

    # create weights store
    weights_store = create_weights_store()

    # load weights
    weights_store.retrieve_weights(run_name, iteration)

    # get arrays
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = ZarrArray.open_from_array_identifier(input_array_identifier)
    output_container = Path(
        output_path,
        "".join(Path(input_container).name.split(".")[:-1]) + f".zarr",
    )  # TODO: zarr hardcoded
    prediction_array_identifier = LocalArrayIdentifier(
        output_container, f"prediction_{run_name}_{iteration}"
    )

    if isinstance(roi, str):
        start, end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in roi.strip("[]").split(",")
            ]
        )
        roi = Roi(
            Coordinate(start),
            Coordinate(end) - Coordinate(start),
        )

    if roi is None:
        roi = input_array.roi
    else:
        roi = roi.snap_to_grid(input_array.voxel_size, mode="grow").intersect(
            input_array.roi
        )

    if isinstance(output_dtype, str):
        output_dtype = np.dtype(output_dtype)

    if isinstance(compute_context, str):
        compute_context = getattr(compute_context, compute_context)()

    dacapo.predict(
        run.model,
        input_array,
        prediction_array_identifier,
        output_roi=roi,
        num_cpu_workers=num_cpu_workers,
        output_dtype=output_dtype,
        compute_context=compute_context,  # type: ignore
        overwrite=overwrite,
    )
