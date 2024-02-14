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
    "--output_roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option("-w", "--num_workers", type=int, default=30)
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
    output_roi: Optional[str | Roi] = None,
    num_workers: int = 30,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    compute_context: ComputeContext | str = LocalTorch(),
    overwrite: bool = True,
):
    dacapo.predict(
        run_name,
        iteration,
        input_container,
        input_dataset,
        output_path,
        output_roi,
        num_workers,
        output_dtype,
        compute_context,
        overwrite,
    )
