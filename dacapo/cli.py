from pathlib import Path
from typing import Optional

import numpy as np

import dacapo
import click
import logging
from daisy import Roi
from dacapo.experiments.datasplits.datasets.dataset import Dataset
from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)


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
    dacapo.train(run_name)  # TODO: run with compute_context


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
@click.option("-w", "--num_workers", type=int, default=30)
@click.option("-dt", "--output_dtype", type=str, default="uint8")
@click.option("-ow", "--overwrite", is_flag=True)
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
@click.option("-w", "--num_workers", type=int, default=30)
@click.option("-dt", "--output_dtype", type=str, default="uint8")
@click.option("-ow", "--overwrite", is_flag=True)
def apply(
    run_name: str,
    input_container: Path | str,
    input_dataset: str,
    output_path: Path | str,
    validation_dataset: Optional[Dataset | str] = None,
    criterion: str = "voi",
    iteration: Optional[int] = None,
    parameters: Optional[PostProcessorParameters | str] = None,
    roi: Optional[Roi | str] = None,
    num_workers: int = 30,
    output_dtype: np.dtype | str = "uint8",
    overwrite: bool = True,
):
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
        num_workers,
        output_dtype,
        overwrite=overwrite,
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
@click.option("-ow", "--overwrite", is_flag=True)
def predict(
    run_name: str,
    iteration: int,
    input_container: Path | str,
    input_dataset: str,
    output_path: Path | str,
    output_roi: Optional[str | Roi] = None,
    num_workers: int = 30,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
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
        overwrite,
    )
