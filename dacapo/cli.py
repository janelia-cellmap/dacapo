```python
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
    """
    This is the main driver function for the dacapo library. It initializes the CLI and sets the logging 
    level for the entire program.

    Args:
        log_level (str): The level of logging to use while running the program. Defaults to INFO.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))

@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The NAME of the run to train."
)
def train(run_name):
    """
    This function starts the training of a model.

    Args:
        run_name (str): The name of the run to train.
    """
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
    """
    This function starts the validation of a trained model at a specific iteration.

    Args:
        run_name (str): The name of the run to validate.
        iteration (int): The iteration at which to validate the run.
    """
    dacapo.validate(run_name, iteration)


@cli.command()
# Additional click options omitted for brevity
def apply(
    run_name: str,
    # Other parameters omitted for brevity
):
    """
    This function applies a trained and validated model to a new dataset.

    Args:
        run_name (str): The name of the run (i.e., training session) to apply.
        input_container (Union[Path, str]): Path to the container with the input data.
        input_dataset (str): Name of the input dataset.
        output_path (Union[Path, str]): Path for the output.
    """
    # Full code omitted for brevity

@cli.command()
# Additional click options omitted for brevity
def predict(
    run_name: str,
    iteration: int,
    input_container: Path | str,
    input_dataset: str,
    output_path: Path | str,
    output_roi: Optional[str | Roi] = None,
    num_workers: int = 30,
    output_dtype: np.dtype | str = np.uint8,  # type: ignore
    compute_context: ComputeContext | str = LocalTorch(),
    overwrite: bool = True,
):
    """
    This function predicts the output for a given input dataset using the model trained at a specific 
    iteration.

    Args:
        run_name (str): The name of the run to use for prediction.
        iteration (int): The training iteration of the model to use for prediction.
        input_container (Union[Path, str]): The path to the container with input data for prediction.
        input_dataset (str): The specific input dataset to use for prediction.
        output_path (Union[Path, str]): The path where prediction output will be stored.
    """
    # Full code omitted for brevity
```