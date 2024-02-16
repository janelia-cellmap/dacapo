"""
Module for running and managing deep learning prediction tasks. It provides CLI for the same and 
also Python functions.

This module uses the DaCapo deep learning framework, Tensorflow and Gunpowder for its operations. 
It leverages on DaCapo for defining prediction models and training parameters, Tensorflow for 
running deep learning models, and Gunpowder for building and executing prediction pipelines.

The core operation of the module is done in the `start_worker` function which takes in input data and 
predicts the output by running a model.

Example usage:

As Python function:
```
start_worker(
  run_name="run1",
  iteration=10,
  input_container="dir1",
  input_dataset="data1",
  output_container="dir2",
  output_dataset="data2",
)
```

From CLI:
```
python dacapo_predict.py start-worker [--run-name "run1"] [--iteration 10] [--input_container "dir1"] 
[--input_dataset "data1"] [--output_container "dir2"] [--output_dataset "data2"]
```
"""

from pathlib import Path
from dacapo.experiments.datasplits.datasets.arrays.zarr_array import ZarrArray
from dacapo.gp.dacapo_array_source import DaCapoArraySource
from dacapo.store.array_store import LocalArrayIdentifier
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.experiments import Run
from dacapo.compute_context import ComputeContext, LocalTorch
import gunpowder as gp
import gunpowder.torch as gp_torch

import daisy
from daisy import Coordinate

import numpy as np
import click

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
    """
    Defining the command line interface group command.
    Provide options for the log level.

    Args:
        log_level (str): Logging level for the running tasks.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


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
@click.option(
    "-oc", "--output_container", required=True, type=click.Path(file_okay=False)
)
@click.option("-od", "--output_dataset", required=True, type=str)
@click.option("-d", "--device", type=str, default="cuda")
def start_worker(
    run_name: str,
    iteration: int,
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
    device: str = "cuda",
):
    """
    This is the main function taking in parameters for running a deep learning prediction model on 
    specified data and generating corresponding outputs.

    Args:
        run_name (str): Name of the run configuration.
        iteration (int): Training iteration to use for prediction.
        input_container (Path | str): File path to input container.
        input_dataset (str): Name of the dataset to use from the input container.
        output_container (Path | str): File path to output container where the predictions will be stored.
        output_dataset (str): Name of the dataset to use from the output container for prediction .
        device (str, optional): Name of the device to use for computations (ex: 'cuda', 'cpu'). Defaults to 'cuda'.
    """


def spawn_worker(
    run_name: str,
    iteration: int,
    raw_array_identifier: "LocalArrayIdentifier",
    prediction_array_identifier: "LocalArrayIdentifier",
    compute_context: ComputeContext = LocalTorch(),
):
    """
    Function to spawn a worker process for prediction.

    Args:
        run_name (str): The name of the model run.
        iteration (int): The model version or iteration.
        raw_array_identifier (LocalArrayIdentifier): Identifier for the raw input array.
        prediction_array_identifier (LocalArrayIdentifier): Identifier for the prediction output array.
        compute_context (ComputeContext, optional): Compute context to use for execution. Defaults to LocalTorch().
    """
    pass


if __name__ == "__main__":
    cli()