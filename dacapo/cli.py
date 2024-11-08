from upath import UPath as Path
from typing import Optional

import numpy as np
import yaml
import dacapo
import click
import logging
from funlib.geometry import Roi, Coordinate
from funlib.persistence import Array
from dacapo.experiments.datasplits.datasets.dataset import Dataset
from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)
from dacapo.blockwise import (
    run_blockwise as _run_blockwise,
    segment_blockwise as _segment_blockwise,
)
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.tmp import open_from_identifier, create_from_identifier

from dacapo.options import DaCapoConfig
import os


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
    Command-line interface for the DACAPO application.

    Args:
        log_level (str): The desired log level for the application.
    Examples:
        To train a model, run:
        ```
        dacapo train --run-name my_run
        ```

        To validate a model, run:
        ```
        dacapo validate --run-name my_run --iteration 100
        ```

        To apply a model, run:
        ```
        dacapo apply --run-name my_run --input-container /path/to/input --input-dataset my_dataset --output-path /path/to/output
        ```

        To predict with a model, run:
        ```
        dacapo predict --run-name my_run --iteration 100 --input-container /path/to/input --input-dataset my_dataset --output-path /path/to/output
        ```

        To run a blockwise operation, run:
        ```
        dacapo run-blockwise --input-container /path/to/input --input-dataset my_dataset --output-container /path/to/output --output-dataset my_output --worker-file /path/to/worker.py --total-roi [0:100,0:100,0:100] --read-roi-size [10,10,10] --write-roi-size [10,10,10] --num-workers 16
        ```

        To segment blockwise, run:
        ```
        dacapo segment-blockwise --input-container /path/to/input --input-dataset my_dataset --output-container /path/to/output --output-dataset my_output --segment-function-file /path/to/segment_function.py --total-roi [0:100,0:100,0:100] --read-roi-size [10,10,10] --write-roi-size [10,10,10] --num-workers 16
        ```
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


logger = logging.getLogger(__name__)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The NAME of the run to train."
)
@click.option(
    "--no-validation", is_flag=True, help="Disable validation after training."
)
def train(run_name, no_validation):
    """
    Train a model with the specified run name.

    Args:
        run_name (str): The name of the run to train.
        no_validation (bool): Flag to disable validation after training.
    """
    do_validate = not no_validation
    dacapo.train(run_name, do_validate=do_validate)


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
def validate(run_name, iteration, num_workers, output_dtype, overwrite):
    dacapo.validate_run(run_name, iteration, num_workers, output_dtype, overwrite)


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The name of the run to apply."
)
@click.option(
    "-ic",
    "--input_container",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The path to the input container.",
)
@click.option(
    "-id",
    "--input_dataset",
    required=True,
    type=str,
    help="The name of the input dataset.",
)
@click.option(
    "-op",
    "--output_path",
    required=True,
    type=click.Path(file_okay=False),
    help="The path to the output directory.",
)
@click.option(
    "-vd",
    "--validation_dataset",
    type=str,
    default=None,
    help="The name of the validation dataset.",
)
@click.option(
    "-c",
    "--criterion",
    default="voi",
    help="The criterion to use for applying the run.",
)
@click.option(
    "-i",
    "--iteration",
    type=int,
    default=None,
    help="The iteration of the model to use for prediction.",
)
@click.option(
    "-p",
    "--parameters",
    type=str,
    default=None,
    help="The parameters for the post-processor.",
)
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option(
    "-w",
    "--num_workers",
    type=int,
    default=30,
    help="The number of workers to use for prediction.",
)
@click.option(
    "-dt",
    "--output_dtype",
    type=str,
    default="uint8",
    help="The output data type.",
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    help="Whether to overwrite existing output files.",
)
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
    """
    Apply a trained run to an input dataset.

    Args:
        run_name (str): The name of the run to apply.
        input_container (Path | str): The path to the input container.
        input_dataset (str): The name of the input dataset.
        output_path (Path | str): The path to the output directory.
        validation_dataset (Dataset | str, optional): The name of the validation dataset. Defaults to None.
        criterion (str, optional): The criterion to use for applying the run. Defaults to "voi".
        iteration (int, optional): The iteration of the model to use for prediction. Defaults to None.
        parameters (PostProcessorParameters | str, optional): The parameters for the post-processor. Defaults to None.
        roi (Roi | str, optional): The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]. Defaults to None.
        num_workers (int, optional): The number of workers to use for prediction. Defaults to 30.
        output_dtype (np.dtype | str, optional): The output data type. Defaults to "uint8".
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
    Raises:
        ValueError: If the run_name is not valid.
    Examples:
        To apply a trained run to an input dataset, run:
        ```
        dacapo apply --run-name my_run --input-container /path/to/input --input-dataset my_dataset --output-path /path/to/output
        ```
    """
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
    help="The path to the input container.",
)
@click.option(
    "-id",
    "--input_dataset",
    required=True,
    type=str,
    help="The name of the input dataset.",
)
@click.option(
    "-op",
    "--output_path",
    required=True,
    type=click.Path(file_okay=False),
    help="The path to the output directory.",
)
@click.option(
    "-roi",
    "--output_roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option(
    "-w",
    "--num_workers",
    type=int,
    default=30,
    help="The number of workers to use for prediction.",
)
@click.option(
    "-dt", "--output_dtype", type=str, default="uint8", help="The output data type."
)
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    help="Whether to overwrite existing output files.",
)
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
    """
    Apply a trained model to predict on a dataset.

    Args:
        run_name (str): The name of the run to apply.
        iteration (int): The training iteration of the model to use for prediction.
        input_container (Path | str): The path to the input container.
        input_dataset (str): The name of the input dataset.
        output_path (Path | str): The path to the output directory.
        output_roi (Optional[str | Roi], optional): The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]. Defaults to None.
        num_workers (int, optional): The number of workers to use for prediction. Defaults to 30.
        output_dtype (np.dtype | str, optional): The output data type. Defaults to np.uint8.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
    Raises:
        ValueError: If the run_name is not valid.
    Examples:
        To predict with a model, run:
        ```
        dacapo predict --run-name my_run --iteration 100 --input-container /path/to/input --input-dataset my_dataset --output-path /path/to/output
        ```
    """
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


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "-ic",
    "--input_container",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The path to the input container.",
)
@click.option(
    "-id",
    "--input_dataset",
    required=True,
    type=str,
    help="The name of the input dataset.",
)
@click.option(
    "-oc",
    "--output_container",
    required=True,
    type=click.Path(file_okay=False),
    help="The path to the output container.",
)
@click.option(
    "-od",
    "--output_dataset",
    required=True,
    type=str,
    help="The name of the output dataset.",
)
@click.option(
    "-w", "--worker_file", required=True, type=str, help="The path to the worker file."
)
@click.option(
    "-tr",
    "--total_roi",
    required=True,
    type=str,
    help="The total roi to be processed. Format is [start:end, start:end, ... ] in voxels. Defaults to the roi of the input dataset. Do not use spaces in CLI argument.",
)
@click.option(
    "-rr",
    "--read_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be read for each block, in the format of [z,y,x] in voxels.",
)
@click.option(
    "-wr",
    "--write_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be written for each block, in the format of [z,y,x] in voxels.",
)
@click.option(
    "-nw", "--num_workers", type=int, default=16, help="The number of workers to use."
)
@click.option(
    "-mr", "--max_retries", type=int, default=2, help="The maximum number of retries."
)
@click.option("-t", "--timeout", type=int, default=None, help="The timeout in seconds.")
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=True,
    help="Whether to overwrite existing output files.",
)
@click.option(
    "-co",
    "-channels_out",
    type=int,
    default=None,
    help="The number of output channels.",
)
@click.option(
    "-dt", "--output_dtype", type=str, default="uint8", help="The output data type."
)
@click.pass_context
def run_blockwise(
    ctx,
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
    worker_file: str | Path,
    total_roi: str | None,
    read_roi_size: str,
    write_roi_size: str,
    num_workers: int = 16,
    max_retries: int = 2,
    timeout: int | None = None,
    overwrite: bool = True,
    channels_out: Optional[int] = None,
    output_dtype: np.dtype | str = "uint8",
    *args,
    **kwargs,
):
    """
    Run blockwise processing on a dataset.

    Args:
        input_container: The path to the input container.
        input_dataset: The name of the input dataset.
        output_container: The path to the output container.
        output_dataset: The name of the output dataset.
        worker_file: The path to the worker file.
        total_roi: The total roi to be processed. Format is [start:end, start:end, ... ] in voxels. Defaults to the roi of the input dataset. Do not use spaces in CLI argument.
        read_roi_size: The size of the roi to be read for each block, in the format of [z,y,x] in voxels.
        write_roi_size: The size of the roi to be written for each block, in the format of [z,y,x] in voxels.
        num_workers: The number of workers to use.
        max_retries: The maximum number of retries.
        timeout: The timeout in seconds.
        overwrite: Whether to overwrite existing output files.
        channels_out: The number of output channels.
        output_dtype: The output data type.
    Raises:
        ValueError: If the run_name is not valid.
    Examples:
        To run a blockwise operation, run:
        ```
        dacapo run-blockwise --input-container /path/to/input --input-dataset my_dataset --output-container /path/to/output --output-dataset my_output --worker-file /path/to/worker.py --total-roi [0:100,0:100,0:100] --read-roi-size [10,10,10] --write-roi-size [10,10,10] --num-workers 16
        ```
    """
    # get arbitrary args and kwargs
    parameters = unpack_ctx(ctx)

    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = open_from_identifier(input_array_identifier)

    _total_roi, read_roi, write_roi, _ = get_rois(
        total_roi, read_roi_size, write_roi_size, input_array
    )

    # prepare output dataset
    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )

    create_from_identifier(
        output_array_identifier,
        input_array.axis_names,
        _total_roi,
        channels_out,
        input_array.voxel_size,
        output_dtype,
        overwrite=overwrite,
        write_size=write_roi.shape,
    )

    _run_blockwise(  # type: ignore
        input_array_identifier=input_array_identifier,
        output_array_identifier=output_array_identifier,
        worker_file=worker_file,
        total_roi=_total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        num_workers=num_workers,
        max_retries=max_retries,
        timeout=timeout,
        parameters=parameters,
        *args,
        **kwargs,
    )


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    "-ic",
    "--input_container",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The path to the input container.",
)
@click.option(
    "-id",
    "--input_dataset",
    required=True,
    type=str,
    help="The name of the input dataset.",
)
@click.option(
    "-oc",
    "--output_container",
    required=True,
    type=click.Path(file_okay=False),
    help="The path to the output container.",
)
@click.option(
    "-od",
    "--output_dataset",
    required=True,
    type=str,
    help="The name of the output dataset.",
)
@click.option(
    "-sf",
    "--segment_function_file",
    required=True,
    type=click.Path(),
    help="The path to the segment function file.",
)
@click.option(
    "-tr",
    "--total_roi",
    type=str,
    help="The total roi to be processed. Format is [start:end,start:end,...] in voxels. Defaults to the roi of the input dataset. Do not use spaces in CLI argument.",
    default=None,
)
@click.option(
    "-rr",
    "--read_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be read for each block, in the format of [z,y,x] in voxels.",
)
@click.option(
    "-wr",
    "--write_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be written for each block, in the format of [z,y,x] in voxels.",
)
@click.option(
    "-c",
    "--context",
    type=str,
    help="The context to be used, in the format of [z,y,x] in voxels. Defaults to the difference between the read and write rois.",
    default=None,
)
@click.option(
    "-nw", "--num_workers", type=int, default=16, help="The number of workers to use."
)
@click.option(
    "-mr", "--max_retries", type=int, default=2, help="The maximum number of retries."
)
@click.option("-t", "--timeout", type=int, default=None, help="The timeout in seconds.")
@click.option(
    "-ow",
    "--overwrite",
    is_flag=True,
    default=True,
    help="Whether to overwrite existing output files.",
)
@click.option(
    "-co",
    "--channels_out",
    type=int,
    default=None,
    help="The number of output channels.",
)
@click.pass_context
def segment_blockwise(
    ctx,
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
    segment_function_file: Path | str,
    total_roi: str,
    read_roi_size: str,
    write_roi_size: str,
    context: str | None,
    num_workers: int = 16,
    max_retries: int = 2,
    timeout=None,
    overwrite: bool = True,
    channels_out: Optional[int] = None,
    *args,
    **kwargs,
):
    """
    Segment the input dataset blockwise using a segment function file.

    Args:
        input_container (str): The path to the input container.
        input_dataset (str): The name of the input dataset.
        output_container (str): The path to the output container.
        output_dataset (str): The name of the output dataset.
        segment_function_file (str): The path to the segment function file.
        total_roi (str): The total roi to be processed. Format is [start:end,start:end,...] in voxels. Defaults to the roi of the input dataset. Do not use spaces in CLI argument.
        read_roi_size (str): The size of the roi to be read for each block, in the format of [z,y,x] in voxels.
        write_roi_size (str): The size of the roi to be written for each block, in the format of [z,y,x] in voxels.
        context (str, optional): The context to be used, in the format of [z,y,x] in voxels. Defaults to the difference between the read and write rois.
        num_workers (int, optional): The number of workers to use. Defaults to 16.
        max_retries (int, optional): The maximum number of retries. Defaults to 2.
        timeout (int, optional): The timeout in seconds. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to True.
        channels_out (int, optional): The number of output channels. Defaults to None.
    Raises:
        ValueError: If the run_name is not valid.
    Examples:
        To segment blockwise, run:
        ```
        dacapo segment-blockwise --input-container /path/to/input --input-dataset my_dataset --output-container /path/to/output --output-dataset my_output --segment-function-file /path/to/segment_function.py --total-roi [0:100,0:100,0:100] --read-roi-size [10,10,10] --write-roi-size [10,10,10] --num-workers 16
        ```
    """
    # get arbitrary args and kwargs
    parameters = unpack_ctx(ctx)

    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = open_from_identifier(input_array_identifier)

    _total_roi, read_roi, write_roi, _context = get_rois(
        total_roi, read_roi_size, write_roi_size, input_array
    )

    if context is not None:
        _context = Coordinate(
            [int(s) for s in context.rstrip("]").lstrip("[").split(",")]
        )

    # create zarr array for output
    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )

    create_from_identifier(
        output_array_identifier,
        input_array.axis_names,
        _total_roi,
        channels_out,
        input_array.voxel_size,
        np.uint64,
        overwrite=overwrite,
        write_size=write_roi.shape,
    )
    print(
        f"Created output array {output_array_identifier.container}:{output_array_identifier.dataset} with ROI {_total_roi}."
    )

    _segment_blockwise(  # type: ignore
        input_array_identifier=input_array_identifier,
        output_array_identifier=output_array_identifier,
        segment_function_file=segment_function_file,
        context=_context,
        total_roi=_total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        num_workers=num_workers,
        max_retries=max_retries,
        timeout=timeout,
        parameters=parameters,
        *args,
        **kwargs,
    )


def prompt_with_choices(prompt_text, choices, default_index=0):
    """
    Prompts the user with a list of choices and returns the selected choice.

    Args:
        prompt_text (str): The prompt text to display to the user.
        choices (list): The list of choices to present.
        default_index (int): The index of the default choice (0-based).

    Returns:
        str: The selected choice.
    """
    while True:
        click.echo(prompt_text)
        for i, choice in enumerate(choices, 1):
            click.echo(f"{i} - {choice}")

        # If the default_index is out of range, set to 0
        default_index = max(0, min(default_index, len(choices) - 1))

        try:
            # Prompt the user for input
            choice_num = click.prompt(
                f"Enter your choice (default: {choices[default_index]})",
                default=default_index + 1,
                type=int,
            )

            # Check if the provided number is valid
            if 1 <= choice_num <= len(choices):
                return choices[choice_num - 1]
            else:
                click.echo("Invalid choice number. Please try again.")
        except click.BadParameter:
            click.echo("Invalid input. Please enter a number.")


@cli.command()
def config():
    if os.path.exists("dacapo.yaml"):
        overwrite = click.confirm(
            "dacapo.yaml already exists. Do you want to overwrite it?", default=False
        )
        if not overwrite:
            click.echo("Aborting configuration creation.")
            return
    runs_base_dir = click.prompt("Enter the base directory for runs", type=str)
    storage_type = prompt_with_choices("Enter the type of storage:", ["files", "mongo"])
    mongo_db_name = None
    mongo_db_host = None
    if storage_type == "mongo":
        mongo_db_name = click.prompt("Enter the name of the MongoDB database", type=str)
        mongo_db_host = click.prompt("Enter the MongoDB host URI", type=str)

    compute_type = prompt_with_choices(
        "Enter the type of compute context:", ["LocalTorch", "Bsub"]
    )
    if compute_type == "Bsub":
        queue = click.prompt("Enter the queue for compute context", type=str)
        num_gpus = click.prompt("Enter the number of GPUs", type=int)
        num_cpus = click.prompt("Enter the number of CPUs", type=int)
        billing = click.prompt("Enter the billing account", type=str)
        compute_context = {
            "type": compute_type,
            "config": {
                "queue": queue,
                "num_gpus": num_gpus,
                "num_cpus": num_cpus,
                "billing": billing,
            },
        }
    else:
        compute_context = {"type": compute_type}

    try:
        generate_config(
            runs_base_dir,
            storage_type,
            compute_type,
            compute_context,
            mongo_db_name,
            mongo_db_host,
        )
    except ValueError as e:
        logger.error(str(e))


def generate_dacapo_yaml(config):
    with open("dacapo.yaml", "w") as f:
        yaml.dump(config.serialize(), f, default_flow_style=False)
    print("dacapo.yaml has been created.")


def generate_config(
    runs_base_dir,
    storage_type,
    compute_type,
    compute_context,
    mongo_db_name=None,
    mongo_db_host=None,
):
    config = DaCapoConfig(
        type=storage_type,
        runs_base_dir=Path(runs_base_dir).expanduser(),
        compute_context=compute_context,
    )

    if storage_type == "mongo":
        if not mongo_db_name or not mongo_db_host:
            raise ValueError(
                "--mongo_db_name and --mongo_db_host are required when type is 'mongo'"
            )
        config.mongo_db_name = mongo_db_name
        config.mongo_db_host = mongo_db_host

    generate_dacapo_yaml(config)


def unpack_ctx(ctx):
    """
    Unpacks the context object and returns a dictionary of keyword arguments.

    Args:
        ctx (object): The context object containing the arguments.
    Returns:
        dict: A dictionary of keyword arguments.
    Raises:
        ValueError: If the run_name is not valid.
    Example:
        >>> ctx = ...
        >>> kwargs = unpack_ctx(ctx)
        >>> print(kwargs)
        {'arg1': value1, 'arg2': value2, ...}
    """
    kwargs = {
        ctx.args[i].lstrip("-"): ctx.args[i + 1] for i in range(0, len(ctx.args), 2)
    }
    for k, v in kwargs.items():
        if v.isnumeric():
            kwargs[k] = int(v)
        elif v.replace(".", "").isnumeric():
            kwargs[k] = float(v)
        print(f"{k}: {kwargs[k]}")
    return kwargs


def get_rois(total_roi, read_roi_size, write_roi_size, input_array: Array):
    """
    Get the ROIs for processing.

    Args:
        total_roi (str): The total ROI to be processed.
        read_roi_size (str): The size of the ROI to be read for each block.
        write_roi_size (str): The size of the ROI to be written for each block.
        input_array: The input array.
    Returns:
        tuple: A tuple containing the total ROI, read ROI, write ROI, and context.
    Raises:
        ValueError: If the run_name is not valid.
    Example:
        >>> total_roi, read_roi, write_roi, context = get_rois(total_roi, read_roi_size, write_roi_size, input_array)
    """
    if total_roi is not None:
        # parse the string into a Roi
        start, end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in total_roi.strip("[]").split(",")
            ]
        )
        _total_roi = (
            Roi(
                Coordinate(start),
                Coordinate(end) - Coordinate(start),
            )
            * input_array.voxel_size
        )
    else:
        _total_roi = input_array.roi

    _read_roi_size = [int(s) for s in read_roi_size.rstrip("]").lstrip("[").split(",")]
    _write_roi_size = [
        int(s) for s in write_roi_size.rstrip("]").lstrip("[").split(",")
    ]
    read_roi = Roi([0, 0, 0], _read_roi_size) * input_array.voxel_size
    # Find different between read and write roi
    context = Coordinate((np.array(_read_roi_size) - np.array(_write_roi_size)) // 2)
    write_roi = (
        Roi(
            context,
            _write_roi_size,
        )
        * input_array.voxel_size
    )
    context = context * input_array.voxel_size

    return _total_roi, read_roi, write_roi, context
