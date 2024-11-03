from upath import UPath as Path
from typing import Optional

import numpy as np
import yaml
import dacapo
import click
import logging
from funlib.geometry import Roi, Coordinate
from dacapo.experiments.datasplits.datasets.dataset import Dataset
from dacapo.experiments.tasks.post_processors.post_processor_parameters import (
    PostProcessorParameters,
)
from dacapo.blockwise import (
    run_blockwise as _run_blockwise,
    segment_blockwise as _segment_blockwise,
)
from dacapo.store.local_array_store import LocalArrayIdentifier
from dacapo.experiments.datasplits.datasets.arrays import ZarrArray
from dacapo.options import DaCapoConfig
import os
from dacapo.utils.uri import setup_uri_scheme, add_config

import requests
import subprocess
import http.server
import socketserver
from urllib.parse import urlparse
import threading


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


logger = logging.getLogger(__name__)


import http.server
import socketserver
import threading
import webbrowser


class SingleRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(
            f"<html><body><h2>{self.server.message}</h2></body></html>".encode("utf-8")
        )


class SingleRequestServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, message):
        super().__init__(server_address, RequestHandlerClass)
        self.message = message
        self.request_count = 0

    def handle_request(self):
        
        super().handle_request()
        self.shutdown()  # Stop server after the first request


def start_server(message):
    with SingleRequestServer(
        ("localhost", 8080), SingleRequestHandler, message
    ) as httpd:
        webbrowser.open(
            "http://localhost:8080"
        )  # Open the browser once the server starts
        httpd.handle_request()


@cli.command()
@click.argument("uri")
def install_config(uri):
    
    print("Downloading configuration...")
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != "dacapo" or parsed_uri.netloc != "install_config":
        print(
            f"Invalid URI: {uri}. Expected format: dacapo://install_config/CONFIG_URL"
        )
        print(f"Scheme: {parsed_uri.scheme}")
        print(f"Path: {parsed_uri.path}")
        print(f"Netloc: {parsed_uri.netloc}")
        print(f"Params: {parsed_uri.params}")
        print(f"Query: {parsed_uri.query}")
        start_server("Invalid URI. Expected format: dacapo://install_config/CONFIG_URL")
        return

    config_url = parsed_uri.path.split("/install_config/")[-1]
    try:
        response = requests.get(config_url)
        response.raise_for_status()
        with open("temp_config.yaml", "wb") as file:
            file.write(response.content)
        result = add_config("temp_config.yaml")
        message = "Configuration added successfully."
    except requests.exceptions.RequestException:
        message = "Failed to download config."
    except subprocess.CalledProcessError as e:
        if "already exists" in str(e):
            message = "Error: Configuration already exists!"
        else:
            message = "An error occurred while adding the configuration."

    # Start server to display the message
    thread = threading.Thread(target=start_server, args=(message,))
    thread.start()
    print("Navigate to http://localhost:8084 to see the status.")
    import sys

    sys.exit(0)


@cli.command()
def setup_uri():
    
    setup_uri_scheme()


@cli.command()
@click.option(
    "-r", "--run-name", required=True, type=str, help="The NAME of the run to train."
)
@click.option(
    "--no-validation", is_flag=True, help="Disable validation after training."
)
def train(run_name, no_validation):
    
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
    
    # get arbitrary args and kwargs
    parameters = unpack_ctx(ctx)

    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = ZarrArray.open_from_array_identifier(input_array_identifier)

    _total_roi, read_roi, write_roi, _ = get_rois(
        total_roi, read_roi_size, write_roi_size, input_array
    )

    # prepare output dataset
    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )

    ZarrArray.create_from_array_identifier(
        output_array_identifier,
        input_array.axes,
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
    
    # get arbitrary args and kwargs
    parameters = unpack_ctx(ctx)

    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    input_array = ZarrArray.open_from_array_identifier(input_array_identifier)

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

    ZarrArray.create_from_array_identifier(
        output_array_identifier,
        input_array.axes,
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


def get_rois(total_roi, read_roi_size, write_roi_size, input_array):
    
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
