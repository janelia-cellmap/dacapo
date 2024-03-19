from pathlib import Path
from typing import Optional

import numpy as np

import dacapo
import click
import logging
from funlib.geometry import Roi, Coordinate
from funlib.persistence import open_ds
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
)
@click.option("-id", "--input_dataset", required=True, type=str)
@click.option(
    "-oc", "--output_container", required=True, type=click.Path(file_okay=False)
)
@click.option("-od", "--output_dataset", required=True, type=str)
@click.option(
    "-w", "--worker_file", required=True, type=str, help="The path to the worker file."
)
@click.option(
    "-tr",
    "--total_roi",
    required=True,
    type=str,
    help="The total roi to be processed. Format is [start:end, start:end, ... ] in voxels. Defaults to the roi of the input dataset. Do not use spaces in CLI argument.",
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
@click.option("-nw", "--num_workers", type=int, default=16)
@click.option("-mr", "--max_retries", type=int, default=2)
@click.option("-t", "--timeout", type=int, default=None)
@click.option("-ow", "--overwrite", is_flag=True, default=True)
@click.option("-co", "-channels_out", type=int, default=None)
@click.option("-dt", "--output_dtype", type=str, default="uint8")
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
)
@click.option("-id", "--input_dataset", required=True, type=str)
@click.option(
    "-oc", "--output_container", required=True, type=click.Path(file_okay=False)
)
@click.option("-od", "--output_dataset", required=True, type=str)
@click.option("-sf", "--segment_function_file", required=True, type=click.Path())
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
@click.option("-nw", "--num_workers", type=int, default=16)
@click.option("-mr", "--max_retries", type=int, default=2)
@click.option("-t", "--timeout", type=int, default=None)
@click.option("-ow", "--overwrite", is_flag=True, default=True)
@click.option("-co", "--channels_out", type=int, default=None)
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


def unpack_ctx(ctx):
    # print(ctx.args)
    kwargs = {
        ctx.args[i].lstrip("-"): ctx.args[i + 1] for i in range(0, len(ctx.args), 2)
    }
    for k, v in kwargs.items():
        if v.isnumeric():
            kwargs[k] = int(v)
        elif v.replace(".", "").isnumeric():
            kwargs[k] = float(v)
        print(f"{k}: {kwargs[k]}")
        # print(f"{type(k)}: {k} --> {type(kwargs[k])} {kwargs[k]}")
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
