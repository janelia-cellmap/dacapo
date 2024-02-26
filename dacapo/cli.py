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
    help="The total roi to be processed. Format is [start:end, start:end, ... ]",
    default=None,
)
@click.option(
    "-rr",
    "--read_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be read for each block.",
)
@click.option(
    "-wr",
    "--write_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be written for each block.",
)
@click.option("-nw", "--num_workers", type=int, default=16)
@click.option("-mr", "--max_retries", type=int, default=2)
@click.option("-t", "--timeout", type=int, default=None)
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
    timeout=None,
    *args,
    **kwargs,
):
    # get arubtrary args and kwargs
    kwargs = unpack_ctx(ctx)

    if total_roi is not None:
        # parse the string into a Roi
        start, end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in total_roi.strip("[]").split(",")
            ]
        )
        _total_roi = Roi(
            Coordinate(start),
            Coordinate(end) - Coordinate(start),
        )
    else:
        input_ds = open_ds(str(input_container), input_dataset)
        _total_roi = input_ds.roi

    read_roi = Roi([0, 0, 0], [int(coord) for coord in read_roi_size.split(",")])
    # Find different between read and write roi
    context = (np.array(write_roi_size) - np.array(read_roi_size)) // 2
    write_roi = Roi(
        context,
        [int(coord) for coord in write_roi_size.split(",")],
    )

    _run_blockwise(
        input_container=input_container,
        input_dataset=input_dataset,
        output_container=output_container,
        output_dataset=output_dataset,
        worker_file=worker_file,
        total_roi=_total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        num_workers=num_workers,
        max_retries=max_retries,
        timeout=timeout,
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
    "-c",
    "--context",
    type=str,
    help="The context to be used, in the format of [x,y,z]. Defaults to the difference between the read and write rois.",
    default=None,
)
@click.option(
    "-tr",
    "--total_roi",
    type=str,
    help="The total roi to be processed. Format is [start:end,start:end,...] Defaults to the roi of the input dataset. Do not use spaces in CLI argument.",
    default=None,
)
@click.option(
    "-rr",
    "--read_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be read for each block, in the format of [x,y,z].",
)
@click.option(
    "-wr",
    "--write_roi_size",
    required=True,
    type=str,
    help="The size of the roi to be written for each block, in the format of [x,y,z].",
)
@click.option("-nw", "--num_workers", type=int, default=16)
@click.option("-mr", "--max_retries", type=int, default=2)
@click.option("-t", "--timeout", type=int, default=None)
@click.option("-tp", "--tmp_prefix", type=str, default="tmp")
@click.pass_context
def segment_blockwise(
    ctx,
    input_container: Path | str,
    input_dataset: str,
    output_container: Path | str,
    output_dataset: str,
    segment_function_file: Path | str,
    context: str | None,
    total_roi: str,
    read_roi_size: str,
    write_roi_size: str,
    num_workers: int = 16,
    max_retries: int = 2,
    timeout=None,
    tmp_prefix: str = "tmp",
    *args,
    **kwargs,
):
    # get arubtrary args and kwargs
    kwargs = unpack_ctx(ctx)

    if total_roi is not None:
        # parse the string into a Roi
        start, end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in total_roi.strip("[]").split(",")
            ]
        )
        _total_roi = Roi(
            Coordinate(start),
            Coordinate(end) - Coordinate(start),
        )
    else:
        input_ds = open_ds(str(input_container), input_dataset)
        _total_roi = input_ds.roi

    read_roi = Roi([0, 0, 0], [int(coord) for coord in read_roi_size.split(",")])
    # Find different between read and write roi
    _context = (np.array(write_roi_size) - np.array(read_roi_size)) // 2
    write_roi = Roi(
        _context,
        [int(coord) for coord in write_roi_size.split(",")],
    )

    if context is None:
        context = _context

    _segment_blockwise(
        input_container=input_container,
        input_dataset=input_dataset,
        output_container=output_container,
        output_dataset=output_dataset,
        segment_function_file=segment_function_file,
        context=_context,
        total_roi=_total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        num_workers=num_workers,
        max_retries=max_retries,
        timeout=timeout,
        tmp_prefix=tmp_prefix,
        *args,
        **kwargs,
    )


def unpack_ctx(ctx):
    kwargs = {
        ctx.args[i].lstrip("-"): ctx.args[i + 1] for i in range(0, len(ctx.args), 2)
    }
    for k, v in kwargs.items():
        print(k, v)
        if v.isnumeric():
            if "." in v:
                kwargs[k] = float(v)
            else:
                kwargs[k] = int(v)
    return kwargs


@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.pass_context
def test(ctx):
    print(ctx.args)
    print(unpack_ctx(ctx))
