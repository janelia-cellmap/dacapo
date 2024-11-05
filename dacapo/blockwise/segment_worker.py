from importlib.machinery import SourceFileLoader
import logging
import os
from upath import UPath as Path
import sys
import click
import daisy
from funlib.persistence import Array

import numpy as np
import yaml
from dacapo.compute_context import create_compute_context
from dacapo.tmp import open_from_identifier


from dacapo.store.array_store import LocalArrayIdentifier

logger = logging.getLogger(__name__)


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
    CLI for running the segment worker.

    Args:
        log_level (str): The log level to use.
    """
    logging.basicConfig(level=getattr(logging, log_level.upper()))


fit = "shrink"
read_write_conflict = True
path = __file__


@cli.command()
@click.option("--input_container", type=str, help="Input container")
@click.option("--input_dataset", type=str, help="Input dataset")
@click.option("--output_container", type=str, help="Output container")
@click.option("--output_dataset", type=str, help="Output dataset")
@click.option("--tmpdir", type=str, help="Temporary directory")
@click.option("--function_path", type=str, help="Path to the segment function")
def start_worker(
    input_container: str | Path,
    input_dataset: str,
    output_container: str | Path,
    output_dataset: str,
    tmpdir: str | Path,
    function_path: str | Path,
    return_io_loop: bool = False,
):
    return start_worker_fn(
        input_container=input_container,
        input_dataset=input_dataset,
        output_container=output_container,
        output_dataset=output_dataset,
        tmpdir=tmpdir,
        function_path=function_path,
        return_io_loop=return_io_loop,
    )


def start_worker_fn(
    input_container: str | Path,
    input_dataset: str,
    output_container: str | Path,
    output_dataset: str,
    tmpdir: str | Path,
    function_path: str | Path,
    return_io_loop: bool = False,
):
    """
    Start a worker to run a segment function on a given dataset.

    Args:
        input_container (str): The input container.
        input_dataset (str): The input dataset.
        output_container (str): The output container.
        output_dataset (str): The output dataset.
        tmpdir (str): The temporary directory.
        function_path (str): The path to the segment function.
        return_io_loop (bool): Whether to return the io loop or run it.
    """

    print("Starting worker")
    # get arrays
    input_array_identifier = LocalArrayIdentifier(Path(input_container), input_dataset)
    print(f"Opening input array {input_array_identifier}")
    input_array = open_from_identifier(input_array_identifier)

    output_array_identifier = LocalArrayIdentifier(
        Path(output_container), output_dataset
    )
    print(f"Opening output array {output_array_identifier}")
    output_array = open_from_identifier(output_array_identifier)

    # Load segment function
    function_name = Path(function_path).stem
    print(f"Loading segment function from {str(function_path)}")
    function = SourceFileLoader(function_name, str(function_path)).load_module()
    segment_function = function.segment_function

    # load default parameters
    if hasattr(function, "default_parameters"):
        parameters = function.default_parameters
    else:
        parameters = {}

    # load parameters saved in tmpdir
    if os.path.exists(os.path.join(tmpdir, "parameters.yaml")):
        print(f"Loading parameters from {os.path.join(tmpdir, 'parameters.yaml')}")
        with open(os.path.join(tmpdir, "parameters.yaml"), "r") as f:
            parameters.update(yaml.safe_load(f))

    # wait for blocks to run pipeline
    def io_loop():
        client = daisy.Client()
        num_voxels_in_block = None

        while True:
            with client.acquire_block() as block:
                if block is None:
                    break
                if num_voxels_in_block is None:
                    num_voxels_in_block = np.prod(block.write_roi.size)

                segmentation = segment_function(input_array, block, **parameters)

                assert (
                    segmentation.dtype == np.uint64
                ), "Instance segmentations returned by segment_function is expected to be uint64"

                id_bump = block.block_id[1] * num_voxels_in_block
                segmentation += id_bump
                segmentation[segmentation == id_bump] = 0

                # wrap segmentation into daisy array
                segmentation = Array(
                    segmentation, roi=block.read_roi, voxel_size=input_array.voxel_size
                )

                # store segmentation in out array
                output_array[block.write_roi] = segmentation[block.write_roi]

                neighbor_roi = block.write_roi.grow(
                    input_array.voxel_size, input_array.voxel_size
                )

                # clip segmentation to 1-voxel context
                segmentation = segmentation.to_ndarray(roi=neighbor_roi, fill_value=0)
                neighbors = output_array._daisy_array.to_ndarray(
                    roi=neighbor_roi, fill_value=0
                )

                unique_pairs = []

                for d in range(3):
                    slices_neg = tuple(
                        slice(None) if dd != d else slice(0, 1) for dd in range(3)
                    )
                    slices_pos = tuple(
                        slice(None) if dd != d else slice(-1, None) for dd in range(3)
                    )

                    pairs_neg = np.array(
                        [
                            segmentation[slices_neg].flatten(),
                            neighbors[slices_neg].flatten(),
                        ]
                    )
                    pairs_neg = pairs_neg.transpose()

                    pairs_pos = np.array(
                        [
                            segmentation[slices_pos].flatten(),
                            neighbors[slices_pos].flatten(),
                        ]
                    )
                    pairs_pos = pairs_pos.transpose()

                    unique_pairs.append(
                        np.unique(np.concatenate([pairs_neg, pairs_pos]), axis=0)
                    )

                unique_pairs = np.concatenate(unique_pairs)
                zero_u = unique_pairs[:, 0] == 0  # type: ignore
                zero_v = unique_pairs[:, 1] == 0  # type: ignore
                non_zero_filter = np.logical_not(np.logical_or(zero_u, zero_v))

                edges = unique_pairs[non_zero_filter]
                nodes = np.unique(edges)

                assert os.path.exists(tmpdir)
                path = os.path.join(tmpdir, f"block_{block.block_id[1]}.npz")
                print(f"Writing ids to {path}")
                with open(path, "wb") as f:
                    np.savez_compressed(
                        f,
                        nodes=nodes,
                        edges=edges,
                    )

    if return_io_loop:
        return io_loop
    else:
        io_loop()


def spawn_worker(
    input_array_identifier: LocalArrayIdentifier,
    output_array_identifier: LocalArrayIdentifier,
    tmpdir: str,
    function_path: str,
):
    """
    Spawn a worker to predict on a given dataset.

    Args:
        model (Model): The model to use for prediction.
        raw_array (Array): The raw data to predict on.
        prediction_array_identifier (LocalArrayIdentifier): The identifier of the prediction array.
    Returns:
        Callable: The function to run the worker.
    """
    compute_context = create_compute_context()
    if not compute_context.distribute_workers:
        return start_worker_fn(
            input_array_identifier.container,
            input_array_identifier.dataset,
            output_array_identifier.container,
            output_array_identifier.dataset,
            tmpdir,
            function_path,
            return_io_loop=True,
        )

    # Make the command for the worker to run
    command = [
        # "python",
        sys.executable,
        path,
        "start-worker",
        "--input_container",
        input_array_identifier.container,
        "--input_dataset",
        input_array_identifier.dataset,
        "--output_container",
        output_array_identifier.container,
        "--output_dataset",
        output_array_identifier.dataset,
        "--tmpdir",
        tmpdir,
        "--function_path",
        function_path,
    ]

    def run_worker():
        """
        Run the worker in the given compute context.
        """
        compute_context.execute(command)

    return run_worker


if __name__ == "__main__":
    cli()
