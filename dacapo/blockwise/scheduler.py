from upath import UPath as Path
import shutil
import tempfile
import time
import daisy
import dacapo.blockwise
from funlib.geometry import Roi, Coordinate
import yaml

from dacapo.blockwise import DaCapoBlockwiseTask
from dacapo import Options
import logging

from dacapo.compute_context import create_compute_context

logger = logging.getLogger(__name__)


def run_blockwise(
    worker_file: str | Path,
    total_roi: Roi,
    read_roi: Roi,
    write_roi: Roi,
    num_workers: int = 16,
    max_retries: int = 1,
    timeout=None,
    upstream_tasks=None,
    *args,
    **kwargs,
):
    """
    Run a function in parallel over a large volume.

    Args:
        worker_file (``str`` or ``Path``):
            The path to the file containing the necessary worker functions:
            ``spawn_worker`` and ``start_worker``.
            Optionally, the file can also contain a ``check_function`` and an ``init_callback_fn``.
        total_roi (``Roi``):
            The ROI to process.
        read_roi (``Roi``):
            The ROI to read from for a block.
        write_roi (``Roi``):
            The ROI to write to for a block.
        num_workers (``int``):
            The number of workers to use.
        max_retries (``int``):
            The maximum number of times a task will be retried if failed
            (either due to failed post check or application crashes or network
            failure)
        *args:
            Additional positional arguments to pass to ``worker_function``.
        **kwargs:
            Additional keyword arguments to pass to ``worker_function``.
    Returns:
        ``Bool``.
    Examples:
        >>> run_blockwise(worker_file, total_roi, read_roi, write_roi, num_workers, max_retries, timeout, upstream_tasks)

    """

    # Make the task
    task = DaCapoBlockwiseTask(
        worker_file,
        total_roi,
        read_roi,
        write_roi,
        num_workers,
        max_retries,
        timeout,
        upstream_tasks,
        *args,
        **kwargs,
    )
    print("Running blockwise with worker_file: ", worker_file)
    print(f"Using compute context: {create_compute_context()}")
    compute_context = create_compute_context()
    print(f"Using compute context: {compute_context}")

    multiprocessing = compute_context.distribute_workers

    success = daisy.run_blockwise([task], multiprocessing=multiprocessing)
    return success


def segment_blockwise(
    segment_function_file: str | Path,
    context: Coordinate,
    total_roi: Roi,
    read_roi: Roi,
    write_roi: Roi,
    num_workers: int = 16,
    max_retries: int = 2,
    timeout=None,
    upstream_tasks=None,
    keep_tmpdir=False,
    *args,
    **kwargs,
):
    """
    Run a segmentation function in parallel over a large volume.

    Args:
        segment_function_file (``str`` or ``Path``):
            The path to the file containing the necessary worker functions:
            ``spawn_worker`` and ``start_worker``.
            Optionally, the file can also contain a ``check_function`` and an ``init_callback_fn``.
        context (``Coordinate``):
            The context to add to the read and write ROI.
        total_roi (``Roi``):
            The ROI to process.
        read_roi (``Roi``):
            The ROI to read from for a block.
        write_roi (``Roi``):
            The ROI to write to for a block.
        num_workers (``int``):
            The number of workers to use.
        max_retries (``int``):
            The maximum number of times a task will be retried if failed
            (either due to failed post check or application crashes or network
            failure)
        timeout (``int``):
            The maximum time in seconds to wait for a worker to complete a task.
        upstream_tasks (``List``):
            List of upstream tasks.
        keep_tmpdir (``bool``):
            Whether to keep the temporary directory.
        *args:
            Additional positional arguments to pass to ``worker_function``.
        **kwargs:
            Additional keyword arguments to pass to ``worker_function``.
    Returns:
            ``Bool``.
    Examples:
        >>> segment_blockwise(segment_function_file, context, total_roi, read_roi, write_roi, num_workers, max_retries, timeout, upstream_tasks)
    """
    options = Options.instance()
    if not options.runs_base_dir.exists():
        options.runs_base_dir.mkdir(parents=True)
    tmpdir = tempfile.mkdtemp(dir=options.runs_base_dir)

    print(
        "Running blockwise segmentation, with segment_function_file: ",
        segment_function_file,
        " in temp directory: ",
        tmpdir,
    )
    print(f"Using compute context: {create_compute_context()}")
    # write parameters to tmpdir
    if "parameters" in kwargs:
        with open(Path(tmpdir, "parameters.yaml"), "w") as f:
            yaml.dump(kwargs.pop("parameters"), f)

    # Make the task
    task = DaCapoBlockwiseTask(
        str(Path(Path(dacapo.blockwise.__file__).parent, "segment_worker.py")),
        total_roi.grow(context, context),
        read_roi,
        write_roi,
        num_workers,
        max_retries,
        timeout,
        upstream_tasks,
        tmpdir=tmpdir,
        function_path=str(segment_function_file),
        *args,
        **kwargs,
    )
    print(
        "Running blockwise segmentation with worker_file: ",
        str(Path(Path(dacapo.blockwise.__file__).parent, "segment_worker.py")),
    )
    success = daisy.run_blockwise([task])

    # give a second for the fist task to finish
    time.sleep(1)
    read_roi = write_roi

    # Make the task
    task = DaCapoBlockwiseTask(
        str(Path(Path(dacapo.blockwise.__file__).parent, "relabel_worker.py")),
        total_roi,
        read_roi,
        write_roi,
        num_workers,
        max_retries,
        timeout,
        upstream_tasks,
        tmpdir=tmpdir,
        *args,
        **kwargs,
    )
    print(
        "Running blockwise relabeling with worker_file: ",
        str(Path(Path(dacapo.blockwise.__file__).parent, "relabel_worker.py")),
    )

    success = success and daisy.run_blockwise([task])

    if success and not keep_tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # Write a relabel script to tmpdir
        output_container = kwargs["output_array_identifier"].container
        output_dataset = kwargs["output_array_identifier"].dataset
        out_string = "from dacapo.blockwise import DaCapoBlockwiseTask\n"
        out_string += (
            "from dacapo.store.local_array_store import LocalArrayIdentifier\n"
        )
        out_string += "import daisy\n"
        out_string += "from funlib.geometry import Roi, Coordinate\n"
        out_string += "from upath import UPath as Path\n"
        out_string += f"output_array_identifier = LocalArrayIdentifier(Path({output_container}), {output_dataset})\n"
        out_string += (
            f"total_roi = Roi({total_roi.get_begin()}, {total_roi.get_shape()})\n"
        )
        out_string += (
            f"read_roi = Roi({read_roi.get_begin()}, {read_roi.get_shape()})\n"
        )
        out_string += (
            f"write_roi = Roi({write_roi.get_begin()}, {write_roi.get_shape()})\n"
        )
        out_string += "task = DaCapoBlockwiseTask(\n"
        out_string += f'    "{str(Path(Path(dacapo.blockwise.__file__).parent, "relabel_worker.py"))}"),\n'
        out_string += "    total_roi,\n"
        out_string += "    read_roi,\n"
        out_string += "    write_roi,\n"
        out_string += f"    {num_workers},\n"
        out_string += f"    {max_retries},\n"
        out_string += f"    {timeout},\n"
        out_string += f"    tmpdir={tmpdir},\n"
        out_string += f"    output_array_identifier=output_array_identifier,\n"
        out_string += ")\n"
        out_string += "success = daisy.run_blockwise([task])\n"
        out_string += "if success:\n"
        out_string += f"    shutil.rmtree({tmpdir}, ignore_errors=True)\n"
        out_string += "else:\n"
        out_string += '    print("Relabeling failed")\n'
        with open(Path(tmpdir, "relabel.py"), "w") as f:
            f.write(out_string)
        raise RuntimeError(
            f"Blockwise segmentation failed. Can rerun with merge files stored at:\n\t{tmpdir}"
            f"Use read_roi: {read_roi} and write_roi: {write_roi} to rerun."
            f"Or simply run the script at {Path(tmpdir, 'relabel.py')}"
        )
    return success
