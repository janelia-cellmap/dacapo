from pathlib import Path
import tempfile
import time
import daisy
import dacapo.blockwise
from funlib.geometry import Roi, Coordinate
import yaml

from dacapo.blockwise import DaCapoBlockwiseTask
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
def run_blockwise(
    worker_file: str | Path,
    total_roi: Roi,
    read_roi: Roi,
    write_roi: Roi,
    num_workers: int = 4,
    max_retries: int = 2,
    timeout=None,
    upstream_tasks=None,
    *args,
    **kwargs,
):
    """Run a function in parallel over a large volume.

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

    """
    logger.warning("Running blockwise with worker_file: %s", worker_file)
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

    logger.warning("Running blockwise task: %s", task)

    return daisy.run_blockwise([task])


def segment_blockwise(
    segment_function_file: str | Path,
    context: Coordinate,
    total_roi: Roi,
    read_roi: Roi,
    write_roi: Roi,
    num_workers: int = 4,
    max_retries: int = 2,
    timeout=None,
    upstream_tasks=None,
    tmp_prefix="tmp",
    *args,
    **kwargs,
):
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
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

        return success and daisy.run_blockwise([task])
