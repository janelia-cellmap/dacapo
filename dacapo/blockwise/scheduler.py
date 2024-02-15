from pathlib import Path
import daisy
from funlib.geometry import Roi

from dacapo.compute_context import ComputeContext
from dacapo.blockwise import DaCapoBlockwiseTask


def run_blockwise(
    worker_file: str or Path,
    compute_context: ComputeContext | str,
    total_roi: Roi,
    read_roi: Roi,
    write_roi: Roi,
    num_workers: int = 16,
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

        compute_context (``ComputeContext``):

            The compute context to use for parallelization.

        *args:

            Additional positional arguments to pass to ``worker_function``.

        **kwargs:

            Additional keyword arguments to pass to ``worker_function``.

    Returns:

            ``Bool``.

    """

    # Make the task
    task = DaCapoBlockwiseTask(
        worker_file,
        compute_context,
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

    return daisy.run_blockwise([task])
