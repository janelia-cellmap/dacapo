from datetime import datetime
from importlib.machinery import SourceFileLoader
from upath import UPath as Path
from daisy import Task, Roi


class DaCapoBlockwiseTask(Task):
    """
    A task to run a blockwise worker function. This task is used to run a
    blockwise worker function on a given ROI.

    Attributes:
        worker_file (str | Path): The path to the worker file.
        total_roi (Roi): The ROI to process.
        read_roi (Roi): The ROI to read from for a block.
        write_roi (Roi): The ROI to write to for a block.
        num_workers (int): The number of workers to use.
        max_retries (int): The maximum number of times a task will be retried if failed
            (either due to failed post check or application crashes or network
            failure)
        timeout: The timeout for the task.
        upstream_tasks: The upstream tasks.
        *args: Additional positional arguments to pass to ``worker_function``.
        **kwargs: Additional keyword arguments to pass to ``worker_function``.
    Methods:
        __init__:
            Initialize the task.
    """

    def __init__(
        self,
        worker_file: str | Path,
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
        """
        Initialize the task.

        Args:
            worker_file (str | Path): The path to the worker file.
            total_roi (Roi): The ROI to process.
            read_roi (Roi): The ROI to read from for a block.
            write_roi (Roi): The ROI to write to for a block.
            num_workers (int): The number of workers to use.
            max_retries (int): The maximum number of times a task will be retried if failed
                (either due to failed post check or application crashes or network
                failure)
            timeout: The timeout for the task.
            upstream_tasks: The upstream tasks.
            *args: Additional positional arguments to pass to ``worker_function``.
            **kwargs: Additional keyword arguments to pass to ``worker_function``.
        """
        # Load worker functions
        worker_name = Path(worker_file).stem
        worker = SourceFileLoader(worker_name, str(worker_file)).load_module()

        # Make the task_id unique
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_id = worker_name + timestamp

        process_function = worker.spawn_worker(*args, **kwargs)
        if hasattr(worker, "check_function"):
            check_function = worker.check_function
        else:
            check_function = None
        if hasattr(worker, "init_callback_fn"):
            init_callback_fn = worker.init_callback_fn
        else:
            init_callback_fn = None
        read_write_conflict = worker.read_write_conflict
        fit = worker.fit

        kwargs = {
            "task_id": task_id,
            "total_roi": total_roi,
            "read_roi": read_roi,
            "write_roi": write_roi,
            "process_function": process_function,
            "check_function": check_function,
            "init_callback_fn": init_callback_fn,
            "read_write_conflict": read_write_conflict,
            "num_workers": num_workers,
            "max_retries": max_retries,
            "fit": fit,
            "timeout": timeout,
            "upstream_tasks": upstream_tasks,
        }

        super().__init__(
            **{k: v for k, v in kwargs.items() if v is not None},
        )
