from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from daisy import Task, Roi
from dacapo.compute_context import ComputeContext
import dacapo.compute_context


class DaCapoBlockwiseTask(Task):
    def __init__(
        self,
        worker_file: str | Path,
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
        if isinstance(compute_context, str):
            compute_context = getattr(dacapo.compute_context, compute_context)()

        # Make the task_id unique
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_id = worker_file + timestamp

        # Load worker functions
        worker_name = Path(worker_file).stem
        worker = SourceFileLoader(worker_name, str(worker_file)).load_module()

        process_function = worker.spawn_worker(
            *args, **kwargs, compute_context=compute_context
        )
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

        super().__init__(
            task_id,
            total_roi,
            read_roi,
            write_roi,
            process_function,
            check_function,
            init_callback_fn,
            read_write_conflict,
            num_workers,
            max_retries,
            fit,
            timeout,
            upstream_tasks,
        )
