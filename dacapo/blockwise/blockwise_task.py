from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path
from daisy import Task, Roi


class DaCapoBlockwiseTask(Task):
    def __init__(
        self,
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