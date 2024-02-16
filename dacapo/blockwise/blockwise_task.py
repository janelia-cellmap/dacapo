"""
This python module defines a class `DaCapoBlockwiseTask` which extends the `Task` class from the `daisy` library.
The class makes use of the compute context from the `dacapo` library and provides utility for spawning
worker processes to perform the tasks.

Classes:

- `DaCapoBlockwiseTask`: Class that extends the `Task` class from `daisy` library.

"""


class DaCapoBlockwiseTask(Task):
    """
    A DaCapo blockwise task that provides features to setup and execute tasks according 
    to specific context.


    Attributes:
    ----------
    worker_file (str | Path): The workflow file for a worker process.
    compute_context (ComputeContext | str): Compute context instance of a worker process.
    total_roi: Total region of interest for a task.
    read_roi: The region of interest that is to be read for a task.
    write_roi: The region of interest that is to be written for a task.
    num_workers (int, optional): Number of workers for the task. Default is 16.
    max_retries (int, optional): Maximum number of retries for executing a task. Default is 2.
    timeout: Maximum duration to wait for a task to finish execution.
    upstream_tasks: Tasks that need to be executed before the current task.
    """

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
        """
        Constructor method to initialize a DaCapo blockwise task.
        """
