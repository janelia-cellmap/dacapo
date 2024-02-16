from pathlib import Path
import daisy
from funlib.geometry import BoundingBox

from dacapo.context import ComputeContext
from dacapo.tasks import BlockwiseTask

def run_blockwise(
  worker_file: str | Path,
  context: ComputeContext | str,
  total_box: BoundingBox,
  read_box: BoundingBox,
  write_box: BoundingBox,
  num_workers: int = 16,
  max_attempts: int = 2,
  timeout=None,
  dependencies=None,
  *args,
  **kwargs,
):
  """
  Coordinate a blockwise computation over a large volume.

  Args:
    worker_file (str or Path): The path to a Python file which defines the 
    method to be run, the process to spawn workers, and the check to be
    applied after each worker's computation.
    
    context (ComputeContext or str): The context to use for computation.
    May either be a ComputeContext instance or a string from which a context
    can be derived.
    
    total_box (BoundingBox): The total bounding box over which to cover 
    with computations.
    
    read_box (BoundingBox): The bounding box for which each worker must 
    read data. This box will be translated across the total_box for each 
    worker.
    
    write_box (BoundingBox): The bounding box within which each worker will 
    write data. This box will be translated across the total_box for each 
    worker.
    
    num_workers (int, optional): The number of workers to accommodate.
    Defaults to 16.
    
    max_attempts (int, optional): The maximum number of times a worker's 
    computation will be attempted, in the event of failure. Defaults to 2.
    
    timeout (None, optional): If a computation runs for longer than this 
    value, it will be cancelled. By default, there is no limit.
    
    dependencies (None, optional): Other tasks that this task depends on. 
    By default, this task is assumed to have no dependencies.
    
    *args: Additional arguments to pass to the worker computation.
    **kwargs: Additional keyword arguments to pass to the worker computation.

  Returns:
    list: A list of the results returned by each worker's computation.
  """
  
  # create the task
  task = BlockwiseTask(
    worker_file,
    context,
    total_box,
    read_box,
    write_box,
    num_workers,
    max_attempts,
    timeout,
    dependencies,
    *args,
    **kwargs,
  )
  
  # run the task with Daisy
  return daisy.run_blockwise([task])