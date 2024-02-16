"""
This script sets up a worker for the Dacapo Python library to perform data processing tasks. It performs these tasks 
using the ZarrArray class and LocalArrayIdentifier class. 

There are two main interfaces provided:
1. start_worker command: This gets arguments from the command line and then performs certain tasks such as getting arrays,
   waiting for blocks to run pipeline, and writing to output array.
2. spawn_worker function: This function is responsible for creating and running the worker in the given compute context.
   It sets up a command line for running the worker and then executes it with the selected compute context.

The script uses Daiy's Client instance to interact with the workers and manages the lifecycle of these workers.

Functions:
cli(log_level) -> None:
    This function sets up the command line interface of script with various options and
    sets the logging level of the interface.

start_worker(input_container: Path | str,input_dataset: str,output_container: Path | str,
             output_dataset: str,threshold: float = 0.0); -> None:
    This function grabs arrays, waits for blocks to run pipeline, and writes to an output array. It gets the necessary
    parameters from the command line options.

spawn_worker(input_array_identifier: "LocalArrayIdentifier", output_array_identifier: "LocalArrayIdentifier", 
             threshold: float = 0.0,compute_context: ComputeContext = LocalTorch()); -> Callable:
    This function creates and runs the worker in the given compute context.
    It sets up a command line for running the worker, and then executes it with the selected compute context. The function 
    returns the worker function.

__name__ == "__main__" -> None:
    This is the entry point of the script. It calls the command line interface function.
"""