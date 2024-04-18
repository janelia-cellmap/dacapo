import os
from upath import UPath as Path
from .compute_context import ComputeContext
import daisy

import attr

from typing import Optional


@attr.s
class Bsub(ComputeContext):
    distribute_workers: Optional[bool] = attr.ib(
        default=True,
        metadata={
            "help_text": "Whether to distribute the workers across multiple nodes or processes."
        },
    )
    """
    The Bsub class is a subclass of the ComputeContext class. It is used to specify the
    context in which computations are to be done. Bsub is used to specify that
    computations are to be done on a cluster using LSF.

    Attributes:
        queue (str): The queue to run on.
        num_gpus (int): The number of gpus to train on. Currently only 1 gpu can be used.
        num_cpus (int): The number of cpus to use to generate training data.
        billing (Optional[str]): Project name that will be paying for this Job.
    Methods:
        device(): Returns the device on which computations are to be done.
        _wrap_command(command): Wraps a command in the context specific command.
    Note:
        The class is a subclass of the ComputeContext class.

    """
    queue: str = attr.ib(default="local", metadata={"help_text": "The queue to run on"})
    num_gpus: int = attr.ib(
        default=1,
        metadata={
            "help_text": "The number of gpus to train on. "
            "Currently only 1 gpu can be used."
        },
    )
    num_cpus: int = attr.ib(
        default=5,
        metadata={"help_text": "The number of cpus to use to generate training data."},
    )
    billing: Optional[str] = attr.ib(
        default=None,
        metadata={"help_text": "Project name that will be paying for this Job."},
    )
    # log_dir: Optional[str] = attr.ib(
    #     default="~/logs/dacapo/",
    #     metadata={"help_text": "The directory to store the logs in."},
    # )

    @property
    def device(self):
        """
        A property method that returns the device on which computations are to be done.

        A device can be a CPU, GPU, TPU, etc. It is used to specify the context in which computations are to be done.

        Returns:
            str: The device on which computations are to be done.
        Examples:
            >>> context = Bsub()
            >>> device = context.device
        """
        if self.num_gpus > 0:
            return "cuda"
        else:
            return "cpu"

    def _wrap_command(self, command):
        """
        A helper method to wrap a command in the context specific command.

        Args:
            command (List[str]): The command to be wrapped.
        Returns:
            List[str]: The wrapped command.
        Examples:
            >>> context = Bsub()
            >>> command = ["python", "script.py"]
            >>> wrapped_command = context._wrap_command(command)
        """
        try:
            client = daisy.Client()
            basename = str(
                Path("./daisy_logs", client.task_id, f"worker_{client.worker_id}")
            )
        except:
            basename = "./daisy_logs/dacapo"
        return (
            [
                "bsub",
                "-q",
                f"{self.queue}",
                "-n",
                f"{self.num_cpus}",
                "-J",
                "dacapo",
                "-o",
                f"{basename}.out",
                "-e",
                f"{basename}.err",
            ]
            + (
                [
                    "-gpu",
                    f"num={self.num_gpus}",
                ]
                if self.num_gpus > 0
                else []
            )
            + (
                [
                    "-P",
                    f"{self.billing}",
                ]
                if self.billing is not None
                else []
            )
            + command
        )
