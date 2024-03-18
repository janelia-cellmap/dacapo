import os
from upath import UPath as Path
from .compute_context import ComputeContext
import daisy

import attr

from typing import Optional


@attr.s
class Bsub(ComputeContext):
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
        if self.num_gpus > 0:
            return "cuda"
        else:
            return "cpu"

    def _wrap_command(self, command):
        client = daisy.Client()
        basename = str(
            Path("./daisy_logs", client.task_id, f"worker_{client.worker_id}")
        )
        return (
            [
                "bsub",
                "-q",
                f"{self.queue}",
                "-n",
                f"{self.num_cpus}",
                "-gpu",
                f"num={self.num_gpus}",
                "-J",
                "dacapo",
                "-o",
                f"{basename}.out",
                "-e",
                f"{basename}.err",
            ]
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
