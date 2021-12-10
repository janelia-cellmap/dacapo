from .compute_context import ComputeContext

import torch
import attr

import subprocess
from typing import Optional


@attr.s
class Bsub(ComputeContext):
    queue: str = attr.ib(default="local", metadata={"help_text": "The queue to run on"})
    num_gpus: int = attr.ib(
        default=1,
        metadata={
            "help_text": "The number of gpus to train on. Currently only 1 gpu can be used."
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

    @property
    def device(self):
        return None

    def command(self, run_name):
        return [
            "bsub",
            "-q",
            f"{self.queue}",
            "-n",
            f"{self.num_cpus}",
            "-gpu",
            f"num={self.num_gpus}",
            "-J",
            run_name,
            "-o",
            f"{run_name}_train.out",
            "-e",
            f"{run_name}_train.err",
        ] + (
            [
                "-P",
                f"{self.billing}",
            ]
            if self.billing is not None
            else []
        )

    def train(self, run_name):
        subprocess.run(self.command(run_name) + ["dacapo", "train", "-r", run_name])
        return True
