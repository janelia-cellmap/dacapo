"""
This Python script implements Bsub class inheriting from ComputeContext. The Bsub class represents a batch submission system such as LSF 
which is used to submit jobs to computing clusters. The Bsub class has attributes like queue, number of GPUs, number of CPUs and the 
billing project name. It includes a property 'device' to check whether GPUs are used and a method 'wrap_command' to submit the job 
to computing cluster with appropriate parameters.

Methods
-------
wrap_command(command):
    Returns the command to be executed on cluster after adding submission-related parameters

Properties
----------
device:
    Returns the device being used for computation - "cuda" if GPU is used else "cpu"
"""

@attr.s
class Bsub(ComputeContext):
    """
    Bsub class representing batch submission system like LSF for job submission. 

    Attributes
    ----------
    queue: str, default="local"
        The queue to run on
    num_gpus: int, default=1
        The number of GPUs to train on. Currently only 1 gpu can be used.
    num_cpus: int, default=5
        The number of CPUs to use to generate training data.
    billing: str, optional, default=None
        Project name that will be paying for this Job.
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

    @property
    def device(self):
        """
        Property that returns the device being used for computation. "cuda" if GPU is used else "cpu".

        Returns
        -------
        str
            The device being used for computation
        """
        if self.num_gpus > 0:
            return "cuda"
        else:
            return "cpu"

    def wrap_command(self, command):
        """
        Prepares the command to be executed on cluster by adding submit job-related parameters.

        Parameters
        ----------
        command : list
            The actual command to be executed on cluster

        Returns
        -------
        list
            The command to be submitted to cluster
        """
        return (
            [
                "bsub",
                "-q",
                f"{self.queue}",
                "-n",
                f"{self.num_cpus}",
                "-gpu",
                f"num={self.num_gpus}",
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