from abc import ABC, abstractmethod
import subprocess


class ComputeContext(ABC):
    @property
    @abstractmethod
    def device(self):
        pass

    def train(self, run_name):
        # A helper method to run train in some other context.
        # This can be on a cluster, in a cloud, through bsub,
        # etc.
        # If training should be done locally, return False,
        # else return True.
        return False

    def wrap_command(self, command):
        # A helper method to wrap a command in the context
        # specific command.
        return command

    def execute(self, command):
        # A helper method to run a command in the context
        # specific way.
        return subprocess.run(self.wrap_command(command))

    def train(self, run_name):
        subprocess.run(self.wrap_command(["dacapo", "train", "-r", run_name]))
        return True
