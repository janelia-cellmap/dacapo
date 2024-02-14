from abc import ABC, abstractmethod
import subprocess


class ComputeContext(ABC):
    @property
    @abstractmethod
    def device(self):
        pass

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
