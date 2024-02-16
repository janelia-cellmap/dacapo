from abc import ABC, abstractmethod
import subprocess


class ComputeContext(ABC):
    @property
    @abstractmethod
    def device(self):
        pass

    def _wrap_command(self, command):
        # A helper method to wrap a command in the context
        # specific command.
        return command

    def wrap_command(self, command):
        command = [str(com) for com in self._wrap_command(command)]
        return " ".join(command)

    def execute(self, command):
        # A helper method to run a command in the context
        # specific way.
        return subprocess.run(self.wrap_command(command))
