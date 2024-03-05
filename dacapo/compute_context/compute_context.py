from abc import ABC, abstractmethod
import subprocess

from dacapo import Options, compute_context


class ComputeContext(ABC):
    @property
    @abstractmethod
    def device(self):
        pass

    def _wrap_command(self, command):
        # A helper method to wrap a command in the context specific command.
        return command

    def wrap_command(self, command):
        command = [str(com) for com in self._wrap_command(command)]
        return command

    def execute(self, command):
        # A helper method to run a command in the context specific way.
        subprocess.run(self.wrap_command(command))


def create_compute_context():
    """Create a compute context based on the global DaCapo options."""

    options = Options.instance()

    if hasattr(compute_context, options.compute_context["type"]):
        return getattr(compute_context, options.compute_context["type"])(
            **options.compute_context["config"]
        )
    else:
        raise ValueError(f"Unknown store type {options.type}")
