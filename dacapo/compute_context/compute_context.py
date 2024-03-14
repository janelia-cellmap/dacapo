from abc import ABC, abstractmethod
import subprocess

from dacapo import Options, compute_context
import logging

logger = logging.getLogger(__name__)

class ComputeContext(ABC):
    @property
    @abstractmethod
    def device(self):
        pass

    def _wrap_command(self, command):
        # A helper method to wrap a command in the context specific command.
        return command

    def wrap_command(self, command):
        logger.warning(f"Wrapping command {command} with {self}")
        command = [str(com) for com in self._wrap_command(command)]
        return command

    def execute(self, command):
        # A helper method to run a command in the context specific way.
        subprocess.run(self.wrap_command(command))


def create_compute_context():
    """Create a compute context based on the global DaCapo options."""
    return compute_context.Bsub()

    # options = Options.instance()

    # if hasattr(compute_context, options.compute_context["type"]):
    #     return getattr(compute_context, options.compute_context["type"])(
    #         **options.compute_context["config"]
    #     )
    # else:
    #     raise ValueError(f"Unknown store type {options.type}")
