from abc import ABC, abstractmethod
from typing import Optional
import attr
import subprocess

from dacapo import Options, compute_context


class ComputeContext(ABC):
    distribute_workers: Optional[bool] = attr.ib(
        default=False,
        metadata={
            "help_text": "Whether to distribute the workers across multiple nodes or processes."
        },
    )
    """
    The ComputeContext class is an abstract base class for defining the context in which computations are to be done.
    It is inherited from the built-in class `ABC` (Abstract Base Classes). Other classes can inherit this class to define
    their own specific variations of the context. It requires to implement several property methods, and also includes
    additional methods related to the context design.

    Attributes:
        device: The device on which computations are to be done.
    Methods:
        _wrap_command(command): Wraps a command in the context specific command.
        wrap_command(command): Wraps a command in the context specific command and returns it.
        execute(command): Runs a command in the context specific way.
    Note:
        The class is abstract and requires to implement the abstract methods.
    """

    @property
    @abstractmethod
    def device(self):
        """
        Abstract property method to define the device on which computations are to be done.

        A device can be a CPU, GPU, TPU, etc. It is used to specify the context in which computations are to be done.

        Returns:
            str: The device on which computations are to be done.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> context = ComputeContext()
            >>> device = context.device
        Note:
            The method should be implemented in the derived class.
        """
        pass

    def _wrap_command(self, command):
        """
        A helper method to wrap a command in the context specific command.

        Args:
            command (List[str]): The command to be wrapped.
        Returns:
            List[str]: The wrapped command.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> context = ComputeContext()
            >>> command = ["python", "script.py"]
            >>> wrapped_command = context._wrap_command(command)
        Note:
            The method should be implemented in the derived class.
        """
        # A helper method to wrap a command in the context specific command.
        return command

    def wrap_command(self, command):
        """
        A method to wrap a command in the context specific command and return it.

        Args:
            command (List[str]): The command to be wrapped.
        Returns:
            List[str]: The wrapped command.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> context = ComputeContext()
            >>> command = ["python", "script.py"]
            >>> wrapped_command = context.wrap_command(command)
        Note:
            The method should be implemented in the derived class.
        """
        command = [str(com) for com in self._wrap_command(command)]
        return command

    def execute(self, command):
        """
        A method to run a command in the context specific way.

        Args:
            command (List[str]): The command to be executed.
        Raises:
            NotImplementedError: If the method is not implemented in the derived class.
        Examples:
            >>> context = ComputeContext()
            >>> command = ["python", "script.py"]
            >>> context.execute(command)
        Note:
            The method should be implemented in the derived class.
        """
        # A helper method to run a command in the context specific way.

        # add pythonpath to the environment
        print("Spawning worker...")
        print("Spawning worker with command: ", self.wrap_command(command))
        # os.environ["PYTHONPATH"] = sys.executable
        subprocess.run(self.wrap_command(command))


def create_compute_context() -> ComputeContext:
    """
    Create a compute context based on the global DaCapo options.

    Returns:
        ComputeContext: The compute context object.
    Raises:
        ValueError: If the store type is unknown.
    Examples:
        >>> context = create_compute_context()
    Note:
        The method is implemented in the module.
    """

    options = Options.instance()

    if hasattr(compute_context, options.compute_context["type"]):
        return getattr(compute_context, options.compute_context["type"])(
            **options.compute_context["config"]
        )
    else:
        raise ValueError(f"Unknown store type {options.type}")
