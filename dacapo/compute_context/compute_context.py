"""
This module provides an abstract base class (ABC) for a ComputeContext.
A ComputeContext is an object that wraps the specific detail of 
where and how computations will be carried out.

"""

from abc import ABC, abstractmethod
import subprocess

class ComputeContext(ABC):
    """
    Abstract Base Class for defining compute context.

    The ComputeContext is a way to encapsulate all of the details 
    and variations that occur between different hardware and software 
    environments in which computations may be carried out.

    """

    @property
    @abstractmethod
    def device(self):
        """
        Abstract method that must be implemented in any concrete class.
        It should return the device where computations will be carried out.
        """
        pass

    def wrap_command(self, command):
        """
        Takes a command as input, and returns the command wrapped for the 
        specific compute context.

        Args:
            command (list or str): The command that needs to be wrapped.

        Returns:
            list or str: The wrapped command.
        """
        return command

    def execute(self, command):
        """
        Runs a command in the context specific way by using subprocess.run. 
        Before running, the command is wrapped using wrap_command.

        Args:
            command (list or str): The command to be executed.

        Returns:
            CompletedProcess: A subprocess.CompletedProcess instance, 
            which represents the process that was run.
        """
        return subprocess.run(self.wrap_command(command))

    def train(self, run_name):
        """
        Runs dacapo train command for given run name.

        Args:
            run_name (str): The name of the run for training.

        Returns:
            bool: Returns True after training command has been executed.
        """
        subprocess.run(self.wrap_command(["dacapo", "train", "-r", run_name]))
        return True