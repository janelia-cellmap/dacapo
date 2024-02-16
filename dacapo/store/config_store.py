class DuplicateNameError(Exception):
    """Exception raised when an attempt is made to store a config with a name that already exists."""

class ConfigStore(ABC):
    """
    An abstract base class used to manage and access different configuration data.

    Subclasses need to implement methods for managing run, task, architecture, trainer,
    datasplit and array configs. 
    """

    @property
    @abstractmethod
    def runs(self):
        """
        Abstract getter method to be overridden by subclasses which 
        contains configuration data for all the runs.
        """
        pass

    @property
    @abstractmethod
    def datasplits(self):
        """
        Abstract getter method to be overridden by subclasses which 
        contains configuration data for all the data splits.
        """
        pass

    @property
    @abstractmethod
    def datasets(self):
        """
        Abstract getter method to be overridden by subclasses which 
        contains configuration data for all the datasets.
        """
        pass

    @property
    @abstractmethod
    def arrays(self):
        """
        Abstract getter method to be overridden by subclasses which 
        contains configuration data for all the arrays.
        """
        pass

    @property
    @abstractmethod
    def tasks(self):
        """
        Abstract getter method to be overridden by subclasses which 
        contains configuration data for all the tasks.
        """
        pass

    @property
    @abstractmethod
    def trainers(self):
        """
        Abstract getter method to be overridden by subclasses which 
        contains configuration data for all the trainers.
        """
        pass

    @property
    @abstractmethod
    def architectures(self):
        """
        Abstract getter method to be overridden by subclasses which 
        contains configuration data for all the architectures.
        """
        pass

    @abstractmethod
    def delete_config(self, database, config_name: str) -> None:
        """Delete a given configuration from the specific type(database) of configuration."""
        pass

    def delete_run_config(self, run_name: str) -> None:
        """Deletes a specific run configuration based on run name."""
        self.delete_config(self.runs, run_name)

    def delete_task_config(self, task_name: str) -> None:
        """Deletes a specific task configuration based on task name."""
        self.delete_config(self.tasks, task_name)

    def delete_architecture_config(self, architecture_name: str) -> None:
        """Deletes a specific architecture configuration based on architecture name."""
        self.delete_config(self.architectures, architecture_name)

    def delete_trainer_config(self, trainer_name: str) -> None:
        """Deletes a specific trainer configuration based on trainer name."""
        self.delete_config(self.trainers, trainer_name)

    def delete_datasplit_config(self, datasplit_name: str) -> None:
        """Deletes a specific datasplit configuration based on datasplit name."""
        self.delete_config(self.datasplits, datasplit_name)

    def delete_array_config(self, array_name: str) -> None:
        """Deletes a specific array configuration based on array name."""
        self.delete_config(self.arrays, array_name)