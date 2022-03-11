from abc import ABC, abstractmethod


class DuplicateNameError(Exception):
    pass


class ConfigStore(ABC):
    """Base class for configuration stores."""

    @abstractmethod
    def store_run_config(self, run_config):
        """Store a run config. This should also store the configs that are part
        of the run config (i.e., task, architecture, trainer, and dataset
        config)."""
        pass

    @abstractmethod
    def retrieve_run_config(self, run_name):
        """Retrieve a run config from a run name."""
        pass

    @abstractmethod
    def retrieve_run_config_names(self):
        """Retrieve all run config names."""
        pass

    @abstractmethod
    def store_task_config(self, task_config):
        """Store a task config."""
        pass

    @abstractmethod
    def retrieve_task_config(self, task_name):
        """Retrieve a task config from a task name."""
        pass

    @abstractmethod
    def retrieve_task_config_names(self):
        """Retrieve all task config names."""
        pass

    @abstractmethod
    def store_architecture_config(self, architecture_config):
        """Store a architecture config."""
        pass

    @abstractmethod
    def retrieve_architecture_config(self, architecture_name):
        """Retrieve a architecture config from a architecture name."""
        pass

    @abstractmethod
    def retrieve_architecture_config_names(self):
        """Retrieve all architecture config names."""
        pass

    @abstractmethod
    def store_trainer_config(self, trainer_config):
        """Store a trainer config."""
        pass

    @abstractmethod
    def retrieve_trainer_config(self, trainer_name):
        """Retrieve a trainer config from a trainer name."""
        pass

    @abstractmethod
    def retrieve_trainer_config_names(self):
        """Retrieve all trainer config names."""
        pass

    @abstractmethod
    def store_datasplit_config(self, datasplit_config):
        """Store a datasplit config."""
        pass

    @abstractmethod
    def retrieve_datasplit_config(self, datasplit_name):
        """Retrieve a datasplit config from a datasplit name."""
        pass

    @abstractmethod
    def retrieve_datasplit_config_names(self):
        """Retrieve all datasplit names."""
        pass

    @abstractmethod
    def store_array_config(self, array_config):
        """Store a array config."""
        pass

    @abstractmethod
    def retrieve_array_config(self, array_name):
        """Retrieve a array config from a array name."""
        pass

    @abstractmethod
    def retrieve_array_config_names(self):
        """Retrieve all array names."""
        pass
