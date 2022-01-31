from abc import ABC, abstractmethod


class StatsStore(ABC):
    """Base class for statistics stores.
    """

    @abstractmethod
    def store_training_stats(self, run_name, training_stats):
        """Store training stats of a given run."""
        pass

    @abstractmethod
    def retrieve_training_stats(self, run_name):
        """Retrieve the training stats for a given run."""
        pass

    @abstractmethod
    def store_validation_iteration_scores(self, run_name, validation_scores):
        """Store the validation iteration scores of a given run."""
        pass

    @abstractmethod
    def retrieve_validation_iteration_scores(self, run_name):
        """Retrieve the validation iteration scores for a given run."""
        pass
