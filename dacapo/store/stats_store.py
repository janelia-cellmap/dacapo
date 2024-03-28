from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from dacapo.experiments.training_stats import TrainingStats
    from dacapo.experiments.validation_scores import (
        ValidationIterationScores,
        ValidationScores,
    )


class StatsStore(ABC):
    """
    Base class for statistics stores.

    Methods:
        store_training_stats(run_name, training_stats): Store the training stats of a given run.
        retrieve_training_stats(run_name): Retrieve the training stats for a given run.
        store_validation_iteration_scores(run_name, validation_scores): Store the validation iteration scores of a given run.
        retrieve_validation_iteration_scores(run_name): Retrieve the validation iteration scores for a given run.
        delete_training_stats(run_name): Delete the training stats associated with a specific run.
    """

    @abstractmethod
    def store_training_stats(self, run_name: str, training_stats: "TrainingStats"):
        """
        Store training stats of a given run.

        Args:
            run_name (str): The name of the run.
            training_stats (TrainingStats): The training stats to store.
        Raises:
            ValueError: If the training stats are already stored.
        Examples:
            >>> store = StatsStore()
            >>> run_name = 'run_0'
            >>> training_stats = TrainingStats()
            >>> store.store_training_stats(run_name, training_stats)
        """
        pass

    @abstractmethod
    def retrieve_training_stats(self, run_name: str) -> "TrainingStats":
        """
        Retrieve the training stats for a given run.

        Args:
            run_name (str): The name of the run.
        Returns:
            TrainingStats: The training stats for the given run.
        Raises:
            ValueError: If the training stats are not available.
        Examples:
            >>> store = StatsStore()
            >>> run_name = 'run_0'
            >>> store.retrieve_training_stats(run_name)
        """
        pass

    @abstractmethod
    def store_validation_iteration_scores(
        self, run_name: str, validation_scores: "ValidationScores"
    ):
        """
        Store the validation iteration scores of a given run.

        Args:
            run_name (str): The name of the run.
            validation_scores (ValidationScores): The validation scores to store.
        Raises:
            ValueError: If the validation iteration scores are already stored.
        Examples:
            >>> store = StatsStore()
            >>> run_name = 'run_0'
            >>> validation_scores = ValidationScores()
            >>> store.store_validation_iteration_scores(run_name, validation_scores)
        """
        pass

    @abstractmethod
    def retrieve_validation_iteration_scores(
        self, run_name: str
    ) -> List["ValidationIterationScores"]:
        """
        Retrieve the validation iteration scores for a given run.

        Args:
            run_name (str): The name of the run.
        Returns:
            List[ValidationIterationScores]: The validation iteration scores for the given run.
        Raises:
            ValueError: If the validation iteration scores are not available.
        Examples:
            >>> store = StatsStore()
            >>> run_name = 'run_0'
            >>> store.retrieve_validation_iteration_scores(run_name)
        """
        pass

    @abstractmethod
    def delete_training_stats(self, run_name: str) -> None:
        """
        Deletes the training statistics for a given run.

        Args:
            run_name (str): The name of the run.
        Raises:
            ValueError: If the training stats are not available.
        Example:
            >>> store = StatsStore()
            >>> run_name = 'run_0'
            >>> store.delete_training_stats(run_name)
        """
        pass
