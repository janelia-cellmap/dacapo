```python
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from dacapo.experiments.training_stats import TrainingStats
    from dacapo.experiments.validation_scores import (
        ValidationIterationScores,
        ValidationScores,
    )


class StatsStore(ABC):
    """Abstract base class that all StatsStore classes should inherit from.

    This class lays out the basic structure of a StatsStore. All StatsStore classes
    must implement these abstract methods for storing, retrieving and deleting
    training or validation stats.
    """

    @abstractmethod
    def store_training_stats(self, run_name: str, training_stats: "TrainingStats"):
        """Abstract method for storing training stats for a specified run.

        Args:
            run_name: The name of the run for which stats should be stored.
            training_stats: The TrainingStats object to be stored.
        """
        pass

    @abstractmethod
    def retrieve_training_stats(self, run_name: str) -> "TrainingStats":
        """Abstract method for retrieving training stats for a specified run.

        Args:
            run_name: The name of the run for which stats should be retrieved.

        Returns:
            A TrainingStats object with the retrieved stats.
        """
        pass

    @abstractmethod
    def store_validation_iteration_scores(
        self, run_name: str, validation_scores: "ValidationScores"
    ):
        """Abstract method for storing validation iteration scores for a specified run.

        Args:
            run_name: The name of the run for which stats should be stored.
            validation_scores: The ValidationScores object to be stored.
        """
        pass

    @abstractmethod
    def retrieve_validation_iteration_scores(
        self, run_name: str
    ) -> List["ValidationIterationScores"]:
        """Abstract method for retrieving validation iteration scores for a specified run.

        Args:
            run_name: The name of the run for which scores should be retrieved.

        Returns:
            A list of ValidationIterationScores objects with the retrieved scores.
        """
        pass

    @abstractmethod
    def delete_training_stats(self, run_name: str) -> None:
        """Abstract method for deleting training stats for a specified run.

        Args:
            run_name: The name of the run for which stats should be deleted.
        """
        pass
```