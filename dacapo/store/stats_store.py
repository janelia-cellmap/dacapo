from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from dacapo.experiments.training_stats import TrainingStats
    from dacapo.experiments.validation_scores import ValidationIterationScores, ValidationScores


class StatsStore(ABC):
    """Base class for statistics stores."""

    @abstractmethod
    def store_training_stats(self, run_name: str, training_stats: "TrainingStats"):
        """Store training stats of a given run."""
        pass

    @abstractmethod
    def retrieve_training_stats(self, run_name: str) -> "TrainingStats":
        """Retrieve the training stats for a given run."""
        pass

    @abstractmethod
    def store_validation_iteration_scores(
        self, run_name: str, validation_scores: "ValidationScores"
    ):
        """Store the validation iteration scores of a given run."""
        pass

    @abstractmethod
    def retrieve_validation_iteration_scores(
        self, run_name: str
    ) -> List["ValidationIterationScores"]:
        """Retrieve the validation iteration scores for a given run."""
        pass

    @abstractmethod
    def delete_training_stats(self, run_name: str) -> None:
        pass
