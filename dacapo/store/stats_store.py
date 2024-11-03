from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from dacapo.experiments.training_stats import TrainingStats
    from dacapo.experiments.validation_scores import (
        ValidationIterationScores,
        ValidationScores,
    )


class StatsStore(ABC):
    

    @abstractmethod
    def store_training_stats(self, run_name: str, training_stats: "TrainingStats"):
        
        pass

    @abstractmethod
    def retrieve_training_stats(self, run_name: str) -> "TrainingStats":
        
        pass

    @abstractmethod
    def store_validation_iteration_scores(
        self, run_name: str, validation_scores: "ValidationScores"
    ):
        
        pass

    @abstractmethod
    def retrieve_validation_iteration_scores(
        self, run_name: str
    ) -> List["ValidationIterationScores"]:
        
        pass

    @abstractmethod
    def delete_training_stats(self, run_name: str) -> None:
        
        pass
