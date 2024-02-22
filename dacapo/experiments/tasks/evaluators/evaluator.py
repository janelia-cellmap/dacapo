import xarray as xr

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, TYPE_CHECKING
import math
import itertools

if TYPE_CHECKING:
    from dacapo.experiments.tasks.evaluators.evaluation_scores import EvaluationScores
    from dacapo.experiments.datasplits.datasets import Dataset
    from dacapo.experiments.datasplits.datasets.arrays import Array
    from dacapo.store.local_array_store import LocalArrayIdentifier
    from dacapo.experiments.tasks.post_processors import PostProcessorParameters
    from dacapo.experiments.validation_scores import ValidationScores

# Custom types for improved readability
OutputIdentifier = Tuple["Dataset", "PostProcessorParameters", str]
Iteration = int
Score = float
BestScore = Optional[Tuple[Iteration, Score]]


class Evaluator(ABC):
    """Base class of all evaluators.

    An evaluator takes a post-processor's output and compares it against
    ground-truth.
    """

    @abstractmethod
    def evaluate(
        self, output_array_identifier: "LocalArrayIdentifier", evaluation_array: "Array"
    ) -> "EvaluationScores":
        """
        Compares and evaluates the output array against the evaluation array.

        Parameters
        ----------
        output_array_identifier : Array
            The output data array to evaluate
        evaluation_array : Array
            The evaluation data array to compare with the output

        Returns
        -------
        EvaluationScores
            The detailed evaluation scores after the comparison.
        """
        pass

    @property
    def best_scores(
        self,
    ) -> Dict[OutputIdentifier, BestScore]:
        if not hasattr(self, "_best_scores"):
            self._best_scores: Dict[OutputIdentifier, BestScore] = {}
        return self._best_scores

    def is_best(
        self,
        dataset: "Dataset",
        parameter: "PostProcessorParameters",
        criterion: str,
        score: "EvaluationScores",
    ) -> bool:
        """
        Check if the provided score is the best for this dataset/parameter/criterion combo
        """
        if not self.store_best(criterion) or math.isnan(getattr(score, criterion)):
            return False
        previous_best = self.best_scores[(dataset, parameter, criterion)]
        if previous_best is None:
            return True
        else:
            _, previous_best_score = previous_best
            if score.higher_is_better(criterion):
                return getattr(score, criterion) > previous_best_score
            else:
                return getattr(score, criterion) < previous_best_score

    def set_best(self, validation_scores: "ValidationScores") -> None:
        """
        Find the best iteration for each dataset/post_processing_parameter/criterion
        """
        scores = validation_scores.to_xarray()

        # type these variables for mypy
        best_indexes: Optional[xr.DataArray]
        best_scores: Optional[xr.DataArray]

        if len(validation_scores.scores) > 0:
            best_indexes, best_scores = validation_scores.get_best(
                scores, dim="iterations"
            )
        else:
            best_indexes, best_scores = None, None
        for dataset, parameters, criterion in itertools.product(
            scores.coords["datasets"].values,
            scores.coords["parameters"].values,
            scores.coords["criteria"].values,
        ):
            if not self.store_best(criterion):
                continue
            if best_scores is None or best_indexes is None:
                self.best_scores[(dataset, parameters, criterion)] = None
            else:
                winner_index, winner_score = (
                    best_indexes.sel(
                        datasets=dataset, parameters=parameters, criteria=criterion
                    ),
                    best_scores.sel(
                        datasets=dataset, parameters=parameters, criteria=criterion
                    ),
                )
                if math.isnan(winner_score.item()):
                    self.best_scores[
                        (
                            dataset,
                            parameters,
                            criterion,
                        )
                    ] = None
                else:
                    self.best_scores[
                        (
                            dataset,
                            parameters,
                            criterion,
                        )
                    ] = (winner_index.item(), winner_score.item())

    @property
    @abstractmethod
    def criteria(self) -> List[str]:
        """
        A list of all criteria for which a model might be "best". i.e. your
        criteria might be "precision", "recall", and "jaccard". It is unlikely
        that the best iteration/post processing parameters will be the same
        for all 3 of these criteria
        """
        pass

    def higher_is_better(self, criterion: str) -> bool:
        """
        Wether or not higher is better for this criterion.
        """
        return self.score.higher_is_better(criterion)

    def bounds(self, criterion: str) -> Tuple[float, float]:
        """
        The bounds for this criterion
        """
        return self.score.bounds(criterion)

    def store_best(self, criterion: str) -> bool:
        """
        The bounds for this criterion
        """
        return self.score.store_best(criterion)

    @property
    @abstractmethod
    def score(self) -> "EvaluationScores":
        pass
