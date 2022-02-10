from dacapo.experiments.tasks.evaluators.evaluation_scores import EvaluationScores
from dacapo.experiments.datasplits.datasets import Dataset
from dacapo.experiments.tasks.post_processors import PostProcessorParameters

from abc import ABC, abstractmethod
from typing import Tuple
import math
import itertools


class Evaluator(ABC):
    """Base class of all evaluators.

    An evaluator takes a post-processor's output and compares it against
    ground-truth.
    """

    @abstractmethod
    def evaluate(self, output_array, evaluation_dataset):
        """Compare an output dataset against ground-truth from an evaluation
        dataset.

        The evaluation dataset is a dictionary mapping from ``DataKey`` to
        ``ArraySource`` or ``GraphSource``.

        Should return an instance of ``EvaluationScores``.
        """
        pass

    def is_best(
        self,
        dataset: Dataset,
        parameter: PostProcessorParameters,
        criterion: str,
        score: EvaluationScores,
    ) -> bool:
        """
        Check if the provided score is the best for this dataset/parameter/criterion combo
        """
        if not self.store_best(criterion) or math.isnan(getattr(score, criterion)):
            return False
        elif self._best_scores[(dataset, parameter, criterion)] is None:
            return True
        else:
            _, previous_best_score = self._best_scores[(dataset, parameter, criterion)]
            if score.higher_is_better(criterion):
                return getattr(score, criterion) > previous_best_score
            else:
                return getattr(score, criterion) < previous_best_score

    def set_best(self, validation_scores) -> None:
        """
        Find the best iteration for each dataset/post_processing_parameter/criterion
        """
        self._best_scores = {}
        scores = validation_scores.to_xarray()
        if len(validation_scores.iteration_scores) > 0:
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
            if best_scores is None:
                self._best_scores[(dataset, parameters, criterion)] = None
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
                    self._best_scores[
                        (
                            dataset,
                            parameters,
                            criterion,
                        )
                    ] = None
                else:
                    self._best_scores[
                        (
                            dataset,
                            parameters,
                            criterion,
                        )
                    ] = (winner_index.item(), winner_score.item())

    @property
    @abstractmethod
    def criteria(self):
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
        self.score.higher_is_better(criterion)

    def bounds(self, criterion: str) -> Tuple[float, float]:
        """
        The bounds for this criterion
        """
        self.score.bounds(criterion)

    def store_best(self, criterion: str) -> bool:
        """
        The bounds for this criterion
        """
        self.score.store_best(criterion)

    @property
    @abstractmethod
    def score(self) -> EvaluationScores:
        pass
