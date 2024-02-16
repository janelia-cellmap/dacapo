import xarray as xr

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List, TYPE_CHECKING
import math
import itertools

if TYPE_CHECKING:
    from dacapo.experiments.tasks.evaluators.evaluation_scores import EvaluationScores
    from dacapo.experiments.datasplits.datasets import Dataset
    from dacapo.experiments.datasplits.datasets.arrays import Array
    from dacapo.experiments.tasks.post_processors import PostProcessorParameters
    from dacapo.experiments.validation_scores import ValidationScores

# Custom types for improved readability
OutputIdentifier = Tuple["Dataset", "PostProcessorParameters", str]
Iteration = int
Score = float
BestScore = Optional[Tuple[Iteration, Score]]

class Evaluator(ABC):
    """
    Abstract base evaluator class. It provides the fundamental structure and methods for
    evaluators. A specific evaluator must inherent this class and implement its methods.

    Attributes
    ----------
    best_scores: Dict[OutputIdentifier, BestScore]
        Dictionary storing the best scores, indexed by OutputIdentifier which is a tuple
        of Dataset, PostProcessorParameters, and criteria string.
    
    """

    @abstractmethod
    def evaluate(
        self, output_array: "Array", eval_array: "Array"
    ) -> "EvaluationScores":
        """
        Compares and evaluates the output array against the evaluation array.

        Parameters
        ----------
        output_array : Array
            The output data array to evaluate
        eval_array : Array
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
        """
        Provides the best scores so far. If not available, an empty dictionary is
        created and returned.
        """
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
        Determine if the provided score is the best for a specific 
        dataset/parameter/criterion combination.

        Parameters
        ----------
        dataset : Dataset
            The dataset for which the evaluation is done
        parameter : PostProcessorParameters
            The post processing parameters used for the given dataset
        criterion : str
            The evaluation criterion
        score : EvaluationScores
            The calculated evaluation scores

        Returns
        -------
        bool
            True if the score is the best, False otherwise.
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
        Identify the best iteration for each dataset/post_processing_parameter/criterion
        and set them as the current best scores.

        Parameters
        ----------
        validation_scores : ValidationScores
            The validation scores from which the best are to be picked.
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
        Determines whether a higher score is better for the given criterion.

        Parameters
        ----------
        criterion : str
            The evaluation criterion

        Returns
        -------
        bool
            True if higher score is better, False otherwise.
        """
        return self.score.higher_is_better(criterion)

    def bounds(self, criterion: str) -> Tuple[float, float]:
        """
        Provides the bounds for the given evaluation criterion.

        Parameters
        ----------
        criterion : str
            The evaluation criterion

        Returns
        -------
        Tuple[float, float]
            The lower and upper bounds for the criterion.
        """
        return self.score.bounds(criterion)

    def store_best(self, criterion: str) -> bool:
        """
        Determine if the best scores should be stored for the given criterion.

        Parameters
        ----------
        criterion : str
            The evaluation criterion

        Returns
        -------
        bool
            True if best scores should be stored, False otherwise.
        """
        return self.score.store_best(criterion)

    @property
    @abstractmethod
    def score(self) -> "EvaluationScores":
        """
        The abstract property to get the overall score of the evaluation.

        Returns
        -------
        EvaluationScores
            The overall evaluation scores.
        """
        pass