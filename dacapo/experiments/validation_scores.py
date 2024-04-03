from .validation_iteration_scores import ValidationIterationScores
from .tasks.evaluators import EvaluationScores
from .tasks.post_processors import PostProcessorParameters
from .datasplits.datasets import Dataset

from typing import List, Tuple
import attr
import numpy as np
import xarray as xr


@attr.s
class ValidationScores:
    """
    Class representing the validation scores for a set of parameters and datasets.

    Attributes:
        parameters (List[PostProcessorParameters]): The list of parameters that are being evaluated.
        datasets (List[Dataset]): The datasets that will be evaluated at each iteration.
        evaluation_scores (EvaluationScores): The scores that are collected on each iteration per
            `PostProcessorParameters` and `Dataset`.
        scores (List[ValidationIterationScores]): A list of evaluation scores and their associated
            post-processing parameters.
    Methods:
        subscores(iteration_scores): Create a new ValidationScores object with a subset of the iteration scores.
        add_iteration_scores(iteration_scores): Add iteration scores to the list of scores.
        delete_after(iteration): Delete scores after a specified iteration.
        validated_until(): Get the number of iterations validated for (the maximum iteration plus one).
        compare(existing_iteration_scores): Compare iteration stats provided from elsewhere to scores we have saved locally.
        criteria(): Get the list of evaluation criteria.
        parameter_names(): Get the list of parameter names.
        to_xarray(): Convert the validation scores to an xarray DataArray.
        get_best(data, dim): Compute the Best scores along dimension "dim" per criterion.
    Notes:
        The `scores` attribute is a list of `ValidationIterationScores` objects, each of which
        contains the scores for a single iteration.
    """

    parameters: List[PostProcessorParameters] = attr.ib(
        metadata={"help_text": "The list of parameters that are being evaluated"}
    )
    datasets: List[Dataset] = attr.ib(
        metadata={"help_text": "The datasets that will be evaluated at each iteration"}
    )
    evaluation_scores: EvaluationScores = attr.ib(
        metadata={
            "help_text": "The scores that are collected on each iteration per "
            "`PostProcessorParameters` and `Dataset`"
        }
    )
    scores: List[ValidationIterationScores] = attr.ib(
        factory=lambda: list(),
        metadata={
            "help_text": "A list of evaluation scores and their associated post-processing parameters."
        },
    )

    def subscores(
        self, iteration_scores: List[ValidationIterationScores]
    ) -> "ValidationScores":
        """
        Create a new ValidationScores object with a subset of the iteration scores.

        Args:
            iteration_scores: The iteration scores to include in the new ValidationScores object.
        Returns:
            A new ValidationScores object with the specified iteration scores.
        Raises:
            ValueError: If the iteration scores are not in the list of scores.
        Examples:
            >>> validation_scores.subscores([validation_scores.scores[0]])
        Note:
            This method is used to create a new ValidationScores object with a subset of the
            iteration scores. This is useful when you want to create a new ValidationScores object
            that only contains the scores up to a certain iteration.

        """
        return ValidationScores(
            self.parameters,
            self.datasets,
            self.evaluation_scores,
            scores=iteration_scores,
        )

    def add_iteration_scores(
        self,
        iteration_scores: ValidationIterationScores,
    ) -> None:
        """
        Add iteration scores to the list of scores.

        Args:
            iteration_scores: The iteration scores to add.
        Raises:
            ValueError: If the iteration scores are already in the list of scores.
        Examples:
            >>> validation_scores.add_iteration_scores(validation_scores.scores[0])
        Note:
            This method is used to add iteration scores to the list of scores. This is useful when
            you want to add scores for a new iteration to the ValidationScores object.

        """
        self.scores.append(iteration_scores)

    def delete_after(self, iteration: int) -> None:
        """
        Delete scores after a specified iteration.

        Args:
            iteration: The iteration after which to delete the scores.
        Raises:
            ValueError: If the iteration scores are not in the list of scores.
        Examples:
            >>> validation_scores.delete_after(0)
        Note:
            This method is used to delete scores after a specified iteration. This is useful when
            you want to delete scores after a certain iteration.

        """
        self.scores = [scores for scores in self.scores if scores.iteration < iteration]

    def validated_until(self) -> int:
        """
        Get the number of iterations validated for (the maximum iteration plus one).

        Returns:
            The number of iterations validated for.
        Raises:
            ValueError: If there are no scores.
        Examples:
            >>> validation_scores.validated_until()
        Note:
            This method is used to get the number of iterations validated for (the maximum iteration
            plus one). This is useful when you want to know how many iterations have been validated.

        """
        if not self.scores:
            return 0
        return max([score.iteration for score in self.scores]) + 1

    def compare(
        self, existing_iteration_scores: List[ValidationIterationScores]
    ) -> Tuple[bool, int]:
        """
        Compare iteration stats provided from elsewhere to scores we have saved locally.
        Local scores take priority. If local scores are at a lower iteration than the
        existing ones, delete the existing ones and replace with local.
        If local iteration > existing iteration, just update existing scores with the last
        overhanging local scores.

        Args:
            existing_iteration_scores: The existing iteration scores to compare with.
        Returns:
            A tuple indicating whether the local scores should replace the existing ones
            and the existing iteration number.
        Raises:
            ValueError: If the iteration scores are not in the list of scores.
        Examples:
            >>> validation_scores.compare([validation_scores.scores[0]])
        Note:
            This method is used to compare iteration stats provided from elsewhere to scores we have
            saved locally. Local scores take priority. If local scores are at a lower iteration than
            the existing ones, delete the existing ones and replace with local. If local iteration >
            existing iteration, just update existing scores with the last overhanging local scores.
        """
        if not existing_iteration_scores:
            return False, 0
        existing_iteration = (
            max([score.iteration for score in existing_iteration_scores]) + 1
        )
        current_iteration = self.validated_until()
        if existing_iteration > current_iteration:
            return True, 0
        else:
            return False, existing_iteration

    @property
    def criteria(self) -> List[str]:
        """
        Get the list of evaluation criteria.

        Returns:
            The list of evaluation criteria.
        Raises:
            ValueError: If there are no scores.
        Examples:
            >>> validation_scores.criteria
        Note:
            This property is used to get the list of evaluation criteria. This is useful when you
            want to know what criteria are being used to evaluate the scores.
        """
        return self.evaluation_scores.criteria

    @property
    def parameter_names(self) -> List[str]:
        """
        Get the list of parameter names.

        Returns:
            The list of parameter names.
        Raises:
            ValueError: If there are no scores.
        Examples:
            >>> validation_scores.parameter_names
        Note:
            This property is used to get the list of parameter names. This is useful when you want
            to know what parameters are being used to evaluate the scores.
        """
        return self.parameters[0].parameter_names

    def to_xarray(self) -> xr.DataArray:
        """
        Convert the validation scores to an xarray DataArray.

        Returns:
            An xarray DataArray representing the validation scores.
        Raises:
            ValueError: If there are no scores.
        Examples:
            >>> validation_scores.to_xarray()
        Note:
            This method is used to convert the validation scores to an xarray DataArray. This is
            useful when you want to work with the validation scores as an xarray DataArray.

        """
        return xr.DataArray(
            np.array(
                [iteration_score.scores for iteration_score in self.scores]
            ).reshape(
                (-1, len(self.datasets), len(self.parameters), len(self.criteria))
            ),
            dims=("iterations", "datasets", "parameters", "criteria"),
            coords={
                "iterations": [
                    iteration_score.iteration for iteration_score in self.scores
                ],
                "datasets": self.datasets,
                "parameters": self.parameters,
                "criteria": self.criteria,
            },
        )

    def get_best(
        self, data: xr.DataArray, dim: str
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute the Best scores along dimension "dim" per criterion.
        Returns both the index associated with the best value, and the
        best value in two separate arrays.

        Args:
            data: The data array to compute the best scores from.
            dim: The dimension along which to compute the best scores.
        Returns:
            A tuple containing the index associated with the best value and the best value
            in two separate arrays.
        Raises:
            ValueError: If the criteria are not in the data array.
        Examples:
            >>> validation_scores.get_best(data, "iterations")
        Note:
            This method is used to compute the Best scores along dimension "dim" per criterion. It
            returns both the index associated with the best value and the best value in two separate
            arrays. This is useful when you want to know the best scores for a given data array.
            Fix: The method is currently not able to handle the case where the criteria are not in the data array.
            To fix this, we need to add a check to see if the criteria are in the data array and raise an error if they are not.

        """
        if "criteria" in data.coords.keys():
            if len(data.coords["criteria"].shape) > 1:
                criteria_bests: List[Tuple[xr.DataArray, xr.DataArray]] = []
                for criterion in data.coords["criteria"].values:
                    if self.evaluation_scores.higher_is_better(criterion.item()):
                        criteria_bests.append(
                            (
                                data.sel(criteria=criterion).idxmax(
                                    dim, skipna=True, fill_value=None
                                ),
                                data.sel(criteria=criterion).max(dim, skipna=True),
                            )
                        )
                    else:
                        criteria_bests.append(
                            (
                                data.sel(criteria=criterion).idxmin(
                                    dim, skipna=True, fill_value=None
                                ),
                                data.sel(criteria=criterion).min(dim, skipna=True),
                            )
                        )
                best_indexes, best_scores = zip(*criteria_bests)
                da_best_indexes, da_best_scores = (
                    xr.concat(best_indexes, dim=data.coords["criteria"]),
                    xr.concat(best_scores, dim=data.coords["criteria"]),
                )
                return (da_best_indexes, da_best_scores)
            else:
                if self.evaluation_scores.higher_is_better(
                    list(data.coords["criteria"].values)[
                        0
                    ]  # TODO: what is the intended behavior here? (hot fix in place)
                    # data.coords["criteria"].item()
                ):
                    return (
                        data.idxmax(dim, skipna=True, fill_value=None),
                        data.max(dim, skipna=True),
                    )
                else:
                    return (
                        data.idxmin(dim, skipna=True, fill_value=None),
                        data.min(dim, skipna=True),
                    )

        else:
            raise ValueError("Cannot determine 'best' without knowing the criterion")
