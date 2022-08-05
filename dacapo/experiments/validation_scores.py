from importlib.metadata import metadata
from .validation_iteration_scores import ValidationIterationScores
from .tasks.evaluators import EvaluationScores
from .tasks.post_processors import PostProcessorParameters
from .datasplits.datasets import Dataset

from typing import List, Tuple, Optional
import attr
import numpy as np
import xarray as xr
import itertools


@attr.s
class ValidationScores:

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

        self.scores.append(iteration_scores)

    def delete_after(self, iteration: int) -> None:

        self.scores = [scores for scores in self.scores if scores.iteration < iteration]

    def validated_until(self) -> int:
        """The number of iterations validated for (the maximum iteration plus
        one)."""

        if not self.scores:
            return 0
        return max([score.iteration for score in self.scores]) + 1

    def compare(
        self, existing_iteration_scores: List[ValidationIterationScores]
    ) -> Tuple[bool, int]:
        """
        Compares iteration stats provided from elsewhere to scores we have saved locally.
        Local scores take priority. If local scores are at a lower iteration than the
        existing ones, delete the existing ones and replace with local.
        If local iteration > existing iteration, just update existing scores with the last
        overhanging local scores.
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
        return self.evaluation_scores.criteria

    @property
    def parameter_names(self) -> List[str]:
        return self.parameters[0].parameter_names

    def to_xarray(self) -> xr.DataArray:
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

    def best(self, array: xr.DataArray) -> List[Optional[xr.DataArray]]:
        """
        For each criterion in the criteria dimension, return the best value.
        May return None if there is no best.
        """
        criterion_bests: List[Optional[xr.DataArray]] = []
        for criterion in array.coords["criteria"].values:
            sub_array = array.sel(criteria=criterion)
            result = sub_array.where(sub_array == sub_array.max(), drop=True).squeeze()
            if result.size == 0:
                criterion_bests.append(None)
            if result.size == 1:
                criterion_bests.append(result)
            else:
                for coord in itertools.product(
                    *[coords.values for coords in result.coords]
                ):
                    current = result.sel(
                        **{d: [c] for d, c in zip(result.coords.keys(), coord)}
                    )
                    if current.value != float("nan"):
                        criterion_bests.append(current)
        return criterion_bests

    def get_best(
        self, data: xr.DataArray, dim: str
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute the Best scores along dimension "dim" per criterion.
        Returns both the index associated with the best value, and the
        best value in two seperate arrays.
        """
        if "criteria" in data.coords.keys():
            if len(data.coords["criteria"].shape) == 1:
                criteria_bests = []
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
                    data.coords["criteria"].item()
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
