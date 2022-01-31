from importlib.metadata import metadata
from .validation_iteration_scores import ValidationIterationScores
from .tasks.evaluators import EvaluationScores
from .tasks.post_processors import PostProcessorParameters
from .datasplits.datasets import Dataset

from typing import List, Tuple, Optional
import attr
import numpy as np
import xarray as xr
import inspect
import itertools


@attr.s
class ValidationScores:

    parameters: list[PostProcessorParameters] = attr.ib(
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

    def subscores(self, iteration_scores: List[ValidationIterationScores]):
        return ValidationScores(
            self.parameters,
            self.datasets,
            self.evaluation_scores,
            scores=iteration_scores,
        )

    def add_iteration_scores(
        self,
        iteration_scores: ValidationIterationScores,
    ):

        self.iteration_scores.append(iteration_scores)

    def delete_after(self, iteration):

        self.iteration_scores = [
            scores for scores in self.iteration_scores if scores.iteration < iteration
        ]

    def validated_until(self):
        """The number of iterations validated for (the maximum iteration plus
        one)."""

        if not self.iteration_scores:
            return 0
        return max([score.iteration for score in self.iteration_scores]) + 1

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
    def criteria(self):
        return self.evaluation_scores.criteria

    @property
    def parameter_names(self):
        return self.parameters[0].parameter_names

    def to_xarray(self):
        return xr.DataArray(
            np.array(
                [iteration_score.scores for iteration_score in self.iteration_scores]
            ).reshape(
                (-1, len(self.datasets), len(self.parameters), len(self.criteria))
            ),
            dims=("iterations", "datasets", "parameters", "criteria"),
            coords={
                "iterations": [
                    iteration_score.iteration
                    for iteration_score in self.iteration_scores
                ],
                "datasets": self.datasets,
                "parameters": self.parameters,
                "criteria": self.criteria,
            },
        )

    def best(self, array: xr.DataArray) -> List[Optional[xr.DataArray]]:
        """
        For each criterion in the criteria dimension, return the best value. May return None if there is no best.
        """
        criterion_bests = []
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

    def get_best(self, criterion=None, higher_is_better=True):
        """
        return the best score according to this criterion
        """

        names = self.get_score_names()
        postprocessor_parameter_names = self.get_postprocessor_parameter_names()

        best_scores = {name: [] for name in names}
        best_score_parameters = {name: [] for name in postprocessor_parameter_names}

        for iteration_score in self.iteration_scores:
            ips = np.array(
                [
                    getattr(parameter_score[1], criterion, np.nan)
                    for parameter_score in iteration_score.parameter_scores
                ],
                dtype=np.float32,
            )
            ips[np.isnan(ips)] = -np.inf if higher_is_better else np.inf
            i = np.argmax(ips) if higher_is_better else np.argmin(ips)
            best_score = iteration_score.parameter_scores[i]

            for name in names:
                try:
                    best_scores[name].append(getattr(best_score[1], name))
                except AttributeError as e:
                    raise AttributeError(iteration_score.iteration) from e

            for name in postprocessor_parameter_names:
                best_score_parameters[name].append(getattr(best_score[0], name))

        return (best_score_parameters, best_scores)

    def _get_best(self, criterion, dataset=None):
        """
        Get the best score according to this criterion.
        return iteration, parameters, score
        """
        iteration_bests = []
        for iteration_score in self.iteration_scores:
            parameters, iteration_best = iteration_score._get_best(criterion)
            iteration_bests.append(
                (iteration_score.iteration, parameters, iteration_best)
            )
