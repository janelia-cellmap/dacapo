from importlib.metadata import metadata
from .validation_iteration_scores import ValidationIterationScores
from .tasks.evaluators import EvaluationScores
from .tasks.post_processors import PostProcessorParameters
from .datasplits.datasets import Dataset

from typing import List
import attr
import numpy as np
import inspect


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
        factor=lambda: list(),
        metadata={
            "help_text": "A list of evaluation scores and their associated post-processing parameters."
        },
    )

    def add_iteration_scores(self, iteration_scores):

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

    def get_attribute_names(self, class_instance):

        attributes = inspect.getmembers(
            class_instance, lambda a: not (inspect.isroutine(a))
        )
        names = [
            a[0]
            for a in attributes
            if not (a[0].startswith("__") and a[0].endswith("__"))
        ]

        return names

    @property
    def criteria(self):
        return self.evaluation_scores.criteria

    @property
    def parameter_names(self):
        return self.parameters[0].parameter_names

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

