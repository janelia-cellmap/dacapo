from .validation_iteration_scores import ValidationIterationScores
from typing import List
import attr
import numpy as np
import inspect

@attr.s
class ValidationScores:

    iteration_scores: List[ValidationIterationScores] = attr.ib(
        default=attr.Factory(list),
        metadata={"help_text": "An ordered list of validation scores by iteration."},
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
            class_instance, lambda a: not(inspect.isroutine(a)))
        names = [a[0] for a in attributes if not(
            a[0].startswith('__') and a[0].endswith('__'))]

        return names

    def get_score_names(self):

        if self.iteration_scores:
            example_parameter_scores = self.iteration_scores[0].parameter_scores
            score_class_instance = example_parameter_scores[0][1]
            return self.get_attribute_names(score_class_instance)

        raise RuntimeError("No scores were added, yet")

    def get_postprocessor_parameter_names(self):

        if self.iteration_scores:
            example_parameter_scores = self.iteration_scores[0].parameter_scores
            postprocessor_class_instance = example_parameter_scores[0][0]
            return self.get_attribute_names(postprocessor_class_instance)

        raise RuntimeError("No scores were added, yet")

    def get_best(self, score_name=None, higher_is_better=True):

        names = self.get_score_names()
        postprocessor_parameter_names = self.get_postprocessor_parameter_names()

        best_scores = {name: [] for name in names}
        best_score_parameters = {name: []
                                 for name in postprocessor_parameter_names}

        for iteration_score in self.iteration_scores:
            ips = np.array([
                getattr(parameter_score[1], score_name, np.nan)
                for parameter_score in iteration_score.parameter_scores
            ], dtype=np.float32)
            ips[np.isnan(ips)] = -np.inf if higher_is_better else np.inf
            i = np.argmax(ips) if higher_is_better else np.argmin(ips)
            best_score = iteration_score.parameter_scores[i]

            for name in names:
                best_scores[name].append(
                    getattr(best_score[1], name)
                )

            for name in postprocessor_parameter_names:
                best_score_parameters[name].append(
                    getattr(best_score[0], name)
                )

        return (best_score_parameters, best_scores)

'''
    def get_score_names(self):

        for scores in self.scores:
            for parameters, sample_scores in scores.items():
                return sample_scores['scores']['average'].keys()

        raise RuntimeError("No scores were added, yet")

    def get_best(self, score_name=None, higher_is_better=True):

        names = self.get_score_names()

        best_scores = {name: [] for name in names}
        for iteration_scores in self.scores:
            ips = np.array([
                parameter_scores['scores']['average'].get(score_name, np.nan)
                for parameter_scores in iteration_scores.values()
            ], dtype=np.float32)
            ips[np.isnan(ips)] = -np.inf if higher_is_better else np.inf
            i = np.argmax(ips) if higher_is_better else np.argmin(ips)
            for name in names:
                best_scores[name].append(
                    list(iteration_scores.values())[i]['scores']['average'].get(name, 0)
                )
        return best_scores
'''
