import numpy as np


class ValidationScores:

    def __init__(self):

        self.iterations = []
        self.scores = []

    def add_validation_iteration(self, iteration, scores):

        self.iterations.append(iteration)
        self.scores.append(scores)

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

    def validated_until(self):
        """The number of iterations trained for (the maximum iteration plus
        one)."""

        if not self.iterations:
            return 0
        return max(self.iterations) + 1
