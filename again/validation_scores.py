import numpy as np


class ValidationScores:

    def __init__(self):

        self.iterations = []
        self.sample_scores = {}

    def add_validation_iteration(self, iteration, sample_scores):

        self.iterations.append(iteration)

        for sample, scores in sample_scores.items():
            if sample not in self.sample_scores:
                self.sample_scores[sample] = {}
            for score, value in scores.items():
                if score not in self.sample_scores[sample]:
                    self.sample_scores[sample][score] = []
                self.sample_scores[sample][score].append(value)

    def get_averages(self):
        """Get the average scores over all samples."""

        averages = {}
        for sample, scores in self.sample_scores.items():
            for score, values in scores.items():
                if score not in averages:
                    averages[score] = np.array(values)
                else:
                    averages[score] += np.array(values)

        num_samples = len(self.sample_scores)
        for score in averages.keys():
            averages[score] /= num_samples

        return averages

    def get_score_names(self):

        for sample, scores in self.sample_scores.items():
            return scores.keys()

        raise RuntimeError("No scores were added, yet")

    def validated_until(self):
        """The number of iterations trained for (the maximum iteration plus
        one)."""

        if not self.iterations:
            return 0
        return max(self.iterations) + 1
