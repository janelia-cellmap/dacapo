
class ValidationScores:

    def __init__(self):

        self.iterations = []
        self.scores = {}

    def add_validation_iteration(self, iteration, scores):

        self.iterations.append(iteration)
        self.scores.append(scores)

    def get_score_names(self):

        for parameters, sample_scores in self.scores.items():
            return sample_scores['scores']['average'].keys()

        raise RuntimeError("No scores were added, yet")

    def validated_until(self):
        """The number of iterations trained for (the maximum iteration plus
        one)."""

        if not self.iterations:
            return 0
        return max(self.iterations) + 1
