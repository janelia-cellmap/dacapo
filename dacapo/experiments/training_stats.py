from .training_iteration_stats import TrainingIterationStats
from typing import List
import attr


@attr.s
class TrainingStats:

    iteration_stats: List[TrainingIterationStats] = attr.ib(
        default=attr.Factory(list))

    def add_iteration_stats(self, iteration_stats):

        self.iteration_stats.append(iteration_stats)

    def delete_after(self, iteration):

        self.iteration_stats = [
            stats
            for stats in self.iteration_stats
            if stats.iteration < iteration
        ]

    def trained_until(self):
        """The number of iterations trained for (the maximum iteration plus
        one)."""

        if not self.iteration_stats:
            return 0
        return max([stat.iteration for stat in self.iteration_stats]) + 1
