from .training_iteration_stats import TrainingIterationStats

import xarray as xr
import numpy as np

from typing import List
import attr


@attr.s
class TrainingStats:

    iteration_stats: List[TrainingIterationStats] = attr.ib(
        default=attr.Factory(list),
        metadata={"help_text": "A ordered list of training stats."},
    )

    def add_iteration_stats(self, iteration_stats):
        if len(self.iteration_stats) > 0:
            assert iteration_stats.iteration == self.iteration_stats[-1].iteration + 1

        self.iteration_stats.append(iteration_stats)

    def delete_after(self, iteration):

        self.iteration_stats = [
            stats for stats in self.iteration_stats if stats.iteration < iteration
        ]

    def trained_until(self):
        """The number of iterations trained for (the maximum iteration plus
        one)."""

        if not self.iteration_stats:
            return 0
        return self.iteration_stats[-1].iteration + 1

    def to_xarray(self):
        return xr.DataArray(
            np.array(
                [iteration_stat.loss for iteration_stat in self.iteration_stats]
            ).reshape((-1,)),
            dims=("iterations"),
            coords={
                "iterations": [
                    iteration_stat.iteration for iteration_stat in self.iteration_stats
                ],
            },
        )
