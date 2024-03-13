from .training_iteration_stats import TrainingIterationStats

import xarray as xr
import numpy as np

from typing import List
import attr

@attr.s
class TrainingStats:
    """
    A class used to represent Training Statistics.

    Attributes:
        iteration_stats: List[TrainingIterationStats]
            an ordered list of training stats.
        
    Methods:
        add_iteration_stats(iteration_stats: TrainingIterationStats) -> None:
            Add a new set of iterations stats to the existing list of iteration 
            stats.
        delete_after(iteration: int) -> None:
            Deletes training stats after a specified iteration number.
        trained_until() -> int:
            Gets the number of iterations that the model has been trained for.
        to_xarray() -> xr.DataArray:
            Converts the iteration statistics to a xarray data array.
    """
    iteration_stats: List[TrainingIterationStats] = attr.ib(
        default=attr.Factory(list),
        metadata={"help_text": "A ordered list of training stats."},
    )

    def add_iteration_stats(self, iteration_stats: TrainingIterationStats) -> None:
        """
        Add a new iteration stats to the current iteration stats.
        
        Args:
            iteration_stats (TrainingIterationStats): a new iteration stats object.
        
        Raises:
            assert: if the new iteration stats do not follow the order of existing iteration stats.
        """
        if len(self.iteration_stats) > 0:
            assert (
                iteration_stats.iteration == self.iteration_stats[-1].iteration + 1
            ), f"Expected iteration {self.iteration_stats[-1].iteration + 1}, got {iteration_stats.iteration}"

        self.iteration_stats.append(iteration_stats)

    def delete_after(self, iteration: int) -> None:
        """
        Deletes training stats after a specified iteration.
        
        Args:
            iteration (int): the iteration after which the stats are to be deleted.
        """
        self.iteration_stats = [
            stats for stats in self.iteration_stats if stats.iteration < iteration
        ]

    def trained_until(self) -> int:
        """
        The number of iterations trained for (the maximum iteration plus one).
        Returns zero if no iterations trained yet.
        
        Returns:
            int: number of iterations that the model has been trained for.
        """
        if not self.iteration_stats:
            return 0
        return self.iteration_stats[-1].iteration + 1

    def to_xarray(self) -> xr.DataArray:
        """
        Converts the iteration stats to a data array format easily manipulatable.
        
        Returns:
            xr.DataArray: xarray DataArray of iteration losses.
        """
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