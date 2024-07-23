from .training_iteration_stats import TrainingIterationStats

import xarray as xr
import numpy as np

from typing import List
import attr
import logging

logger = logging.getLogger(__name__)


@attr.s
class TrainingStats:
    """
    A class used to represent Training Statistics. It contains a list of training
    iteration statistics. It also provides methods to add new iteration stats,
    delete stats after a specified iteration, get the number of iterations trained
    for, and convert the stats to a xarray data array.

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
    Note:
        The iteration stats list is structured as follows:
        - The outer list contains the stats for each iteration.
        - The inner list contains the stats for each training iteration.
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
        Examples:
            >>> training_stats = TrainingStats()
            >>> training_stats.add_iteration_stats(TrainingIterationStats(0, 0.1))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(1, 0.2))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(2, 0.3))
            >>> training_stats.iteration_stats
            [TrainingIterationStats(iteration=0, loss=0.1),
             TrainingIterationStats(iteration=1, loss=0.2),
             TrainingIterationStats(iteration=2, loss=0.3)]
        Note:
            The iteration stats list is structured as follows:
            - The outer list contains the stats for each iteration.
            - The inner list contains the stats for each training iteration.
        """
        if len(self.iteration_stats) > 0:
            if iteration_stats.iteration <= self.iteration_stats[-1].iteration:
                logger.error(
                    f"Expected iteration {self.iteration_stats[-1].iteration + 1}, got {iteration_stats.iteration}. will remove stats after {iteration_stats.iteration-1}"
                )
                self.delete_after(iteration_stats.iteration - 1)

        self.iteration_stats.append(iteration_stats)

    def delete_after(self, iteration: int) -> None:
        """
        Deletes training stats after a specified iteration.

        Args:
            iteration (int): the iteration after which the stats are to be deleted.
        Raises:
            assert: if the iteration number is less than the maximum iteration number.
        Examples:
            >>> training_stats = TrainingStats()
            >>> training_stats.add_iteration_stats(TrainingIterationStats(0, 0.1))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(1, 0.2))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(2, 0.3))
            >>> training_stats.delete_after(1)
            >>> training_stats.iteration_stats
            [TrainingIterationStats(iteration=0, loss=0.1)]
        Note:
            The iteration stats list is structured as follows:
            - The outer list contains the stats for each iteration.
            - The inner list contains the stats for each training iteration.
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
        Raises:
            assert: if the iteration stats list is empty.
        Examples:
            >>> training_stats = TrainingStats()
            >>> training_stats.add_iteration_stats(TrainingIterationStats(0, 0.1))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(1, 0.2))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(2, 0.3))
            >>> training_stats.trained_until()
            3
        Note:
            The iteration stats list is structured as follows:
            - The outer list contains the stats for each iteration.
            - The inner list contains the stats for each training iteration.
        """
        if not self.iteration_stats:
            return 0
        return self.iteration_stats[-1].iteration + 1

    def to_xarray(self) -> xr.DataArray:
        """
        Converts the iteration stats to a data array format easily manipulatable.

        Returns:
            xr.DataArray: xarray DataArray of iteration losses.
        Raises:
            assert: if the iteration stats list is empty.
        Examples:
            >>> training_stats = TrainingStats()
            >>> training_stats.add_iteration_stats(TrainingIterationStats(0, 0.1))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(1, 0.2))
            >>> training_stats.add_iteration_stats(TrainingIterationStats(2, 0.3))
            >>> training_stats.to_xarray()
            <xarray.DataArray (iterations: 3)>
            array([0.1, 0.2, 0.3])
            Coordinates:
              * iterations  (iterations) int64 0 1 2
        Note:
            The iteration stats list is structured as follows:
            - The outer list contains the stats for each iteration.
            - The inner list contains the stats for each training iteration.
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
