import numpy as np
from gunpowder.nodes.batch_filter import BatchFilter
from collections.abc import Iterable
import logging

logger = logging.getLogger(__file__)


class GammaAugment(BatchFilter):
    """
    Class for applying gamma noise augmentation.

    Attributes:
        arrays: An iterable collection of np arrays to augment
        gamma_min: A float representing the lower limit of gamma perturbation
        gamma_max: A float representing the upper limit of gamma perturbation

    Methods:
        setup(): Method to configure the internal state of the class
        process(): Method to apply gamma noise to the desired arrays
        __augment(): Private method to perform the actual augmentation

    """

    def __init__(self, arrays, gamma_min, gamma_max):
        """
        Initializing the Variables.

        Args:
            arrays : An iterable collection of np arrays to augment
            gamma_min : A float representing  the lower limit of gamma perturbation
            gamma_max : A float representing the upper limit of gamma perturbation
        Raises:
            AssertionError: If gamma_max is less than gamma_min
        Examples:
            >>> GammaAugment(arrays, gamma_min, gamma_max)
            GammaAugment(arrays, gamma_min, gamma_max)
        """
        if not isinstance(arrays, Iterable):
            arrays = [
                arrays,
            ]
        self.arrays = arrays
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        assert self.gamma_max >= self.gamma_min

    def setup(self):
        """
        Configuring the internal state by iterating over arrays.

        Raises:
            AssertionError: If the array data type is not float32 or float64
        Examples:
            >>> setup()
            setup()
        """
        for array in self.arrays:
            self.updates(array, self.spec[array])

    def process(self, batch, request):
        """
        Method to apply gamma noise to the desired arrays.

        Args:
            batch : The input batch to be processed.
            request : An object which holds the requested output location.
        Returns:
            The batch with the gamma noise applied.
        Raises:
            AssertionError: If the array data type is not float32 or float64
        Examples:
            >>> process(batch, request)
            process(batch, request)
        """
        sample_gamma_min = (max(self.gamma_min, 1.0 / self.gamma_min) - 1) * (-1) ** (
            self.gamma_min < 1
        )
        sample_gamma_max = (max(self.gamma_max, 1.0 / self.gamma_max) - 1) * (-1) ** (
            self.gamma_max < 1
        )
        gamma = np.random.uniform(sample_gamma_min, sample_gamma_max)
        if gamma < 0:
            gamma = 1.0 / (-gamma + 1)
        else:
            gamma = gamma + 1
        for array in self.arrays:
            raw = batch.arrays[array]

            assert raw.data.dtype == np.float32 or raw.data.dtype == np.float64, (
                "Gamma augmentation requires float "
                "types for the raw array (not "
                + str(raw.data.dtype)
                + "). Consider using Normalize before."
            )

            raw.data = self.__augment(raw.data, gamma)

    def __augment(self, a, gamma):
        """
        Private method to perform the actual augmentation.

        Args:
            a: raw array to be augmented
            gamma: gamma index to be applied
        Returns:
            The augmented array.
        Raises:
            AssertionError: If the array data type is not float32 or float64
        Examples:
            >>> __augment(a, gamma)
            __augment(a, gamma)
        """
        # normalize a
        a_min = a.min()
        a_max = a.max()
        if abs(a_min - a_max) > 1e-3:
            # apply gamma noise
            a = (a - a_min) / (a_max - a_min)
            noisy_a = a**gamma
            # undo normalization
            noisy_a = a * (a_max - a_min) + a_min
            return noisy_a
        else:
            logger.info("Skipping gamma noise since denominator would be too small")
            return a
