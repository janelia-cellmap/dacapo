import numpy as np

from gunpowder.nodes.batch_filter import BatchFilter
from collections import Iterable

import logging

logger = logging.getLogger(__file__)


class GammaAugment(BatchFilter):
    """
    An Augment to apply gamma noise
    """

    def __init__(self, arrays, gamma_min, gamma_max):
        if not isinstance(arrays, Iterable):
            arrays = [
                arrays,
            ]
        self.arrays = arrays
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        assert self.gamma_max >= self.gamma_min

    def setup(self):
        for array in self.arrays:
            self.updates(array, self.spec[array])

    def process(self, batch, request):
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
        # normalize a
        a_min = a.min()
        a_max = a.max()
        # apply gamma noise
        a = (a - a_min) / (a_max - a_min)
        noisy_a = a**gamma
        # undo normalization
        noisy_a = a * (a_max - a_min) + a_min
        return noisy_a
