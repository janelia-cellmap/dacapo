# from dacapo.stateless.arraysources.helpers import ArraySource

from dacapo.experiments.datasplits.datasets.arrays import Array

import gunpowder as gp
from gunpowder.profiling import Timing
from gunpowder.array_spec import ArraySpec

import numpy as np

class DaCapoArraySource(gp.BatchProvider):
    """A DaCapo Array source node

    Args:

        Array (Array):

            The DaCapo Array to pull data from

        key (``gp.ArrayKey``):

            The key to provide data into
    """

    def __init__(self, array: Array, key: gp.ArrayKey):
        self.array = array
        self.array_spec = ArraySpec(
            roi=self.array.roi, voxel_size=self.array.voxel_size
        )
        self.key = key

    def setup(self):
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        output = gp.Batch()

        timing_provide = Timing(self, "provide")
        timing_provide.start()

        spec = self.array_spec.copy()
        spec.roi = request[self.key].roi

        data = self.array[spec.roi]
        if "c" not in self.array.axes:
            # add a channel dimension
            data = np.expand_dims(data, 0)
        output[self.key] = gp.Array(data, spec=spec)

        timing_provide.stop()

        output.profiling_stats.add(timing_provide)

        return output
