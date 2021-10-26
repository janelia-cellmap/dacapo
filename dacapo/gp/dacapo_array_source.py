# from dacapo.stateless.arraysources.helpers import ArraySource

from dacapo.experiments.datasplits.datasets.arrays import Array

import gunpowder as gp
from gunpowder.array_spec import ArraySpec


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

        spec = self.array_spec.copy()
        spec.roi = request[self.key].roi

        output[self.key] = gp.Array(self.array[spec.roi], spec=spec)

        return output
