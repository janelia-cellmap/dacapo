# from dacapo.stateless.arraysources.helpers import ArraySource

from dacapo.experiments.datasplits.datasets.arrays import Array

import gunpowder as gp
from gunpowder.profiling import Timing
from gunpowder.array_spec import ArraySpec

import numpy as np


class DaCapoArraySource(gp.BatchProvider):
    """
    A DaCapo Array source node

    Attributes:
        array (Array): The array to be served.
        key (gp.ArrayKey): The key of the array to be served.
    Methods:
        setup(): Set up the provider.
        provide(request): Provides the array for the requested ROI.
    Note:
        This class is a subclass of gunpowder.BatchProvider and is used to
        serve array data to gunpowder pipelines.
    """

    def __init__(self, array: Array, key: gp.ArrayKey):
        """
        Create a DaCapoArraySource object.

        Args:
            array (Array): The array to be served.
            key (gp.ArrayKey): The key of the array to be served.
        Raises:
            TypeError: If key is not of type gp.ArrayKey.
            TypeError: If array is not of type Array.
        Examples:
            >>> from dacapo.experiments.datasplits.datasets.arrays import Array
            >>> from gunpowder import ArrayKey
            >>> array = Array()
            >>> array_source = DaCapoArraySource(array, gp.ArrayKey("ARRAY"))
        """
        self.array = array
        self.array_spec = ArraySpec(
            roi=self.array.roi, voxel_size=self.array.voxel_size
        )
        self.key = key

    def setup(self):
        """
        Adds the key and the array spec to the provider.

        Raises:
            RuntimeError: If the key is already provided.
        Examples:
            >>> array_source.setup()

        """
        self.provides(self.key, self.array_spec.copy())

    def provide(self, request):
        """
        Provides data based on the given request.

        Args:
            request (gp.BatchRequest): The request for data
        Returns:
            gp.Batch: The batch containing the provided data
        Raises:
            ValueError: If the input data contains NaN values
        Examples:
            >>> array_source.provide(request)

        """
        output = gp.Batch()

        timing_provide = Timing(self, "provide")
        timing_provide.start()

        spec = self.array_spec.copy()
        spec.roi = request[self.key].roi

        if spec.roi.empty:
            data = np.zeros((0,) * len(self.array.axes))
        else:
            data = self.array[spec.roi]
        if "c" not in self.array.axes:
            # add a channel dimension
            data = np.expand_dims(data, 0)
        if np.any(np.isnan(data)):
            raise ValueError("INPUT DATA CAN'T BE NAN")
        output[self.key] = gp.Array(data, spec=spec)

        timing_provide.stop()

        output.profiling_stats.add(timing_provide)

        return output
