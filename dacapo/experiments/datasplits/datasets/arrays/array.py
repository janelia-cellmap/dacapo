from funlib.geometry import Coordinate, Roi

import numpy as np

from typing import Optional, Dict, Any, List, Iterable
from abc import ABC, abstractmethod


class Array(ABC):
    @property
    @abstractmethod
    def attrs(self) -> Dict[str, Any]:
        """
        Return a dictionary of metadata attributes stored on this array.
        """
        pass

    @property
    @abstractmethod
    def axes(self) -> List[str]:
        """Returns the axes of this dataset as a string of charactes, as they
        are indexed. Permitted characters are:

            * ``zyx`` for spatial dimensions
            * ``c`` for channels
            * ``s`` for samples
        """
        pass

    @property
    @abstractmethod
    def dims(self) -> int:
        """Returns the number of spatial dimensions."""
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> Coordinate:
        """The size of a voxel in physical units."""
        pass

    @property
    @abstractmethod
    def roi(self) -> Roi:
        """The total ROI of this array, in world units."""
        pass

    @property
    @abstractmethod
    def dtype(self) -> Any:
        """The dtype of this array, in numpy dtypes"""
        pass

    @property
    @abstractmethod
    def num_channels(self) -> Optional[int]:
        """
        The number of channels provided by this dataset.
        Should return None if the channel dimension doesn't exist.
        """
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray[Any, Any]:
        """
        Get a numpy like readable and writable view into this array.
        """
        pass

    @property
    @abstractmethod
    def writable(self) -> bool:
        """
        Can we write to this Array?
        """
        pass

    def __getitem__(self, roi: Roi) -> np.ndarray:
        if not self.roi.contains(roi):
            raise ValueError(f"Cannot fetch data from outside my roi: {self.roi}!")

        assert roi.offset % self.voxel_size == Coordinate(
            (0,) * self.dims
        ), f"Given roi offset: {roi.offset} is not a multiple of voxel_size: {self.voxel_size}"
        assert roi.shape % self.voxel_size == Coordinate(
            (0,) * self.dims
        ), f"Given roi shape: {roi.shape} is not a multiple of voxel_size: {self.voxel_size}"

        slices = tuple(self._slices(roi))

        return self.data[slices]

    def _can_neuroglance(self) -> bool:
        return False

    def _neuroglancer_layer(self):
        pass

    def _slices(self, roi) -> Iterable[slice]:
        offset = (roi.offset - self.roi.offset) / self.voxel_size
        shape = roi.shape / self.voxel_size
        spatial_slices: Dict[str, slice] = {
            a: slice(o, o + s)
            for o, s, a in zip(offset, shape, self.axes[-self.dims :])
        }
        slices: List[slice] = []
        for axis in self.axes:
            if axis == "b" or axis == "c":
                slices.append(slice(None, None))
            else:
                slices.append(spatial_slices[axis])
        return slices
