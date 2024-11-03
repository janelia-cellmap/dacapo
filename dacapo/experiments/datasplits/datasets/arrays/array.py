from funlib.geometry import Coordinate, Roi

import numpy as np

from typing import Optional, Dict, Any, List, Iterable
from abc import ABC, abstractmethod


class Array(ABC):
    

    @property
    @abstractmethod
    def attrs(self) -> Dict[str, Any]:
        
        pass

    @property
    @abstractmethod
    def axes(self) -> List[str]:
        
        pass

    @property
    @abstractmethod
    def dims(self) -> int:
        
        pass

    @property
    @abstractmethod
    def voxel_size(self) -> Coordinate:
        
        pass

    @property
    @abstractmethod
    def roi(self) -> Roi:
        
        pass

    @property
    @abstractmethod
    def dtype(self) -> Any:
        
        pass

    @property
    @abstractmethod
    def num_channels(self) -> Optional[int]:
        
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        
        pass

    @property
    @abstractmethod
    def writable(self) -> bool:
        
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

    def _slices(self, roi: Roi) -> Iterable[slice]:
        
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
