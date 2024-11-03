from funlib.geometry import Coordinate

import torch

from abc import ABC, abstractmethod


class Architecture(torch.nn.Module, ABC):
    

    @property
    @abstractmethod
    def input_shape(self) -> Coordinate:
        
        pass

    @property
    def eval_shape_increase(self) -> Coordinate:
        
        return Coordinate((0,) * self.input_shape.dims)

    @property
    @abstractmethod
    def num_in_channels(self) -> int:
        
        pass

    @property
    @abstractmethod
    def num_out_channels(self) -> int:
        
        pass

    @property
    def dims(self) -> int:
        
        return self.input_shape.dims

    def scale(self, input_voxel_size: Coordinate) -> Coordinate:
        
        return input_voxel_size
