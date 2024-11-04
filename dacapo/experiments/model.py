from dacapo.experiments.architectures.architecture import Architecture

from funlib.geometry import Coordinate

import torch

from typing import Tuple


class Model(torch.nn.Module):
    num_out_channels: int
    num_in_channels: int

    def __init__(
        self,
        architecture: Architecture,
        prediction_head: torch.nn.Module,
        eval_activation: torch.nn.Module | None = None,
    ):
        super().__init__()

        self.architecture = architecture
        self.prediction_head = prediction_head
        self.chain = torch.nn.Sequential(architecture, prediction_head)
        self.num_in_channels = architecture.num_in_channels

        self.input_shape = architecture.input_shape
        self.eval_input_shape = self.input_shape + architecture.eval_shape_increase
        self.num_out_channels, self.output_shape = self.compute_output_shape(
            self.input_shape
        )
        self.eval_activation = eval_activation

        # UPDATE WEIGHT INITIALIZATION TO USE KAIMING
        # TODO: put this somewhere better, there might be
        # conv layers that aren't follwed by relus?
        for _name, layer in self.named_modules():
            if isinstance(layer, torch.nn.modules.conv._ConvNd):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x):
        result = self.chain(x)
        if not self.training and self.eval_activation is not None:
            result = self.eval_activation(result)
        return result

    def compute_output_shape(self, input_shape: Coordinate) -> Tuple[int, Coordinate]:
        return self.__get_output_shape(input_shape, self.num_in_channels)

    def __get_output_shape(
        self, input_shape: Coordinate, in_channels: int
    ) -> Tuple[int, Coordinate]:
        device = torch.device("cpu")
        for parameter in self.parameters():
            device = parameter.device
            break

        dummy_data = torch.zeros((1, in_channels) + input_shape, device=device)
        with torch.no_grad():
            out = self.forward(dummy_data)
        return out.shape[1], Coordinate(out.shape[2:])

    def scale(self, voxel_size: Coordinate) -> Coordinate:
        return self.architecture.scale(voxel_size)
