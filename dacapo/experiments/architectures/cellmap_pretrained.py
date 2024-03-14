from funlib.geometry import Coordinate

import torch
from dacapo.experiments.architectures import Architecture
import cellmap_models


class CellMapPretrained(Architecture):
    """ """

    def __init__(self, architecture):
        self.model_type = architecture.model_type
        self.model_name = architecture.model_name
        self.checkpoint = architecture.checkpoint
        self._eval_shape_increase = architecture.eval_shape_increase
        self._input_shape = architecture.input_shape
        self.fit_mode = architecture.fit_mode
        model_loader = getattr(cellmap_models, self.model_type)
        self.model = model_loader.load(f"{self.model_name}/{self.checkpoint}")

    @property
    def input_shape(self) -> Coordinate:
        """
        Abstract method to define the spatial input shape for the neural network architecture.
        The shape should not account for the channels and batch dimensions.

        Returns:
            Coordinate: The spatial input shape.
        """
        if self._input_shape is None:
            return self.model.min_input_shape
        else:
            shape = self.model.round_to_valid_input_shape(
                Coordinate(self._input_shape), self.fit_mode
            )
            self._input_shape = shape
            return shape

    @property
    def eval_shape_increase(self) -> Coordinate:
        """
        Provides information about how much to increase the input shape during prediction.

        Returns:
            Coordinate: An instance representing the amount to increase in each dimension of the input shape.
        """
        if self._eval_shape_increase is None:
            return Coordinate((0,) * self.input_shape.dims)
        else:
            eval_shape = self._eval_shape_increase + self.input_shape
            eval_shape = self.model.round_to_valid_input_shape(
                eval_shape, self.fit_mode
            )
            return Coordinate(eval_shape - self.input_shape)

    @property
    def num_in_channels(self) -> int:
        """
        Abstract method to return number of input channels required by the architecture.

        Returns:
            int: Required number of input channels.
        """
        return self.model.in_channels

    @property
    def num_out_channels(self) -> int:
        """
        Abstract method to return the number of output channels provided by the architecture.

        Returns:
            int: Number of output channels.
        """
        return self.model.classes_out

    @property
    def dims(self) -> int:
        """
        Returns the number of dimensions of the input shape.

        Returns:
            int: The number of dimensions.
        """
        return self.input_shape.dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to define the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.model(x)
