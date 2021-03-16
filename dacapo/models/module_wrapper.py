import torch
from funlib.geometry import Coordinate

from typing import Optional


class ModuleWrapper(torch.nn.Module):
    """A thin wrapper around torch.nn.Module.

    Attributes:

        context (int):

            The context required by this module. I.e. How much the
            spacial context of the output shrinks relative to the
            input.

        translation_equivariant_step (Coordinate):

            The smallest step that the input can be shifted, s.t.
            the same shift is applied to the output.
            e.g. product of the downsample steps in the UNet.

        fmaps_in (int):

            The number of feature maps to be input into this model.

        fmaps_out (int):

            The number of feature maps produced by this model.
    """

    def __init__(self, context: Optional[Coordinate], fmaps_in, fmaps_out):
        super(ModuleWrapper, self).__init__()
        self.context = context
        self.__shape_map = {"input": {}, "output": {}}
        self.fmaps_in = fmaps_in
        self.fmaps_out = fmaps_out

    def output_shape(self, input_shape: Coordinate):
        if self.context is not None:
            return input_shape - self.context
        else:
            if input_shape not in self.__shape_map["input"]:
                self.__shape_map["input"][input_shape] = self.__get_output_shape(
                    input_shape, self.fmaps_in
                )
            return self.__shape_map["input"][input_shape]

    def input_shape(self, output_shape: Coordinate):
        if self.context is not None:
            return output_shape + self.context
        else:
            raise Exception(
                "Cannot guess input shape from output shape without "
                "knowing this modules needed context"
            )

    def num_parameters(self):
        """Get the total number of paramters of this model."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, checkpoint_name, optimizer):
        """Save this model and the state of the given optimizer."""

        state = {
            "model_state_dict": self.state_dict(),
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(state, checkpoint_name)

    def load(self, checkpoint_name, optimizer=None):
        """Restore this model and optionally the state of the optimizer."""

        checkpoint = torch.load(checkpoint_name)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def __get_output_shape(self, input_shape, fmaps_in):
        """Given the number of input channels and an input size, computes the
        shape of the output."""

        device = "cpu"
        for parameter in self.parameters():
            device = parameter.device
            break

        dummy_data = torch.zeros((1, fmaps_in) + input_shape, device=device)
        out = self.forward(dummy_data)
        return tuple(out.shape[2:])
