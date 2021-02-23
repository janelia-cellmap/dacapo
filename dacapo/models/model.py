import torch

from typing import List, Optional
import attr


@attr.s
class ModelConfig:
    name: str = attr.ib()
    input_shape: List[int] = attr.ib()
    output_shape: Optional[List[int]] = attr.ib()
    fmaps_in: Optional[int] = attr.ib()
    fmaps_out: int = attr.ib()


class Model(torch.nn.Module):
    """A thin wrapper around torch.nn.Module.

    Attributes:

        input_shape (tuple of int):

              The spatial input shape (i.e., without batch or feature map
              dimension).

        output_shape (tuple of int):

              The spatial output shape (i.e., without batch or feature map
              dimension).

        fmaps_in (int):

              The number of feature maps to be input into this model.

        fmaps_out (int):

              The number of feature maps produced by this model.
    """

    def __init__(self, model_config: ModelConfig):
        super(Model, self).__init__()
        self.input_shape = model_config.input_shape
        self.__output_shape = model_config.output_shape
        self.fmaps_in = model_config.fmaps_in
        self.fmaps_out = model_config.fmaps_out

    @property
    def output_shape(self):
        if self.__output_shape is None:
            self.__output_shape = self.__get_output_shape(
                self.input_shape, self.fmaps_in
            )
        return self.__output_shape

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
