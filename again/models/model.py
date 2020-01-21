import torch


class Model(torch.nn.Module):
    """A thin wrapper around torch.nn.Module."""

    def __init__(self):
        super(Model, self).__init__()

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def output_size(self, channels, input_size):

        device = 'cpu'
        for parameter in self.parameters():
            device = parameter.device

        dummy_data = torch.zeros(
            (1, channels) + input_size,
            device=device)
        out = self.forward(dummy_data)
        return tuple(out.shape[2:])
