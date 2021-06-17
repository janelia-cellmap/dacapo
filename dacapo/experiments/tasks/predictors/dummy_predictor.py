from .predictor import Predictor
from dacapo.experiments import Model
import torch


class DummyPredictor(Predictor):

    def __init__(self, embedding_dims):
        self.embedding_dims = embedding_dims

    def create_model(self, architecture, dataset):

        head = torch.nn.Conv3d(
            architecture.num_out_channels,
            self.embedding_dims,
            kernel_size=3)

        return Model(architecture, head)

    def create_target(self, gt):
        pass
